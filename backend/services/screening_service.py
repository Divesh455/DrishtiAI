from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch


# =============================================================
# PROJECT ROOT
# =============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

if str(BASE_DIR) not in sys.path:

    sys.path.append(
        str(BASE_DIR)
    )


# =============================================================
# PROJECT IMPORTS
# =============================================================

from backend.models.dr_classifier import (
    create_model
)

from backend.models.quality_checker import (
    ImageQualityChecker
)

from backend.models.lesion_detector import (
    LesionDetector
)

from backend.services.jev_service import (
    EvidenceService
)

from backend.services.output_service import (
    OutputService
)

from backend.xai.gradcam import (
    GradCAMExplainer
)

from training.config import (
    DEVICE,
    NUM_CLASSES,
    MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES,
)


class ScreeningService:

    """
    Complete DrishtiAI local AI screening pipeline.

    Pipeline:

        Retinal Image
              ↓
        Image Quality
              ↓
        DR Classification
              ↓
        Grad-CAM++
              ↓
        Lesion Detection
              ↓
        Structured Evidence
              ↓
        Screening Result
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        dr_model_path=MODEL_PATH,
        lesion_model_path=(
            BASE_DIR /
            "weights" /
            "lesion_model.pth"
        )
    ):

        self.device = DEVICE

        print(
            "\nInitializing "
            "DrishtiAI Screening Service..."
        )

        # =====================================================
        # 1. DR CLASSIFIER
        # =====================================================

        print(
            "\nLoading DR classification model..."
        )

        self.dr_model = create_model(
            num_classes=NUM_CLASSES,
            device=self.device
        )

        checkpoint = torch.load(
            dr_model_path,
            map_location=self.device
        )

        if (
            isinstance(
                checkpoint,
                dict
            )
            and
            "model_state_dict"
            in checkpoint
        ):

            self.dr_model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            self.dr_model.load_state_dict(
                checkpoint
            )

        self.dr_model.eval()

        print(
            "DR classification model loaded."
        )

        # =====================================================
        # 2. GRAD-CAM++
        # =====================================================

        print(
            "\nLoading Grad-CAM++ explainer..."
        )

        self.gradcam = GradCAMExplainer(
            model=self.dr_model,
            device=self.device,
            image_size=IMAGE_SIZE
        )

        print(
            "Grad-CAM++ explainer loaded."
        )

        # =====================================================
        # 3. IMAGE QUALITY
        # =====================================================

        print(
            "\nLoading image quality checker..."
        )

        self.quality_checker = (
            ImageQualityChecker()
        )

        print(
            "Image quality checker loaded."
        )

        # =====================================================
        # 4. LESION DETECTOR
        # =====================================================

        print(
            "\nLoading lesion detection model..."
        )

        self.lesion_detector = (
            LesionDetector(
                model_path=str(
                    lesion_model_path
                ),
                image_size=512,
                device=self.device
            )
        )

        print(
            "Lesion detection model loaded."
        )

        # =====================================================
        # 5. EVIDENCE SERVICE
        # =====================================================

        print(
            "\nLoading evidence service..."
        )

        self.evidence_service = (
            EvidenceService()
        )

        print(
            "Evidence service loaded."
        )

        # =====================================================
        # 6. OUTPUT SERVICE
        # =====================================================

        print(
            "\nLoading output service..."
        )

        self.output_service = (
            OutputService()
        )

        print(
            "Output service loaded."
        )

        print(
            "\nScreening service ready."
        )

    # =========================================================
    # CONVERT TO RGB NUMPY
    # =========================================================

    def _to_rgb_numpy(
        self,
        image
    ):

        """
        Convert PIL Image or OpenCV image
        to RGB NumPy format.
        """

        # -----------------------------------------------------
        # PIL
        # -----------------------------------------------------

        if isinstance(
            image,
            Image.Image
        ):

            image = image.convert(
                "RGB"
            )

            return np.array(
                image
            )

        # -----------------------------------------------------
        # NUMPY
        # -----------------------------------------------------

        if isinstance(
            image,
            np.ndarray
        ):

            if image.ndim == 3:

                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

            if image.ndim == 2:

                return image

        # -----------------------------------------------------
        # INVALID
        # -----------------------------------------------------

        raise TypeError(
            "Image must be a PIL Image "
            "or NumPy array."
        )

    # =========================================================
    # DR CLASSIFIER PREPROCESSING
    # =========================================================

    def _preprocess_for_classifier(
        self,
        image
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        pil_image = (
            Image.fromarray(
                rgb_image
            ).convert(
                "RGB"
            )
        )

        # -----------------------------------------------------
        # Resize
        # -----------------------------------------------------

        pil_image = (
            pil_image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )
        )

        # -----------------------------------------------------
        # Normalize 0-1
        # -----------------------------------------------------

        image_array = (
            np.array(
                pil_image
            ).astype(
                np.float32
            ) / 255.0
        )

        # -----------------------------------------------------
        # ImageNet normalization
        # -----------------------------------------------------

        mean = np.array(
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=np.float32
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=np.float32
        )

        image_array = (
            image_array - mean
        ) / std

        # -----------------------------------------------------
        # HWC -> CHW
        # -----------------------------------------------------

        image_array = (
            np.transpose(
                image_array,
                (2, 0, 1)
            )
        )

        # -----------------------------------------------------
        # NumPy -> Tensor
        # -----------------------------------------------------

        tensor = torch.tensor(
            image_array,
            dtype=torch.float32
        )

        # -----------------------------------------------------
        # Add batch
        # -----------------------------------------------------

        tensor = (
            tensor.unsqueeze(0)
        )

        return tensor.to(
            self.device
        )

    # =========================================================
    # DR CLASSIFICATION
    # =========================================================

    def _classify_dr(
        self,
        image
    ):

        input_tensor = (
            self._preprocess_for_classifier(
                image
            )
        )

        with torch.no_grad():

            output = (
                self.dr_model(
                    input_tensor
                )
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1
                )
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        grade = int(
            prediction.item()
        )

        confidence = float(
            confidence.item()
        )

        probability_values = (
            probabilities[0]
            .cpu()
            .numpy()
            .tolist()
        )

        probability_dict = {}

        for index in range(
            NUM_CLASSES
        ):

            probability_dict[
                CLASS_NAMES[index]
            ] = round(
                float(
                    probability_values[
                        index
                    ]
                ),
                4
            )

        return {

            "grade":
                grade,

            "label":
                CLASS_NAMES[
                    grade
                ],

            "confidence":
                round(
                    confidence,
                    4
                ),

            "probabilities":
                probability_dict
        }

    # =========================================================
    # IMAGE QUALITY
    # =========================================================

    def _check_quality(
        self,
        image
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        return (
            self.quality_checker.analyze(
                rgb_image
            )
        )

    # =========================================================
    # GRAD-CAM++
    # =========================================================

    def _generate_gradcam(
        self,
        image,
        target_class
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        pil_image = (
            Image.fromarray(
                rgb_image
            ).convert(
                "RGB"
            )
        )

        return (
            self.gradcam.generate(
                image=pil_image,
                target_class=target_class
            )
        )

    # =========================================================
    # SAVE GRAD-CAM RESULTS
    # =========================================================

    def _save_gradcam_results(
        self,
        gradcam_result,
        image_name,
        heatmap_dir,
        overlay_dir
    ):

        """
        Save Grad-CAM++ results into
        the current screening folder.
        """

        return (
            self.gradcam.save_results(
                result=gradcam_result,

                output_name=image_name,

                heatmap_dir=heatmap_dir,

                overlay_dir=overlay_dir
            )
        )

    # =========================================================
    # LESION DETECTION
    # =========================================================

    def _detect_lesions(
        self,
        image
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        # LesionDetector expects BGR

        bgr_image = (
            cv2.cvtColor(
                rgb_image,
                cv2.COLOR_RGB2BGR
            )
        )

        return (
            self.lesion_detector.detect(
                bgr_image
            )
        )

    # =========================================================
    # BUILD EVIDENCE
    # =========================================================

    def _build_evidence(
        self,
        classification_result,
        quality_result,
        lesion_result
    ):

        return (
            self.evidence_service.build_evidence(
                dr_result=
                    classification_result,

                quality_result=
                    quality_result,

                lesion_result=
                    lesion_result
            )
        )

    # =========================================================
    # COMPLETE SCREENING
    # =========================================================

    def screen(
        self,
        image,
        screening_id=None
    ):

        """
        Run complete screening.

        Steps:

            1. Quality
            2. DR Classification
            3. Grad-CAM++
            4. Lesion Detection
            5. Evidence
        """

        # =====================================================
        # CREATE SCREENING ID
        # =====================================================

        if screening_id is None:

            screening_id = (
                self.output_service
                .create_screening_id()
            )

        # =====================================================
        # CREATE OUTPUT DIRECTORIES
        # =====================================================

        output_dirs = (
            self.output_service
            .create_screening_folder(
                screening_id
            )
        )

        # =====================================================
        # STEP 1 — QUALITY
        # =====================================================

        print(
            "\n[1/5] Checking image quality..."
        )

        quality_result = (
            self._check_quality(
                image
            )
        )

        print(
            f"Quality status: "
            f"{quality_result['status']}"
        )

        print(
            f"Quality score: "
            f"{quality_result['score']}"
        )

        # =====================================================
        # STOP IF POOR
        # =====================================================

        if (
            quality_result[
                "status"
            ] == "poor"
        ):

            print(
                "\nImage rejected because "
                "quality is insufficient."
            )

            return {

                "screening_id":
                    screening_id,

                "status":
                    "rejected",

                "message":
                    (
                        "Image quality is insufficient "
                        "for reliable AI screening."
                    ),

                "quality":
                    quality_result,

                "classification":
                    None,

                "explainability":
                    None,

                "lesions":
                    None,

                "evidence":
                    None
            }

        # =====================================================
        # STEP 2 — DR CLASSIFICATION
        # =====================================================

        print(
            "\n[2/5] Running DR classification..."
        )

        classification_result = (
            self._classify_dr(
                image
            )
        )

        print(
            f"DR Grade: "
            f"{classification_result['grade']}"
        )

        print(
            f"DR Label: "
            f"{classification_result['label']}"
        )

        print(
            f"Confidence: "
            f"{classification_result['confidence']}"
        )

        # =====================================================
        # STEP 3 — GRAD-CAM++
        # =====================================================

        print(
            "\n[3/5] Generating "
            "Grad-CAM++ explanation..."
        )

        gradcam_result = (
            self._generate_gradcam(
                image=image,

                target_class=(
                    classification_result[
                        "grade"
                    ]
                )
            )
        )

        print(
            "Grad-CAM++ explanation generated."
        )

        # -----------------------------------------------------
        # Save Grad-CAM++ outputs
        # -----------------------------------------------------

        saved_gradcam = (
            self._save_gradcam_results(

                gradcam_result=
                    gradcam_result,

                image_name=
                    screening_id,

                heatmap_dir=
                    output_dirs[
                        "heatmap_dir"
                    ],

                overlay_dir=
                    output_dirs[
                        "overlay_dir"
                    ]
            )
        )

        print(
            f"Heatmap saved: "
            f"{saved_gradcam['heatmap_path']}"
        )

        print(
            f"Overlay saved: "
            f"{saved_gradcam['overlay_path']}"
        )

        # =====================================================
        # STEP 4 — LESION DETECTION
        # =====================================================

        print(
            "\n[4/5] Running lesion detection..."
        )

        lesion_result = (
            self._detect_lesions(
                image
            )
        )

        print(
            f"Lesion count: "
            f"{lesion_result['lesion_count']}"
        )

        # =====================================================
        # STEP 5 — EVIDENCE
        # =====================================================

        print(
            "\n[5/5] Building structured evidence..."
        )

        evidence = (
            self._build_evidence(

                classification_result=
                    classification_result,

                quality_result=
                    quality_result,

                lesion_result=
                    lesion_result
            )
        )

        print(
            "Structured evidence created."
        )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        return {

            "screening_id":
                screening_id,

            "status":
                "completed",

            # -------------------------------------------------
            # Quality
            # -------------------------------------------------

            "quality":
                quality_result,

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------

            "classification":
                classification_result,

            # -------------------------------------------------
            # Explainability
            # -------------------------------------------------

            "explainability": {

                "method":
                    "Grad-CAM++",

                "target_class":
                    gradcam_result[
                        "target_class"
                    ],

                "confidence":
                    gradcam_result[
                        "confidence"
                    ],

                "explanation":
                    gradcam_result[
                        "explanation"
                    ],

                "heatmap_path":
                    saved_gradcam[
                        "heatmap_path"
                    ],

                "overlay_path":
                    saved_gradcam[
                        "overlay_path"
                    ]
            },

            # -------------------------------------------------
            # Lesions
            # -------------------------------------------------

            "lesions":
                lesion_result,

            # -------------------------------------------------
            # Structured evidence
            # -------------------------------------------------

            "evidence":
                evidence
        }