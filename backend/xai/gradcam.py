from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)


class GradCAMExplainer:

    """
    Grad-CAM++ explanation service for the
    EfficientNet diabetic retinopathy classifier.
    """

    def __init__(
        self,
        model,
        device,
        image_size=224
    ):

        self.model = model
        self.device = device
        self.image_size = image_size

        # =====================================================
        # TARGET LAYER
        # =====================================================

        self.target_layers = [
            self.model.model.features[-1]
        ]

        # =====================================================
        # IMAGE PREPROCESSING
        # =====================================================

        self.transform = transforms.Compose(
            [

                transforms.Resize(
                    (
                        image_size,
                        image_size
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],

                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )

    # =========================================================
    # GENERATE GRAD-CAM++
    # =========================================================

    def generate(
        self,
        image: Image.Image,
        target_class=None
    ):

        # -----------------------------------------------------
        # Convert image to RGB
        # -----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        # Keep original image
        original_image = image.copy()

        # -----------------------------------------------------
        # Resize image for visualization
        # -----------------------------------------------------

        resized_image = image.resize(
            (
                self.image_size,
                self.image_size
            )
        )

        # -----------------------------------------------------
        # Convert to NumPy float image
        # -----------------------------------------------------

        rgb_image = (
            np.array(
                resized_image
            ).astype(
                np.float32
            ) / 255.0
        )

        # -----------------------------------------------------
        # Create model tensor
        # -----------------------------------------------------

        input_tensor = self.transform(
            image
        )

        input_tensor = (
            input_tensor.unsqueeze(0)
        )

        input_tensor = input_tensor.to(
            self.device
        )

        # =====================================================
        # MODEL PREDICTION
        # =====================================================

        self.model.eval()

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = (
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )

        # =====================================================
        # SELECT TARGET CLASS
        # =====================================================

        if target_class is None:

            target_class = (
                predicted_class
            )

        # =====================================================
        # GRAD-CAM++
        # =====================================================

        targets = [
            ClassifierOutputTarget(
                target_class
            )
        ]

        with GradCAMPlusPlus(
            model=self.model,
            target_layers=self.target_layers
        ) as cam:

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )

            grayscale_cam = (
                grayscale_cam[0]
            )

        # =====================================================
        # CREATE OVERLAY
        # =====================================================

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # =====================================================
        # TARGET CLASS CONFIDENCE
        # =====================================================

        confidence = float(
            probabilities[
                0,
                target_class
            ].item()
        )

        # =====================================================
        # RESULT
        # =====================================================

        return {

            "predicted_class":
                int(predicted_class),

            "target_class":
                int(target_class),

            "confidence":
                confidence,

            "explanation":
                (
                    "Grad-CAM++ highlights retinal "
                    "regions that contributed strongly "
                    "to the selected prediction."
                ),

            "heatmap":
                grayscale_cam,

            "overlay":
                visualization,

            "original_image":
                original_image
        }

    # =========================================================
    # SAVE GRAD-CAM RESULTS
    # =========================================================

    def save_results(
        self,
        result,
        output_name,
        heatmap_dir=None,
        overlay_dir=None
    ):

        """
        Save Grad-CAM++ heatmap and overlay.

        Optional directories allow the screening service
        to create unique output folders for each screening.
        """

        # =====================================================
        # DEFAULT DIRECTORIES
        # =====================================================

        if heatmap_dir is None:

            heatmap_dir = Path(
                "backend/outputs/heatmaps"
            )

        else:

            heatmap_dir = Path(
                heatmap_dir
            )

        if overlay_dir is None:

            overlay_dir = Path(
                "backend/outputs/overlays"
            )

        else:

            overlay_dir = Path(
                overlay_dir
            )

        # =====================================================
        # CREATE DIRECTORIES
        # =====================================================

        heatmap_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        overlay_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =====================================================
        # HEATMAP
        # =====================================================

        heatmap = (
            result["heatmap"] * 255
        ).astype(
            np.uint8
        )

        heatmap_path = (
            heatmap_dir /
            f"{output_name}_heatmap.jpg"
        )

        cv2.imwrite(
            str(heatmap_path),
            heatmap
        )

        # =====================================================
        # OVERLAY
        # =====================================================

        overlay = cv2.cvtColor(
            result["overlay"],
            cv2.COLOR_RGB2BGR
        )

        overlay_path = (
            overlay_dir /
            f"{output_name}_overlay.jpg"
        )

        cv2.imwrite(
            str(overlay_path),
            overlay
        )

        # =====================================================
        # RETURN PATHS
        # =====================================================

        return {

            "heatmap_path":
                str(heatmap_path),

            "overlay_path":
                str(overlay_path)
        }