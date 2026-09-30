import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import cv2

from backend.services.screening_service import ScreeningService
from training.config import IDRID_IMAGE_DIR


def main():

    print("=" * 70)
    print("DrishtiAI - COMPLETE SCREENING PIPELINE TEST")
    print("=" * 70)

    # =========================================================
    # FIND TEST IMAGE
    # =========================================================

    images = sorted(
        Path(IDRID_IMAGE_DIR).glob("*.jpg")
    )

    if not images:
        raise FileNotFoundError(
            "No IDRiD images found."
        )

    image_path = images[0]

    print(
        f"\nTest image:"
    )

    print(
        image_path
    )

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            "Could not read test image."
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # =========================================================
    # CREATE SCREENING SERVICE
    # =========================================================

    print(
        "\nInitializing screening service..."
    )

    service = ScreeningService()

    # =========================================================
    # RUN COMPLETE PIPELINE
    # =========================================================

    print(
        "\nRunning screening pipeline..."
    )

    result = service.screen(
        image
    )

    # =========================================================
    # SCREENING RESULT
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SCREENING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nStatus: "
        f"{result['status']}"
    )

    # =========================================================
    # IMAGE QUALITY
    # =========================================================

    print(
        "\n--- IMAGE QUALITY ---"
    )

    quality = result.get(
        "quality"
    )

    print(
        f"Status: "
        f"{quality.get('status')}"
    )

    print(
        f"Score: "
        f"{quality.get('score')}"
    )

    # =========================================================
    # CHECK IF SCREENING WAS REJECTED
    # =========================================================

    if result["status"] == "rejected":

        print(
            "\nScreening stopped because "
            "image quality was insufficient."
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "PIPELINE TEST COMPLETED"
        )

        print(
            "=" * 70
        )

        return

    # =========================================================
    # DR CLASSIFICATION
    # =========================================================

    print(
        "\n--- DR CLASSIFICATION ---"
    )

    classification = result.get(
        "classification"
    )

    if classification:

        print(
            f"Grade: "
            f"{classification['grade']}"
        )

        print(
            f"Label: "
            f"{classification['label']}"
        )

        print(
            f"Confidence: "
            f"{classification['confidence']}"
        )

        print(
            "\nProbabilities:"
        )

        for (
            label,
            probability
        ) in classification[
            "probabilities"
        ].items():

            print(
                f"  {label}: "
                f"{probability}"
            )

    # =========================================================
    # GRAD-CAM++
    # =========================================================

    print(
        "\n--- GRAD-CAM++ EXPLANATION ---"
    )

    explanation = result.get(
        "explainability"
    )

    if explanation:

        print(
            f"Method: "
            f"{explanation['method']}"
        )

        print(
            f"Target class: "
            f"{explanation['target_class']}"
        )

        print(
            f"Confidence: "
            f"{explanation['confidence']}"
        )

        print(
            f"Heatmap: "
            f"{explanation['heatmap_path']}"
        )

        print(
            f"Overlay: "
            f"{explanation['overlay_path']}"
        )

        print(
            f"Explanation: "
            f"{explanation['explanation']}"
        )

    else:

        print(
            "No Grad-CAM result."
        )

    # =========================================================
    # LESION DETECTION
    # =========================================================

    print(
        "\n--- LESION DETECTION ---"
    )

    lesion_result = result.get(
        "lesions"
    )

    if lesion_result:

        print(
            f"Status: "
            f"{lesion_result['status']}"
        )

        print(
            f"Lesion count: "
            f"{lesion_result['lesion_count']}"
        )

        for (
            index,
            lesion
        ) in enumerate(
            lesion_result["lesions"],
            start=1
        ):

            print(
                f"\nLesion {index}"
            )

            print(
                f"  Type: "
                f"{lesion['type']}"
            )

            print(
                f"  Confidence: "
                f"{lesion['confidence']}"
            )

            print(
                f"  Bounding box: "
                f"{lesion['bbox']}"
            )

            print(
                f"  Area: "
                f"{lesion['area_pixels_512']}"
            )

    else:

        print(
            "No lesion result."
        )

    # =========================================================
    # STRUCTURED AI EVIDENCE
    # =========================================================

    print(
        "\n--- STRUCTURED AI EVIDENCE ---"
    )

    evidence = result.get(
        "evidence"
    )

    if evidence:

        print(
            "\nDR classification evidence:"
        )

        print(
            evidence.get(
                "dr_classification"
            )
        )

        print(
            "\nImage quality evidence:"
        )

        print(
            evidence.get(
                "image_quality"
            )
        )

        print(
            "\nLesion evidence:"
        )

        print(
            evidence.get(
                "lesion_evidence"
            )
        )

    else:

        print(
            "No structured evidence."
        )

    # =========================================================
    # FINAL
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE TEST COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()