import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import cv2

from backend.models.lesion_detector import (
    LesionDetector
)

from training.config import (
    IDRID_IMAGE_DIR
)


def main():

    print("=" * 70)
    print("DrishtiAI - LESION DETECTOR SERVICE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    images = sorted(
        Path(IDRID_IMAGE_DIR).glob("*.jpg")
    )

    if not images:

        raise FileNotFoundError(
            "No IDRiD images found."
        )

    image_path = images[0]

    print(
        f"\nImage: {image_path}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            "Could not read image."
        )

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector = LesionDetector(
        model_path="weights/lesion_model.pth"
    )

    print(
        f"Device: {detector.device}"
    )

    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    result = detector.detect(
        image
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nDetection result:")
    print(
        f"Status: {result['status']}"
    )

    print(
        f"Lesion count: "
        f"{result['lesion_count']}"
    )

    for index, lesion in enumerate(
        result["lesions"],
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

    print("\nDetector test completed.")


if __name__ == "__main__":
    main()