from pathlib import Path
import cv2

from backend.models.lesion_detector import LesionDetector


ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train_images"
)


def find_first_image():
    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ]

    for pattern in extensions:
        matches = list(
            IMAGE_DIR.glob(pattern)
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No fundus images found in {IMAGE_DIR}"
    )


def main():

    print("=" * 70)
    print("CLEMENTP LESION DETECTOR TEST")
    print("=" * 70)

    image_path = find_first_image()

    print()
    print("Test image:")
    print(image_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    print(
        "Image shape:",
        image.shape
    )

    print()
    print("Initializing detector...")

    detector = LesionDetector(
        confidence_threshold=0.40
    )

    print()
    print("Running ClementP inference...")

    result = detector.detect(
        image,
        screening_id="CLEMENTP_TEST"
    )

    print()
    print("=" * 70)
    print("CLEMENTP RESULT")
    print("=" * 70)

    print(
        "Status:",
        result.get("status")
    )

    print(
        "Model:",
        result.get("model")
    )

    print(
        "Architecture:",
        result.get("model_architecture")
    )

    print(
        "Lesion count:",
        result.get("lesion_count")
    )

    print()
    print("Class summary:")

    for name, info in result.get(
        "class_summary",
        {}
    ).items():

        print(
            f"  {info['display_name']}: "
            f"{info['count']}"
        )

    print()
    print("Overlay URL:")
    print(
        result.get("overlay_url")
    )

    print()
    print("Detected lesions:")

    for lesion in result.get(
        "lesions",
        []
    ):

        print(
            f"  - {lesion['display_name']} | "
            f"confidence={lesion['confidence']:.4f} | "
            f"bbox={lesion['bbox']}"
        )

    print()
    print("=" * 70)
    print("CLEMENTP LESION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()