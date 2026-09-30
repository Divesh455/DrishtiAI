import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

from PIL import Image

from backend.services.quality_service import (
    QualityService
)


def main():

    image_path = input(
        "Enter retinal image path: "
    ).strip()

    print("\nLoading image...")

    try:

        image = Image.open(
            image_path
        )

        image = image.convert(
            "RGB"
        )

    except Exception as e:

        print(
            f"Failed to load image: {e}"
        )

        return

    quality_service = QualityService()

    result = quality_service.check_pil_image(
        image
    )

    print("\n" + "=" * 60)

    print(
        "DrishtiAI - IMAGE QUALITY RESULT"
    )

    print("=" * 60)

    print(
        f"\nStatus : "
        f"{result['status']}"
    )

    print(
        f"Score  : "
        f"{result['score']:.2f}"
    )

    print("\nChecks:")

    checks = result["checks"]

    print(
        f"Resolution       : "
        f"{checks['resolution']['passed']}"
    )

    print(
        f"Blur             : "
        f"{checks['blur']['passed']}"
    )

    print(
        f"Brightness       : "
        f"{checks['brightness']['passed']}"
    )

    print(
        f"Contrast         : "
        f"{checks['contrast']['passed']}"
    )

    print(
        f"Retina Visibility: "
        f"{checks['retina_visibility']['passed']}"
    )

    print("\nDetailed values:")

    print(
        f"Image size       : "
        f"{checks['resolution']['width']} x "
        f"{checks['resolution']['height']}"
    )

    print(
        f"Blur score       : "
        f"{checks['blur']['blur_score']:.2f}"
    )

    print(
        f"Brightness       : "
        f"{checks['brightness']['brightness']:.2f}"
    )

    print(
        f"Contrast         : "
        f"{checks['contrast']['contrast']:.2f}"
    )

    print(
        f"Visibility ratio : "
        f"{checks['retina_visibility']['visibility_ratio']:.2f}"
    )

    print("\nRecommendation:")

    print(
        result["recommendation"]
    )

    print("=" * 60)


if __name__ == "__main__":

    main()