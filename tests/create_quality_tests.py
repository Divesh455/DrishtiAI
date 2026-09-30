import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

import cv2

from PIL import Image

from backend.services.quality_service import (
    QualityService
)


def main():

    image_path = input(
        "Enter a good retinal image path: "
    ).strip()

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not load image."
        )

        return

    # Create very blurry image
    blurry = cv2.GaussianBlur(
        image,
        (51, 51),
        0
    )

    output_path = (
        Path("data")
        / "processed"
        / "test_blurry.jpg"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        blurry
    )

    print(
        f"\nBlurry image saved at:"
        f"\n{output_path}"
    )

    # Test quality
    pil_image = Image.open(
        output_path
    ).convert("RGB")

    service = QualityService()

    result = service.check_pil_image(
        pil_image
    )

    print("\n" + "=" * 50)

    print("BLUR TEST")

    print("=" * 50)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Score: "
        f"{result['score']}"
    )

    print(
        f"Blur score: "
        f"{result['checks']['blur']['blur_score']:.2f}"
    )

    print(
        f"\nRecommendation:"
        f"\n{result['recommendation']}"
    )


if __name__ == "__main__":

    main()