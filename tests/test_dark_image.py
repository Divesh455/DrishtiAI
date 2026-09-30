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
        "Enter retinal image path: "
    ).strip()

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not load image."
        )

        return

    # Make image dark
    dark = (
        image.astype("float32")
        * 0.20
    )

    dark = dark.clip(
        0,
        255
    ).astype("uint8")

    output_path = (
        Path("data")
        / "processed"
        / "test_dark.jpg"
    )

    cv2.imwrite(
        str(output_path),
        dark
    )

    print(
        f"Dark image saved at:"
        f"\n{output_path}"
    )

    pil_image = Image.open(
        output_path
    ).convert("RGB")

    service = QualityService()

    result = service.check_pil_image(
        pil_image
    )

    print("\n" + "=" * 50)

    print("DARK IMAGE TEST")

    print("=" * 50)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Brightness: "
        f"{result['checks']['brightness']['brightness']:.2f}"
    )

    print(
        f"\nRecommendation:"
        f"\n{result['recommendation']}"
    )


if __name__ == "__main__":

    main()