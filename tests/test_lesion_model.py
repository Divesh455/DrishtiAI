import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from backend.models.lesion_detector import (
    create_lesion_model
)


def main():

    print("=" * 70)
    print("DrishtiAI - LESION MODEL TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model = create_lesion_model(
        num_classes=5,
        device=device
    )

    # Fake retinal image
    test_image = torch.randn(
        1,
        3,
        512,
        512
    ).to(device)

    print(
        f"Input shape: {test_image.shape}"
    )

    with torch.no_grad():

        output = model(
            test_image
        )

    print(
        f"Output shape: {output.shape}"
    )

    expected_shape = (
        1,
        5,
        512,
        512
    )

    print(
        f"Expected shape: {expected_shape}"
    )

    if tuple(output.shape) == expected_shape:

        print(
            "\nSUCCESS: U-Net output shape is correct."
        )

    else:

        print(
            "\nERROR: Unexpected output shape."
        )

    print("\nModel test completed.")


if __name__ == "__main__":
    main()