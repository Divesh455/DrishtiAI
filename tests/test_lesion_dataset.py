import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from training.config import (
    IDRID_IMAGE_DIR,
    IDRID_MICROANEURYSM_DIR,
    IDRID_HEMORRHAGE_DIR,
    IDRID_HARD_EXUDATE_DIR,
    IDRID_SOFT_EXUDATE_DIR,
)

from training.lesion_dataset import (
    IDRiDLesionDataset
)


def get_complete_images():

    image_files = sorted(
        Path(IDRID_IMAGE_DIR).glob("*.jpg")
    )

    complete_images = []

    for image_path in image_files:

        image_id = image_path.stem

        ma = (
            Path(IDRID_MICROANEURYSM_DIR)
            / f"{image_id}_MA.tif"
        )

        he = (
            Path(IDRID_HEMORRHAGE_DIR)
            / f"{image_id}_HE.tif"
        )

        ex = (
            Path(IDRID_HARD_EXUDATE_DIR)
            / f"{image_id}_EX.tif"
        )

        se = (
            Path(IDRID_SOFT_EXUDATE_DIR)
            / f"{image_id}_SE.tif"
        )

        if (
            ma.exists()
            and he.exists()
            and ex.exists()
            and se.exists()
        ):

            complete_images.append(
                image_path
            )

    return complete_images


def main():

    print("=" * 70)
    print("DrishtiAI - LESION DATASET TEST")
    print("=" * 70)

    images = get_complete_images()

    print(
        f"\nComplete annotated images: {len(images)}"
    )

    if not images:

        print(
            "\nERROR: No complete images found."
        )

        return

    print("\nFirst images:")

    for image in images[:10]:

        print(
            f"  {image.name}"
        )

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = IDRiDLesionDataset(
        images,
        image_size=512,
        augment=False
    )

    print(
        f"\nDataset length: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Read first sample
    # --------------------------------------------------------

    image, mask = dataset[0]

    print(
        f"\nImage tensor shape: {image.shape}"
    )

    print(
        f"Mask tensor shape: {mask.shape}"
    )

    print(
        f"Image data type: {image.dtype}"
    )

    print(
        f"Mask data type: {mask.dtype}"
    )

    unique_classes = mask.unique().tolist()

    print(
        f"\nClasses present: {unique_classes}"
    )

    print(
        "\nExpected classes:"
    )

    print("  0 = Background")
    print("  1 = Microaneurysm")
    print("  2 = Hemorrhage")
    print("  3 = Hard Exudate")
    print("  4 = Soft Exudate")

    print("\nDataset test completed.")


if __name__ == "__main__":
    main()