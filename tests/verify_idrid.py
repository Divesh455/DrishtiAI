import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image

from training.config import (
    IDRID_IMAGE_DIR,
    IDRID_MICROANEURYSM_DIR,
    IDRID_HEMORRHAGE_DIR,
    IDRID_HARD_EXUDATE_DIR,
    IDRID_SOFT_EXUDATE_DIR,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_EXTENSIONS = {".tif", ".tiff", ".png"}


def get_files(directory, extensions):
    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(
        [
            file
            for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in extensions
        ]
    )


def print_directory(name, directory, extensions):

    files = get_files(directory, extensions)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Directory : {directory}")
    print(f"Files     : {len(files)}")

    for file in files[:5]:
        print(f"  {file.name}")

    if len(files) > 5:
        print("  ...")

    return files


def get_image_id(filename):

    # IDRiD_01.jpg -> IDRiD_01
    return Path(filename).stem


def get_mask_id(filename):

    # IDRiD_01_MA.tif -> IDRiD_01
    # IDRiD_01_HE.tif -> IDRiD_01
    # IDRiD_01_EX.tif -> IDRiD_01
    # IDRiD_01_SE.tif -> IDRiD_01

    stem = Path(filename).stem

    for suffix in ["_MA", "_HE", "_EX", "_SE"]:

        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem


def check_matching_images(
    image_files,
    mask_files,
    lesion_name
):

    image_ids = {
        get_image_id(file.name)
        for file in image_files
    }

    mask_ids = {
        get_mask_id(file.name)
        for file in mask_files
    }

    matching = image_ids & mask_ids

    missing_masks = image_ids - mask_ids

    extra_masks = mask_ids - image_ids

    print(f"\n{lesion_name} matching")
    print("-" * 60)

    print(f"Images available : {len(image_ids)}")
    print(f"Masks available  : {len(mask_ids)}")
    print(f"Matching         : {len(matching)}")

    if missing_masks:

        print(
            f"Missing masks    : {len(missing_masks)}"
        )

        print("Examples:")

        for item in sorted(missing_masks)[:5]:
            print(f"  {item}")

    if extra_masks:

        print(
            f"Extra masks      : {len(extra_masks)}"
        )

        for item in sorted(extra_masks)[:5]:
            print(f"  {item}")

    if not missing_masks and not extra_masks:
        print("Status           : OK")


def check_image_size(image_files):

    if not image_files:
        print("\nNo images found.")
        return

    print("\nImage size check")
    print("-" * 60)

    for file in image_files[:5]:

        try:

            with Image.open(file) as image:

                print(
                    f"{file.name}: "
                    f"{image.size[0]} x "
                    f"{image.size[1]} "
                    f"({image.mode})"
                )

        except Exception as error:

            print(
                f"{file.name}: ERROR - {error}"
            )


def main():

    print("=" * 70)
    print("DrishtiAI - IDRiD DATASET VERIFICATION")
    print("=" * 70)

    print("\nIDRiD root:")
    print(IDRID_IMAGE_DIR.parent.parent)

    # --------------------------------------------------------
    # Original images
    # --------------------------------------------------------

    image_files = print_directory(
        "Original Training Images",
        IDRID_IMAGE_DIR,
        IMAGE_EXTENSIONS
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    microaneurysm_files = print_directory(
        "Microaneurysm Masks",
        IDRID_MICROANEURYSM_DIR,
        MASK_EXTENSIONS
    )

    hemorrhage_files = print_directory(
        "Hemorrhage Masks",
        IDRID_HEMORRHAGE_DIR,
        MASK_EXTENSIONS
    )

    hard_exudate_files = print_directory(
        "Hard Exudate Masks",
        IDRID_HARD_EXUDATE_DIR,
        MASK_EXTENSIONS
    )

    soft_exudate_files = print_directory(
        "Soft Exudate Masks",
        IDRID_SOFT_EXUDATE_DIR,
        MASK_EXTENSIONS
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    check_matching_images(
        image_files,
        microaneurysm_files,
        "Microaneurysm"
    )

    check_matching_images(
        image_files,
        hemorrhage_files,
        "Hemorrhage"
    )

    check_matching_images(
        image_files,
        hard_exudate_files,
        "Hard Exudate"
    )

    check_matching_images(
        image_files,
        soft_exudate_files,
        "Soft Exudate"
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    check_image_size(image_files)

    print("\n" + "=" * 70)
    print("IDRiD verification completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()