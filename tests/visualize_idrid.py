import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from training.config import (
    IDRID_IMAGE_DIR,
    IDRID_MICROANEURYSM_DIR,
    IDRID_HEMORRHAGE_DIR,
    IDRID_HARD_EXUDATE_DIR,
    IDRID_SOFT_EXUDATE_DIR,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("backend/outputs/idrid_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND MASK
# ============================================================

def find_mask(directory, image_stem, suffix):

    possible_extensions = [
        ".tif",
        ".tiff",
        ".png",
    ]

    for extension in possible_extensions:

        path = (
            Path(directory)
            / f"{image_stem}_{suffix}{extension}"
        )

        if path.exists():
            return path

    return None


# ============================================================
# FIND IMAGE WITH ALL AVAILABLE MASKS
# ============================================================

def find_first_image():

    image_files = sorted(
        Path(IDRID_IMAGE_DIR).glob("*.jpg")
    )

    if not image_files:

        image_files = sorted(
            Path(IDRID_IMAGE_DIR).glob("*.jpeg")
        )

    for image_path in image_files:

        image_stem = image_path.stem

        ma = find_mask(
            IDRID_MICROANEURYSM_DIR,
            image_stem,
            "MA"
        )

        he = find_mask(
            IDRID_HEMORRHAGE_DIR,
            image_stem,
            "HE"
        )

        ex = find_mask(
            IDRID_HARD_EXUDATE_DIR,
            image_stem,
            "EX"
        )

        se = find_mask(
            IDRID_SOFT_EXUDATE_DIR,
            image_stem,
            "SE"
        )

        # Prefer an image that has all four masks
        if ma and he and ex and se:
            return image_path

    # If no image has all four masks,
    # use the first available image.
    if image_files:
        return image_files[0]

    return None


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path, target_size):

    if path is None:

        return np.zeros(
            (target_size[1], target_size[0]),
            dtype=np.uint8
        )

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise ValueError(
            f"Could not read mask: {path}"
        )

    if (
        mask.shape[1] != target_size[0]
        or mask.shape[0] != target_size[1]
    ):

        mask = cv2.resize(
            mask,
            target_size,
            interpolation=cv2.INTER_NEAREST
        )

    # Convert all non-zero pixels to 1
    mask = (mask > 0).astype(np.uint8)

    return mask


# ============================================================
# APPLY MASK OVERLAY
# ============================================================

def apply_mask(
    image,
    mask,
    color,
    alpha=0.45
):

    result = image.copy()

    mask_bool = mask > 0

    overlay = np.zeros_like(image)

    overlay[:, :] = color

    result[mask_bool] = (
        image[mask_bool] * (1 - alpha)
        + overlay[mask_bool] * alpha
    ).astype(np.uint8)

    return result


# ============================================================
# ADD LABEL
# ============================================================

def add_label(image, text):

    output = image.copy()

    cv2.rectangle(
        output,
        (10, 10),
        (330, 55),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DrishtiAI - IDRiD MASK VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_first_image()

    if image_path is None:

        print(
            "\nERROR: No IDRiD training images found."
        )

        print(
            "\nExpected folder:"
        )

        print(IDRID_IMAGE_DIR)

        return

    print("\nImage selected:")
    print(image_path)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    height, width = image.shape[:2]

    target_size = (
        width,
        height
    )

    image_stem = image_path.stem

    print(
        f"\nImage ID: {image_stem}"
    )

    print(
        f"Image size: {width} x {height}"
    )

    # --------------------------------------------------------
    # Find masks
    # --------------------------------------------------------

    microaneurysm_path = find_mask(
        IDRID_MICROANEURYSM_DIR,
        image_stem,
        "MA"
    )

    hemorrhage_path = find_mask(
        IDRID_HEMORRHAGE_DIR,
        image_stem,
        "HE"
    )

    hard_exudate_path = find_mask(
        IDRID_HARD_EXUDATE_DIR,
        image_stem,
        "EX"
    )

    soft_exudate_path = find_mask(
        IDRID_SOFT_EXUDATE_DIR,
        image_stem,
        "SE"
    )

    print("\nMasks:")

    print(
        "Microaneurysm:",
        microaneurysm_path
    )

    print(
        "Hemorrhage:",
        hemorrhage_path
    )

    print(
        "Hard Exudate:",
        hard_exudate_path
    )

    print(
        "Soft Exudate:",
        soft_exudate_path
    )

    # --------------------------------------------------------
    # Load masks
    # --------------------------------------------------------

    microaneurysm_mask = load_mask(
        microaneurysm_path,
        target_size
    )

    hemorrhage_mask = load_mask(
        hemorrhage_path,
        target_size
    )

    hard_exudate_mask = load_mask(
        hard_exudate_path,
        target_size
    )

    soft_exudate_mask = load_mask(
        soft_exudate_path,
        target_size
    )

    # --------------------------------------------------------
    # Individual overlays
    # --------------------------------------------------------

    micro_overlay = apply_mask(
        image,
        microaneurysm_mask,
        (0, 0, 255)
    )

    hemorrhage_overlay = apply_mask(
        image,
        hemorrhage_mask,
        (0, 165, 255)
    )

    hard_overlay = apply_mask(
        image,
        hard_exudate_mask,
        (0, 255, 255)
    )

    soft_overlay = apply_mask(
        image,
        soft_exudate_mask,
        (0, 255, 0)
    )

    # --------------------------------------------------------
    # Combined overlay
    # --------------------------------------------------------

    combined = image.copy()

    combined = apply_mask(
        combined,
        microaneurysm_mask,
        (0, 0, 255)
    )

    combined = apply_mask(
        combined,
        hemorrhage_mask,
        (0, 165, 255)
    )

    combined = apply_mask(
        combined,
        hard_exudate_mask,
        (0, 255, 255)
    )

    combined = apply_mask(
        combined,
        soft_exudate_mask,
        (0, 255, 0)
    )

    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    original_display = add_label(
        image,
        "Original"
    )

    micro_display = add_label(
        micro_overlay,
        "Microaneurysms"
    )

    hemorrhage_display = add_label(
        hemorrhage_overlay,
        "Haemorrhages"
    )

    hard_display = add_label(
        hard_overlay,
        "Hard Exudates"
    )

    soft_display = add_label(
        soft_overlay,
        "Soft Exudates"
    )

    combined_display = add_label(
        combined,
        "All Lesions"
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_original.jpg"
        ),
        original_display
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_microaneurysms.jpg"
        ),
        micro_display
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_haemorrhages.jpg"
        ),
        hemorrhage_display
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_hard_exudates.jpg"
        ),
        hard_display
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_soft_exudates.jpg"
        ),
        soft_display
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{image_stem}_all_lesions.jpg"
        ),
        combined_display
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nVisualization saved to:")
    print(OUTPUT_DIR)

    print("\nFiles created:")

    print(
        f"  {image_stem}_original.jpg"
    )

    print(
        f"  {image_stem}_microaneurysms.jpg"
    )

    print(
        f"  {image_stem}_haemorrhages.jpg"
    )

    print(
        f"  {image_stem}_hard_exudates.jpg"
    )

    print(
        f"  {image_stem}_soft_exudates.jpg"
    )

    print(
        f"  {image_stem}_all_lesions.jpg"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()