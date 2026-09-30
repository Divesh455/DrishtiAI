import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from training.config import (
    IDRID_IMAGE_DIR,
    IDRID_MICROANEURYSM_DIR,
    IDRID_HEMORRHAGE_DIR,
    IDRID_HARD_EXUDATE_DIR,
    IDRID_SOFT_EXUDATE_DIR,
    NUM_LESION_CLASSES,
    LESION_CLASSES,
    WEIGHTS_DIR,
)

from training.lesion_dataset import IDRiDLesionDataset

from backend.models.lesion_detector import create_lesion_model


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512
BATCH_SIZE = 2
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42

MODEL_PATH = WEIGHTS_DIR / "lesion_model.pth"

CLASS_NAMES = {
    0: "Background",
    1: "Microaneurysm",
    2: "Hemorrhage",
    3: "Hard Exudate",
    4: "Soft Exudate",
}


# ============================================================
# FIND COMPLETE IMAGES
# ============================================================

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
            complete_images.append(image_path)

    return complete_images


# ============================================================
# DICE
# ============================================================

def calculate_dice(
    prediction,
    target,
    class_id
):

    prediction_class = (
        prediction == class_id
    )

    target_class = (
        target == class_id
    )

    intersection = np.logical_and(
        prediction_class,
        target_class
    ).sum()

    prediction_area = prediction_class.sum()

    target_area = target_class.sum()

    denominator = (
        prediction_area
        + target_area
    )

    if denominator == 0:

        return None

    return (
        2.0 * intersection
        / denominator
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    prediction,
    target,
    class_id
):

    prediction_class = (
        prediction == class_id
    )

    target_class = (
        target == class_id
    )

    intersection = np.logical_and(
        prediction_class,
        target_class
    ).sum()

    union = np.logical_or(
        prediction_class,
        target_class
    ).sum()

    if union == 0:

        return None

    return (
        intersection
        / union
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DrishtiAI - IDRiD LESION MODEL EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Lesion model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    image_paths = get_complete_images()

    print(
        f"Complete images: {len(image_paths)}"
    )

    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED
    )

    print(
        f"Training images: {len(train_paths)}"
    )

    print(
        f"Validation images: {len(val_paths)}"
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = IDRiDLesionDataset(
        val_paths,
        image_size=IMAGE_SIZE,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_lesion_model(
        num_classes=NUM_LESION_CLASSES,
        device=device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nLoaded model:"
    )

    print(MODEL_PATH)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    dice_scores = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    iou_scores = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    print("\nEvaluating...\n")

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            predictions = (
                predictions.cpu().numpy()
            )

            masks = masks.numpy()

            for batch_index in range(
                len(predictions)
            ):

                prediction = predictions[
                    batch_index
                ]

                target = masks[
                    batch_index
                ]

                for class_id in CLASS_NAMES:

                    dice = calculate_dice(
                        prediction,
                        target,
                        class_id
                    )

                    iou = calculate_iou(
                        prediction,
                        target,
                        class_id
                    )

                    if dice is not None:

                        dice_scores[
                            class_id
                        ].append(dice)

                    if iou is not None:

                        iou_scores[
                            class_id
                        ].append(iou)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 70)
    print("LESION SEGMENTATION RESULTS")
    print("=" * 70)

    for class_id, class_name in CLASS_NAMES.items():

        dice_values = dice_scores[class_id]

        iou_values = iou_scores[class_id]

        if dice_values:

            mean_dice = np.mean(
                dice_values
            )

        else:

            mean_dice = 0.0

        if iou_values:

            mean_iou = np.mean(
                iou_values
            )

        else:

            mean_iou = 0.0

        print(
            f"\n{class_name}"
        )

        print(
            f"  Dice: {mean_dice:.4f}"
        )

        print(
            f"  IoU : {mean_iou:.4f}"
        )

    # --------------------------------------------------------
    # Lesion-only average
    # --------------------------------------------------------

    lesion_dice = []

    lesion_iou = []

    for class_id in range(1, 5):

        if dice_scores[class_id]:

            lesion_dice.append(
                np.mean(
                    dice_scores[class_id]
                )
            )

        if iou_scores[class_id]:

            lesion_iou.append(
                np.mean(
                    iou_scores[class_id]
                )
            )

    print("\n" + "=" * 70)

    if lesion_dice:

        print(
            "Mean Lesion Dice: "
            f"{np.mean(lesion_dice):.4f}"
        )

    if lesion_iou:

        print(
            "Mean Lesion IoU : "
            f"{np.mean(lesion_iou):.4f}"
        )

    print("=" * 70)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()