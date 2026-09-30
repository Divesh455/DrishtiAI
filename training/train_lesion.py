import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import random

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from training.config import (
    IDRID_IMAGE_DIR,
    NUM_LESION_CLASSES,
    WEIGHTS_DIR,
)

from training.lesion_dataset import (
    IDRiDLesionDataset
)

from backend.models.lesion_detector import (
    create_lesion_model
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

BATCH_SIZE = 2

EPOCHS = 30

LEARNING_RATE = 0.0001

VALIDATION_SIZE = 0.2

RANDOM_SEED = 42

MODEL_PATH = (
    WEIGHTS_DIR /
    "lesion_model.pth"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# FIND COMPLETE DATA
# ============================================================

def get_complete_images():

    from training.config import (
        IDRID_MICROANEURYSM_DIR,
        IDRID_HEMORRHAGE_DIR,
        IDRID_HARD_EXUDATE_DIR,
        IDRID_SOFT_EXUDATE_DIR,
    )

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


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        num_classes = logits.shape[1]

        total_dice = 0.0

        for class_index in range(
            num_classes
        ):

            prediction = probabilities[
                :, class_index
            ]

            target = (
                targets == class_index
            ).float()

            intersection = (
                prediction * target
            ).sum()

            denominator = (
                prediction.sum()
                + target.sum()
            )

            dice = (
                (2.0 * intersection + self.smooth)
                /
                (denominator + self.smooth)
            )

            total_dice += dice

        return 1.0 - (
            total_dice / num_classes
        )


# ============================================================
# COMBINED LOSS
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = self.cross_entropy(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return ce_loss + dice_loss


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 70)
    print(
        "DrishtiAI - IDRiD LESION SEGMENTATION TRAINING"
    )
    print("=" * 70)

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
        f"\nComplete images: {len(image_paths)}"
    )

    if len(image_paths) < 2:

        raise RuntimeError(
            "Not enough IDRiD images for training."
        )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

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
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = IDRiDLesionDataset(
        train_paths,
        image_size=IMAGE_SIZE,
        augment=True
    )

    val_dataset = IDRiDLesionDataset(
        val_paths,
        image_size=IMAGE_SIZE,
        augment=False
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"\nTrain batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_lesion_model(
        num_classes=NUM_LESION_CLASSES,
        device=device
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = CombinedLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nStarting training...\n")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "num_classes":
                        NUM_LESION_CLASSES,

                    "image_size":
                        IMAGE_SIZE,

                    "best_val_loss":
                        best_val_loss,
                },
                MODEL_PATH
            )

            print(
                f"  ✓ Best model saved: {MODEL_PATH}"
            )

    print("\n" + "=" * 70)
    print("Training completed.")
    print("=" * 70)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()