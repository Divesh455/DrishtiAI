import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from backend.models.dr_classifier import (
    create_model
)

from training.config import (
    TRAIN_CSV,
    TRAIN_IMAGE_DIR,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    RANDOM_SEED,
    FOCAL_GAMMA,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LR,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    DEVICE,
    VALIDATION_SIZE,
    SPLIT_RANDOM_STATE,
)

from training.dataset import (
    DRDataset,
    get_train_transforms,
    get_val_transforms,
)

from training.utils import (
    set_seed,
    calculate_metrics,
    calculate_class_weights,
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma


    def forward(
        self,
        logits,
        targets
    ):

        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=1
        )

        probs = torch.exp(
            log_probs
        )

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        focal_factor = (
            1.0 - target_probs
        ) ** self.gamma

        loss = -focal_factor * target_log_probs

        if self.alpha is not None:

            alpha_t = self.alpha[
                targets
            ]

            loss = alpha_t * loss

        return loss.mean()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_targets = []

    all_predictions = []


    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM
        )


        optimizer.step()


        running_loss += (
            loss.item() *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )


    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_targets = []

    all_predictions = []


    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            targets
        )


        running_loss += (
            loss.item() *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )


    return epoch_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DrishtiAI - Diabetic Retinopathy Training")
    print("=" * 70)


    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )


    if not TRAIN_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found: "
            f"{TRAIN_IMAGE_DIR}"
        )


    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAIN_CSV
    )


    required_columns = {
        "id_code",
        "diagnosis"
    }


    missing_columns = (
        required_columns -
        set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            f"Missing CSV columns: "
            f"{missing_columns}"
        )


    print(
        f"\nTotal dataset images: {len(df)}"
    )


    # --------------------------------------------------------
    # CHECK LABELS
    # --------------------------------------------------------

    print(
        "\nClass distribution:"
    )

    print(
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        stratify=df["diagnosis"],
        random_state=SPLIT_RANDOM_STATE
    )


    print(
        f"\nTraining samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = DRDataset(
        dataframe=train_df,
        image_dir=TRAIN_IMAGE_DIR,
        transform=get_train_transforms(
            IMAGE_SIZE
        )
    )


    val_dataset = DRDataset(
        dataframe=val_df,
        image_dir=TRAIN_IMAGE_DIR,
        transform=get_val_transforms(
            IMAGE_SIZE
        )
    )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_df["diagnosis"].values,
        NUM_CLASSES
    )


    class_weights = class_weights.to(
        DEVICE
    )


    print(
        "\nClass weights:"
    )

    for i, weight in enumerate(
        class_weights.cpu().numpy()
    ):

        print(
            f"{i} - "
            f"{CLASS_NAMES[i]}: "
            f"{weight:.4f}"
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES,
        device=DEVICE
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=FOCAL_GAMMA
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR
    )


    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_macro_f1 = -1.0

    best_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]


        print(
            f"\nLearning Rate: {current_lr:.8f}"
        )


        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )


        print(
            f"\nVal Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall: "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Weighted-F1: "
            f"{val_metrics['weighted_f1']:.4f}"
        )


        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0


            MODEL_SAVE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            checkpoint = {

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    NUM_CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "best_val_macro_f1":
                    best_macro_f1,

                "best_val_accuracy":
                    best_accuracy,

                "class_names":
                    CLASS_NAMES,

                "class_weights":
                    class_weights.cpu(),

            }


            torch.save(
                checkpoint,
                MODEL_SAVE_PATH
            )


            print(
                "\n*** New best model saved ***"
            )

            print(
                f"Best Macro-F1: "
                f"{best_macro_f1:.4f}"
            )


        else:

            epochs_without_improvement += 1


            print(
                f"\nNo improvement."
                f" Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Model saved to:"
        f"\n{MODEL_SAVE_PATH}"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()