import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from backend.models.dr_classifier import create_model

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
# CHECKPOINT PATH
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "weights"
    / "dr_training_checkpoint.pth"
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets,
    ):

        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=1,
        )

        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probs = probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = (
            1.0 - target_probs
        ) ** self.gamma

        loss = (
            -focal_factor
            * target_log_probs
        )

        if self.alpha is not None:

            alpha_t = self.alpha[targets]

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
    device,
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
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
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
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
    device,
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    best_macro_f1,
    best_accuracy,
    epochs_without_improvement,
    class_weights,
):

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_macro_f1":
            best_macro_f1,

        "best_accuracy":
            best_accuracy,

        "epochs_without_improvement":
            epochs_without_improvement,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "class_names":
            CLASS_NAMES,

        "class_weights":
            class_weights.cpu(),

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "focal_gamma":
            FOCAL_GAMMA,

        "random_seed":
            RANDOM_SEED,
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(
        ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    # Replace old checkpoint only after
    # the new checkpoint has been written.
    temporary_path.replace(
        CHECKPOINT_PATH
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    device,
):

    if not CHECKPOINT_PATH.exists():

        print(
            "\nNo training checkpoint found."
        )

        print(
            "Starting training from epoch 1."
        )

        return (
            1,
            -1.0,
            0.0,
            0,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "CHECKPOINT FOUND"
    )

    print(
        "=" * 70
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    completed_epoch = int(
        checkpoint["epoch"]
    )

    best_macro_f1 = float(
        checkpoint.get(
            "best_macro_f1",
            -1.0,
        )
    )

    best_accuracy = float(
        checkpoint.get(
            "best_accuracy",
            0.0,
        )
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    next_epoch = (
        completed_epoch + 1
    )

    print(
        f"Completed epoch: "
        f"{completed_epoch}"
    )

    print(
        f"Best Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Next epoch: "
        f"{next_epoch}"
    )

    print(
        "=" * 70
    )

    return (
        next_epoch,
        best_macro_f1,
        best_accuracy,
        epochs_without_improvement,
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    model,
    epoch,
    best_macro_f1,
    best_accuracy,
    class_weights,
):

    MODEL_SAVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        MODEL_SAVE_PATH,
    )

    print(
        "\n*** BEST MODEL SAVED ***"
    )

    print(
        f"Epoch: {epoch}"
    )

    print(
        f"Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Path: "
        f"{MODEL_SAVE_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "DrishtiAI - Resumable DR Training"
    )

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
            torch.cuda.get_device_name(0),
        )

    else:

        print(
            "GPU not available."
        )

        print(
            "Training will run on CPU."
        )


    # --------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found:\n"
            f"{TRAIN_CSV}"
        )

    if not TRAIN_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{TRAIN_IMAGE_DIR}"
        )


    # --------------------------------------------------------
    # LOAD DATAFRAME
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAIN_CSV
    )

    required_columns = {
        "id_code",
        "diagnosis",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing CSV columns: "
            f"{missing_columns}"
        )


    print(
        f"\nTotal dataset images: "
        f"{len(df)}"
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
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
        random_state=SPLIT_RANDOM_STATE,
    )


    print(
        f"\nTraining samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_df)}"
    )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = DRDataset(
        dataframe=train_df,
        image_dir=TRAIN_IMAGE_DIR,
        transform=get_train_transforms(
            IMAGE_SIZE
        ),
    )

    val_dataset = DRDataset(
        dataframe=val_df,
        image_dir=TRAIN_IMAGE_DIR,
        transform=get_val_transforms(
            IMAGE_SIZE
        ),
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
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_df["diagnosis"].values,
        NUM_CLASSES,
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
        device=DEVICE,
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=FOCAL_GAMMA,
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )
    )


    # --------------------------------------------------------
    # LOAD OR INITIALIZE CHECKPOINT
    # --------------------------------------------------------

    (
        start_epoch,
        best_macro_f1,
        best_accuracy,
        epochs_without_improvement,
    ) = load_checkpoint(
        model,
        optimizer,
        scheduler,
        DEVICE,
    )


    # --------------------------------------------------------
    # ALREADY FINISHED?
    # --------------------------------------------------------

    if start_epoch > NUM_EPOCHS:

        print(
            "\nTraining has already reached "
            f"epoch {NUM_EPOCHS}."
        )

        print(
            "Nothing more to train."
        )

        return


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    print(
        "\nStarting/resuming training..."
    )


    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
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

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )


        # ----------------------------------------------------
        # CURRENT LR
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"\nLearning Rate: "
            f"{current_lr:.8f}"
        )

        print(
            f"\nTrain Loss: "
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
        # UPDATE SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )


        # ----------------------------------------------------
        # CHECK BEST MODEL
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

            epochs_without_improvement = 0

            save_best_model(
                model=model,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                best_accuracy=best_accuracy,
                class_weights=class_weights,
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nNo improvement."
            )

            print(
                f"Early stopping patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


        # ----------------------------------------------------
        # SAVE RESUME CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_macro_f1=best_macro_f1,
            best_accuracy=best_accuracy,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            class_weights=class_weights,
        )


        print(
            "\nCheckpoint saved:"
        )

        print(
            CHECKPOINT_PATH
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
        f"Best Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"\nBest model:"
    )

    print(
        MODEL_SAVE_PATH
    )

    print(
        f"\nResume checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()