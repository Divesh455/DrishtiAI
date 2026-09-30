import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from backend.models.dr_classifier import create_model

from training.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    VALIDATION_SIZE,
    RANDOM_SEED,
    CLASS_NAMES,
    DEVICE,
)

from training.dataset import DRDataset


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
    """
    Make training as reproducible as possible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Focal Loss gives more importance to difficult examples
    and reduces the contribution of easy examples.

    alpha:
        Class-specific weighting.

    gamma:
        Controls how strongly easy examples are down-weighted.
    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
    ):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):

        # Log probabilities
        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=1
        )

        # Probabilities
        probs = torch.exp(log_probs)

        # Select target class
        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        target_probs = probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        # Focal modulation
        focal_factor = (1.0 - target_probs) ** self.gamma

        # Base loss
        loss = -focal_factor * target_log_probs

        # Class weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


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

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
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
    device,
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(prefix, metrics):

    print(
        f"{prefix} Accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"{prefix} Precision     : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"{prefix} Recall        : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"{prefix} Macro-F1      : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"{prefix} Weighted-F1   : "
        f"{metrics['weighted_f1']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DrishtiAI - IMPROVED DR TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed()

    print()
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    csv_path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "train.csv"
    )

    image_dir = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "train_images"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found:\n{csv_path}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Total images: {len(df)}"
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        stratify=df["diagnosis"],
        random_state=RANDOM_SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(
        f"Training images: {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Dataset distributions
    # --------------------------------------------------------

    print()
    print("Training class distribution:")

    train_counts = (
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    for class_id in range(NUM_CLASSES):

        count = int(
            train_counts.get(
                class_id,
                0
            )
        )

        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<20}: "
            f"{count}"
        )

    print()
    print("Validation class distribution:")

    val_counts = (
        val_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    for class_id in range(NUM_CLASSES):

        count = int(
            val_counts.get(
                class_id,
                0
            )
        )

        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<20}: "
            f"{count}"
        )

    # ========================================================
    # TRANSFORMS
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    # Training augmentation
    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ])

    # Validation must remain deterministic
    val_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ])

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = DRDataset(
        dataframe=train_df,
        image_dir=image_dir,
        transform=train_transform
    )

    val_dataset = DRDataset(
        dataframe=val_df,
        image_dir=image_dir,
        transform=val_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print()
    print("=" * 70)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 70)

    class_counts = np.array([
        train_counts.get(
            class_id,
            0
        )
        for class_id in range(NUM_CLASSES)
    ])

    total_samples = class_counts.sum()

    # Balanced weighting:
    # N / (number_of_classes * class_count)
    class_weights = (
        total_samples /
        (
            NUM_CLASSES *
            class_counts
        )
    )

    class_weights = (
        class_weights /
        class_weights.mean()
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<20}: "
            f"{class_weights[class_id]:.4f}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model(
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    print(
        "✓ EfficientNet-B0 initialized "
        "with ImageNet pretrained weights"
    )

    # ========================================================
    # LOSS
    # ========================================================

    print()
    print("=" * 70)
    print("LOSS FUNCTION")
    print("=" * 70)

    print("Using Focal Loss")
    print("Gamma: 2.0")
    print("Class weighting: enabled")

    criterion = FocalLoss(
        alpha=class_weights_tensor,
        gamma=2.0
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # ========================================================
    # TRAINING SETTINGS
    # ========================================================

    best_macro_f1 = -1.0
    best_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    early_stopping_patience = 5

    weights_dir = (
        PROJECT_ROOT /
        "weights"
    )

    weights_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # IMPORTANT:
    # Save the new experiment separately.
    new_model_path = (
        weights_dir /
        "dr_model_focal.pth"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Early stopping patience: "
        f"{early_stopping_patience}"
    )

    print()

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print("-" * 70)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Train Macro-F1  : "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision   : "
            f"{val_metrics['macro_precision']:.4f}"
        )

        print(
            f"Val Recall      : "
            f"{val_metrics['macro_recall']:.4f}"
        )

        print(
            f"Val Macro-F1    : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Weighted-F1 : "
            f"{val_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        current_macro_f1 = (
            val_metrics["macro_f1"]
        )

        if current_macro_f1 > best_macro_f1:

            best_macro_f1 = current_macro_f1

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
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
                    class_weights.tolist(),

                "epoch":
                    epoch,

                "loss_function":
                    "FocalLoss",

                "focal_gamma":
                    2.0,
            }

            torch.save(
                checkpoint,
                new_model_path
            )

            print()
            print(
                "✓ NEW BEST MODEL SAVED"
            )

            print(
                f"  Macro-F1: "
                f"{best_macro_f1:.4f}"
            )

            print(
                f"  Accuracy: "
                f"{best_accuracy:.4f}"
            )

            print(
                f"  Path: "
                f"{new_model_path}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{early_stopping_patience})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):

            print()
            print(
                "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 70
            )

            print(
                f"No Macro-F1 improvement "
                f"for {early_stopping_patience} epochs."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best Epoch          : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation "
        f"Macro-F1           : "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Validation "
        f"Accuracy           : "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Model saved at      : "
        f"{new_model_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The original dr_model.pth was NOT overwritten."
    )

    print(
        "The new Focal Loss model is saved as "
        "dr_model_focal.pth."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()