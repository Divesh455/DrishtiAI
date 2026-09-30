from pathlib import Path

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_CSV = (
    RAW_DATA_DIR /
    "train.csv"
)

IMAGE_DIR = (
    RAW_DATA_DIR /
    "train_images"
)

WEIGHTS_DIR = (
    BASE_DIR /
    "weights"
)

MODEL_PATH = (
    WEIGHTS_DIR /
    "dr_model.pth"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 5


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

NUM_EPOCHS = 20

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

VALIDATION_SIZE = 0.20

RANDOM_SEED = 42


# ============================================================
# LOSS SETTINGS
# ============================================================

# Small label smoothing helps reduce
# overconfident predictions.

LABEL_SMOOTHING = 0.05


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-7


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# DATA LOADER SETTINGS
# ============================================================

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = {

    0: "No DR",

    1: "Mild",

    2: "Moderate",

    3: "Severe",

    4: "Proliferative DR"

}


# ============================================================
# IDRiD DATASET
# ============================================================

IDRID_DIR = (
    RAW_DATA_DIR /
    "idrid"
)

IDRID_IMAGE_DIR = (
    IDRID_DIR /
    "images" /
    "training"
)

IDRID_ANNOTATION_DIR = (
    IDRID_DIR /
    "annotations" /
    "training"
)


IDRID_MICROANEURYSM_DIR = (
    IDRID_ANNOTATION_DIR /
    "microaneurysms"
)


IDRID_HEMORRHAGE_DIR = (
    IDRID_ANNOTATION_DIR /
    "haemorrhages"
)


IDRID_HARD_EXUDATE_DIR = (
    IDRID_ANNOTATION_DIR /
    "hard_exudates"
)


IDRID_SOFT_EXUDATE_DIR = (
    IDRID_ANNOTATION_DIR /
    "soft_exudates"
)


LESION_CLASSES = {

    1: "microaneurysm",

    2: "hemorrhage",

    3: "hard_exudate",

    4: "soft_exudate"

}


# Segmentation:
#
# 0 = background
# 1 = microaneurysm
# 2 = hemorrhage
# 3 = hard exudate
# 4 = soft exudate

NUM_LESION_CLASSES = 5


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================

def print_config():

    print("=" * 60)

    print(
        "DrishtiAI Training Configuration"
    )

    print("=" * 60)

    print(
        f"Dataset CSV          : {TRAIN_CSV}"
    )

    print(
        f"Image Folder         : {IMAGE_DIR}"
    )

    print(
        f"Model Path           : {MODEL_PATH}"
    )

    print(
        f"Image Size           : {IMAGE_SIZE}"
    )

    print(
        f"Classes              : {NUM_CLASSES}"
    )

    print(
        f"Batch Size           : {BATCH_SIZE}"
    )

    print(
        f"Epochs               : {NUM_EPOCHS}"
    )

    print(
        f"Learning Rate        : {LEARNING_RATE}"
    )

    print(
        f"Weight Decay         : {WEIGHT_DECAY}"
    )

    print(
        f"Label Smoothing      : {LABEL_SMOOTHING}"
    )

    print(
        f"Validation Size      : {VALIDATION_SIZE}"
    )

    print(
        f"Early Stop Patience  : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Device               : {DEVICE}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_config()