# compare_dr_models.py
#
# Compare:
#   1. Current DrishtiAI DR model
#   2. Senanur 5-fold ONNX DR model
#
# Dataset:
#   APTOS 2019
#
# Evaluation:
#   - Accuracy
#   - Macro F1
#   - Weighted F1
#   - Quadratic Weighted Kappa
#   - Per-grade Precision / Recall / F1
#   - Referable DR sensitivity / specificity
#   - Model agreement
#   - Unique wins
#   - Inference time
#   - Per-image predictions
#   - Confusion matrices
#
# Expected project structure:
#
# DrishtiAI/
# ├── data/
# │   └── raw/
# │       ├── train.csv
# │       └── train_images/
# ├── weights/
# │   └── dr_model.pth
# ├── external_models/
# │   └── senanur_dr/
# │       ├── fold1.onnx
# │       ├── fold2.onnx
#       ├── fold3.onnx
#       ├── fold4.onnx
#       ├── fold5.onnx
#       └── export.json
# └── compare_dr_models.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)

import onnxruntime as ort

# IMPORTANT:
# The checkpoint was trained with torchvision EfficientNet-B0,
# NOT timm EfficientNet-B0.
from torchvision.models import efficientnet_b0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"

OUR_MODEL_PATH = PROJECT_ROOT / "weights" / "dr_model.pth"

SENANUR_DIR = PROJECT_ROOT / "external_models" / "senanur_dr"

OUTPUT_DIR = PROJECT_ROOT / "dr_model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE_OUR_MODEL = 224

# Senanur official export:
# preprocessing size = 512
# model input size = 384
SENANUR_PREPROCESS_SIZE = 512
SENANUR_MODEL_SIZE = 384

NUM_CLASSES = 5

RANDOM_STATE = 42
VAL_SIZE = 0.20

REFERABLE_THRESHOLD = 2

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ImageNet normalization
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("DrishtiAI DR MODEL COMPARISON")
print("=" * 80)
print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")
print()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def strip_state_dict_prefixes(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove common checkpoint prefixes.

    Example:
        module.features.0.0.weight
    becomes:
        features.0.0.weight
    """

    cleaned = {}

    prefixes = (
        "module.",
        "model.",
        "network.",
        "net.",
    )

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_checkpoint_state_dict(
    checkpoint_path: Path,
) -> Dict[str, torch.Tensor]:
    """
    Load model_state_dict from a PyTorch checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Sometimes the checkpoint itself is the state_dict.
            state_dict = checkpoint

    else:
        raise RuntimeError(
            f"Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Checkpoint state_dict is not a dictionary."
        )

    state_dict = strip_state_dict_prefixes(state_dict)

    return state_dict


# ============================================================
# LOAD OUR DR MODEL
# ============================================================

def load_our_model() -> nn.Module:

    print("-" * 80)
    print("Loading current DrishtiAI model")
    print("-" * 80)

    print(f"Checkpoint: {OUR_MODEL_PATH}")

    state_dict = load_checkpoint_state_dict(
        OUR_MODEL_PATH
    )

    print(
        f"Checkpoint state source: "
        f"{'model_state_dict / state_dict / raw'}"
    )

    # IMPORTANT:
    # Your checkpoint uses torchvision EfficientNet-B0
    # naming such as:
    #
    # features.0.0.weight
    # features.1.0.block.0.0.weight
    #
    # Therefore we must use torchvision EfficientNet-B0.

    model = efficientnet_b0(
        weights=None
    )

    # Replace final classifier with 5 DR classes.
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    # Load trained checkpoint.
    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print(f"Missing keys    : {len(missing_keys)}")
    print(f"Unexpected keys : {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing key examples:")
        for key in missing_keys[:20]:
            print("  ", key)

    if unexpected_keys:
        print("\nUnexpected key examples:")
        for key in unexpected_keys[:20]:
            print("  ", key)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "\nCurrent DR checkpoint could not be loaded "
            "completely.\n"
            "The architecture and checkpoint keys do not match."
        )

    model.to(DEVICE)
    model.eval()

    print("\nCurrent DrishtiAI model loaded successfully.")

    return model


# ============================================================
# OUR MODEL PREPROCESSING
# ============================================================

def preprocess_for_our_model(
    image_bgr: np.ndarray,
) -> torch.Tensor:
    """
    Preprocessing used for current DrishtiAI model.

    Steps:
        BGR -> RGB
        Resize 224x224
        Convert to float
        /255
        ImageNet normalization
        HWC -> CHW
        Batch dimension
    """

    if image_bgr is None:
        raise ValueError("Input image is None.")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_rgb = cv2.resize(
        image_rgb,
        (
            IMAGE_SIZE_OUR_MODEL,
            IMAGE_SIZE_OUR_MODEL,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image_rgb.astype(np.float32) / 255.0

    image = (
        image - IMAGENET_MEAN
    ) / IMAGENET_STD

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# PREDICT WITH OUR MODEL
# ============================================================

@torch.no_grad()
def predict_our_model(
    model: nn.Module,
    image_bgr: np.ndarray,
) -> Tuple[int, float, Dict[int, float], float]:
    """
    Returns:

        grade
        confidence
        probabilities
        inference_time_seconds
    """

    tensor = preprocess_for_our_model(
        image_bgr
    )

    start = time.perf_counter()

    logits = model(tensor)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1,
    )

    elapsed = (
        time.perf_counter() - start
    )

    grade = int(
        predicted_class.item()
    )

    confidence_value = float(
        confidence.item()
    )

    probs = {
        class_id: float(
            probabilities[0, class_id].item()
        )
        for class_id in range(NUM_CLASSES)
    }

    return (
        grade,
        confidence_value,
        probs,
        elapsed,
    )


# ============================================================
# SENANUR PREPROCESSING
# ============================================================

def senanur_auto_crop(
    image_bgr: np.ndarray,
    tol: int = 7,
) -> np.ndarray:
    """
    Reproduce the preprocessing logic from the
    Senanur APTOS grading repository.

    The original preprocessing removes the black frame
    around the fundus image.
    """

    if image_bgr is None:
        raise ValueError(
            "Input image is None."
        )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    mask = gray > tol

    coords = np.argwhere(mask)

    if coords.size == 0:
        return image_bgr

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image_bgr[
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    return cropped


def senanur_pad_to_square(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Pad image to square using black pixels.
    """

    h, w = image_bgr.shape[:2]

    if h == w:
        return image_bgr

    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top

    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return padded


def preprocess_for_senanur(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the Senanur baseline preprocessing.

    Official configuration:

        preprocessing size = 512
        CLAHE = false
        square mode = pad

    Then:

        BGR -> RGB
        resize 384x384
        /255
        ImageNet normalization
        CHW
        batch
    """

    # --------------------------------------------------------
    # 1. Auto crop black frame
    # --------------------------------------------------------

    image = senanur_auto_crop(
        image_bgr,
        tol=7,
    )

    # --------------------------------------------------------
    # 2. Resize preprocessing image to 512
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            SENANUR_PREPROCESS_SIZE,
            SENANUR_PREPROCESS_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # 3. Pad square
    # --------------------------------------------------------

    image = senanur_pad_to_square(
        image
    )

    # --------------------------------------------------------
    # 4. Final model input resize to 384
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            SENANUR_MODEL_SIZE,
            SENANUR_MODEL_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # 5. BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # 6. /255
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # 7. ImageNet normalization
    # --------------------------------------------------------

    image = (
        image - IMAGENET_MEAN
    ) / IMAGENET_STD

    # --------------------------------------------------------
    # 8. HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # --------------------------------------------------------
    # 9. Add batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0,
    )

    return image.astype(
        np.float32
    )


# ============================================================
# LOAD SENANUR ONNX MODELS
# ============================================================

def load_senanur_models():
    """
    Load all 5 Senanur ONNX folds.
    """

    print()
    print("-" * 80)
    print("Loading Senanur 5-fold ONNX model")
    print("-" * 80)

    fold_paths = [
        SENANUR_DIR / "fold1.onnx",
        SENANUR_DIR / "fold2.onnx",
        SENANUR_DIR / "fold3.onnx",
        SENANUR_DIR / "fold4.onnx",
        SENANUR_DIR / "fold5.onnx",
    ]

    sessions = []

    for index, fold_path in enumerate(
        fold_paths,
        start=1,
    ):

        if not fold_path.exists():
            raise FileNotFoundError(
                f"Senanur fold not found:\n"
                f"{fold_path}"
            )

        print(
            f"Loading fold {index}: "
            f"{fold_path.name}"
        )

        session = ort.InferenceSession(
            str(fold_path),
            providers=[
                "CPUExecutionProvider"
            ],
        )

        sessions.append(
            session
        )

    print(
        f"\nLoaded {len(sessions)} "
        f"Senanur ONNX folds."
    )

    return sessions


# ============================================================
# SENANUR OUTPUT DECODING
# ============================================================

def extract_scalar_output(
    raw_output,
) -> float:
    """
    Convert an ONNX output to one scalar.

    Supports common shapes:
        [1]
        [1, 1]
        scalar
    """

    value = np.asarray(
        raw_output
    )

    value = value.reshape(-1)

    if value.size == 0:
        raise RuntimeError(
            "ONNX model returned an empty output."
        )

    return float(
        value[0]
    )


def score_to_grade(
    score: float,
    thresholds: List[float],
) -> int:
    """
    Senanur uses:

        np.digitize(score, thresholds)

    for grades 0..4.
    """

    return int(
        np.digitize(
            score,
            thresholds,
        )
    )


# ============================================================
# SENANUR SINGLE IMAGE PREDICTION
# ============================================================

def predict_senanur(
    sessions,
    image_bgr: np.ndarray,
    thresholds: List[float],
):
    """
    Run all five Senanur folds.

    Returns:
        grade
        confidence
        fold_scores
        fold_grades
        spread
        mean_score
        inference_time
    """

    input_tensor = preprocess_for_senanur(
        image_bgr
    )

    fold_scores = []

    start = time.perf_counter()

    for session in sessions:

        input_name = (
            session.get_inputs()[0].name
        )

        outputs = session.run(
            None,
            {
                input_name: input_tensor
            },
        )

        if not outputs:
            raise RuntimeError(
                "Senanur ONNX model returned no outputs."
            )

        score = extract_scalar_output(
            outputs[0]
        )

        fold_scores.append(
            score
        )

    elapsed = (
        time.perf_counter() - start
    )

    fold_scores_np = np.asarray(
        fold_scores,
        dtype=np.float64,
    )

    mean_score = float(
        fold_scores_np.mean()
    )

    spread = float(
        fold_scores_np.std()
    )

    fold_grades = [
        score_to_grade(
            score,
            thresholds,
        )
        for score in fold_scores
    ]

    final_grade = score_to_grade(
        mean_score,
        thresholds,
    )

    # Ordinal regression does not directly produce
    # a 5-class softmax probability.
    #
    # We therefore use normalized inverse distance
    # from the final grade's interval as an approximate
    # confidence indicator.
    #
    # This is a comparison confidence only, NOT a calibrated
    # medical probability.

    confidence = estimate_senanur_confidence(
        mean_score,
        final_grade,
        thresholds,
    )

    return {
        "grade": final_grade,
        "confidence": confidence,
        "fold_scores": fold_scores,
        "fold_grades": fold_grades,
        "spread": spread,
        "mean_score": mean_score,
        "inference_time": elapsed,
    }


# ============================================================
# SENANUR CONFIDENCE ESTIMATION
# ============================================================

def estimate_senanur_confidence(
    score: float,
    grade: int,
    thresholds: List[float],
) -> float:
    """
    Estimate an ordinal-model confidence from the
    distance to adjacent grade thresholds.

    IMPORTANT:
    This is not the original model's calibrated probability.
    It is only useful for side-by-side comparison output.
    """

    if grade <= 0:

        if len(thresholds) == 0:
            return 1.0

        distance = thresholds[0] - score

        scale = max(
            abs(thresholds[0]),
            1.0,
        )

        confidence = (
            0.5
            + 0.5 * min(
                max(distance / scale, 0.0),
                1.0,
            )
        )

        return float(
            np.clip(confidence, 0.0, 1.0)
        )

    if grade >= len(thresholds):

        distance = (
            score - thresholds[-1]
        )

        scale = max(
            abs(thresholds[-1]),
            1.0,
        )

        confidence = (
            0.5
            + 0.5 * min(
                max(distance / scale, 0.0),
                1.0,
            )
        )

        return float(
            np.clip(confidence, 0.0, 1.0)
        )

    lower_threshold = thresholds[
        grade - 1
    ]

    upper_threshold = thresholds[
        grade
    ]

    interval_width = (
        upper_threshold
        - lower_threshold
    )

    if interval_width <= 0:
        return 0.5

    distance_to_boundary = min(
        score - lower_threshold,
        upper_threshold - score,
    )

    normalized = (
        distance_to_boundary
        / (interval_width / 2.0)
    )

    confidence = 0.5 + 0.5 * np.clip(
        normalized,
        0.0,
        1.0,
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


# ============================================================
# LOAD SENANUR THRESHOLDS
# ============================================================

def load_senanur_thresholds() -> List[float]:

    export_path = (
        SENANUR_DIR / "export.json"
    )

    if not export_path.exists():
        raise FileNotFoundError(
            f"Senanur export.json not found:\n"
            f"{export_path}"
        )

    with open(
        export_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    thresholds = data.get(
        "thresholds"
    )

    if not thresholds:
        raise RuntimeError(
            "Could not find Senanur thresholds "
            "inside export.json."
        )

    thresholds = [
        float(x)
        for x in thresholds
    ]

    if len(thresholds) != 4:
        raise RuntimeError(
            "Expected 4 Senanur grade thresholds "
            f"for grades 0-4, got {len(thresholds)}."
        )

    print(
        "Senanur thresholds:",
        thresholds,
    )

    return thresholds


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():
    """
    Load APTOS train.csv and reproduce the same
    80/20 stratified validation split.

    random_state=42
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"APTOS CSV not found:\n{CSV_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"APTOS image directory not found:\n{IMAGE_DIR}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    required_columns = {
        "id_code",
        "diagnosis",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing CSV columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean rows
    # --------------------------------------------------------

    df = df[
        df["id_code"].notna()
        & df["diagnosis"].notna()
    ].copy()

    df["id_code"] = (
        df["id_code"]
        .astype(str)
        .str.strip()
    )

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Only evaluate images that physically exist
    # --------------------------------------------------------

    valid_rows = []

    for _, row in df.iterrows():

        image_id = row["id_code"]

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
        )

        if image_path.exists():
            valid_rows.append(
                {
                    "id_code": image_id,
                    "diagnosis": int(
                        row["diagnosis"]
                    ),
                    "image_path": str(
                        image_path
                    ),
                }
            )

    valid_df = pd.DataFrame(
        valid_rows
    )

    print(
        f"Total images available: "
        f"{len(valid_df)}"
    )

    # --------------------------------------------------------
    # Reproduce 80/20 stratified split
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        valid_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=valid_df[
            "diagnosis"
        ],
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    print(
        f"Training images:   {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )

    print(
        "Validation split: "
        "80/20, random_state=42, stratified"
    )

    print()

    print("Validation class distribution:")

    distribution = (
        val_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    for grade, count in distribution.items():

        print(
            f"  Grade {grade} "
            f"({CLASS_NAMES[int(grade)]}): "
            f"{count}"
        )

    return train_df, val_df


# ============================================================
# METRICS
# ============================================================

def calculate_referable_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: int = 2,
):
    """
    Referable DR:
        grade >= 2

    Positive = referable
    Negative = non-referable
    """

    true_binary = (
        y_true >= threshold
    ).astype(int)

    pred_binary = (
        y_pred >= threshold
    ).astype(int)

    sensitivity = recall_score(
        true_binary,
        pred_binary,
        zero_division=0,
    )

    specificity = recall_score(
        1 - true_binary,
        1 - pred_binary,
        zero_division=0,
    )

    precision = precision_score(
        true_binary,
        pred_binary,
        zero_division=0,
    )

    return {
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "precision": float(
            precision
        ),
    }


def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """
    Calculate all primary metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    referable = calculate_referable_metrics(
        y_true,
        y_pred,
        threshold=REFERABLE_THRESHOLD,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[
            CLASS_NAMES[i]
            for i in range(NUM_CLASSES)
        ],
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            precision_macro
        ),
        "macro_recall": float(
            recall_macro
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "quadratic_weighted_kappa": float(
            qwk
        ),
        "referable_dr": referable,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================
# MODEL AGREEMENT
# ============================================================

def calculate_agreement(
    y_true: np.ndarray,
    our_pred: np.ndarray,
    senanur_pred: np.ndarray,
):
    """
    Compare predictions from both models.
    """

    agreement_mask = (
        our_pred == senanur_pred
    )

    agreement_rate = (
        float(
            agreement_mask.mean()
        )
        if len(agreement_mask) > 0
        else 0.0
    )

    # Correctness
    our_correct = (
        our_pred == y_true
    )

    senanur_correct = (
        senanur_pred == y_true
    )

    # Both correct
    both_correct = (
        our_correct
        & senanur_correct
    )

    # Only ours correct
    only_ours_correct = (
        our_correct
        & ~senanur_correct
    )

    # Only Senanur correct
    only_senanur_correct = (
        ~our_correct
        & senanur_correct
    )

    # Neither correct
    neither_correct = (
        ~our_correct
        & ~senanur_correct
    )

    return {
        "agreement_rate": float(
            agreement_rate
        ),
        "agreement_count": int(
            agreement_mask.sum()
        ),
        "total_images": int(
            len(y_true)
        ),
        "both_correct": int(
            both_correct.sum()
        ),
        "only_our_model_correct": int(
            only_ours_correct.sum()
        ),
        "only_senanur_correct": int(
            only_senanur_correct.sum()
        ),
        "neither_correct": int(
            neither_correct.sum()
        ),
    }


# ============================================================
# LOAD IMAGES
# ============================================================

def read_image(
    image_path: str,
) -> np.ndarray:

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    _, val_df = load_dataset()

    # --------------------------------------------------------
    # Load our model
    # --------------------------------------------------------

    our_model = load_our_model()

    # --------------------------------------------------------
    # Load Senanur models
    # --------------------------------------------------------

    senanur_sessions = (
        load_senanur_models()
    )

    senanur_thresholds = (
        load_senanur_thresholds()
    )

    # --------------------------------------------------------
    # Prediction storage
    # --------------------------------------------------------

    y_true = []

    our_predictions = []

    senanur_predictions = []

    per_image_rows = []

    our_times = []

    senanur_times = []

    # --------------------------------------------------------
    # Iterate over validation dataset
    # --------------------------------------------------------

    total_images = len(val_df)

    print()
    print("=" * 80)
    print("STARTING COMPARISON")
    print("=" * 80)
    print(
        f"Images to evaluate: {total_images}"
    )
    print()

    for index, row in val_df.iterrows():

        image_id = row["id_code"]

        true_grade = int(
            row["diagnosis"]
        )

        image_path = row[
            "image_path"
        ]

        print(
            f"[{index + 1:4d}/{total_images}] "
            f"{image_id}",
            end=" ",
            flush=True,
        )

        try:

            image = read_image(
                image_path
            )

            # ------------------------------------------------
            # Current DrishtiAI model
            # ------------------------------------------------

            (
                our_grade,
                our_confidence,
                our_probabilities,
                our_time,
            ) = predict_our_model(
                our_model,
                image,
            )

            # ------------------------------------------------
            # Senanur
            # ------------------------------------------------

            senanur_result = (
                predict_senanur(
                    senanur_sessions,
                    image,
                    senanur_thresholds,
                )
            )

            senanur_grade = int(
                senanur_result["grade"]
            )

            senanur_confidence = float(
                senanur_result["confidence"]
            )

            senanur_fold_scores = (
                senanur_result[
                    "fold_scores"
                ]
            )

            senanur_fold_grades = (
                senanur_result[
                    "fold_grades"
                ]
            )

            senanur_spread = float(
                senanur_result["spread"]
            )

            senanur_mean_score = float(
                senanur_result[
                    "mean_score"
                ]
            )

            senanur_time = float(
                senanur_result[
                    "inference_time"
                ]
            )

            # ------------------------------------------------
            # Save arrays
            # ------------------------------------------------

            y_true.append(
                true_grade
            )

            our_predictions.append(
                our_grade
            )

            senanur_predictions.append(
                senanur_grade
            )

            our_times.append(
                our_time
            )

            senanur_times.append(
                senanur_time
            )

            # ------------------------------------------------
            # Per-image record
            # ------------------------------------------------

            row_output = {
                "id_code": image_id,
                "image_path": image_path,
                "true_grade": true_grade,
                "true_label": CLASS_NAMES[
                    true_grade
                ],

                "our_grade": our_grade,
                "our_label": CLASS_NAMES[
                    our_grade
                ],
                "our_confidence": our_confidence,
                "our_correct": bool(
                    our_grade == true_grade
                ),
                "our_inference_seconds": our_time,

                "our_prob_grade_0": (
                    our_probabilities[0]
                ),
                "our_prob_grade_1": (
                    our_probabilities[1]
                ),
                "our_prob_grade_2": (
                    our_probabilities[2]
                ),
                "our_prob_grade_3": (
                    our_probabilities[3]
                ),
                "our_prob_grade_4": (
                    our_probabilities[4]
                ),

                "senanur_grade": senanur_grade,
                "senanur_label": CLASS_NAMES[
                    senanur_grade
                ],
                "senanur_confidence_estimate": (
                    senanur_confidence
                ),
                "senanur_correct": bool(
                    senanur_grade
                    == true_grade
                ),
                "senanur_mean_score": (
                    senanur_mean_score
                ),
                "senanur_spread": (
                    senanur_spread
                ),
                "senanur_inference_seconds": (
                    senanur_time
                ),

                "senanur_fold1_score": (
                    senanur_fold_scores[0]
                ),
                "senanur_fold2_score": (
                    senanur_fold_scores[1]
                ),
                "senanur_fold3_score": (
                    senanur_fold_scores[2]
                ),
                "senanur_fold4_score": (
                    senanur_fold_scores[3]
                ),
                "senanur_fold5_score": (
                    senanur_fold_scores[4]
                ),

                "senanur_fold1_grade": (
                    senanur_fold_grades[0]
                ),
                "senanur_fold2_grade": (
                    senanur_fold_grades[1]
                ),
                "senanur_fold3_grade": (
                    senanur_fold_grades[2]
                ),
                "senanur_fold4_grade": (
                    senanur_fold_grades[3]
                ),
                "senanur_fold5_grade": (
                    senanur_fold_grades[4]
                ),

                "models_agree": bool(
                    our_grade == senanur_grade
                ),

                "our_absolute_error": abs(
                    our_grade
                    - true_grade
                ),

                "senanur_absolute_error": abs(
                    senanur_grade
                    - true_grade
                ),
            }

            per_image_rows.append(
                row_output
            )

            print(
                f"| GT={true_grade} "
                f"| Ours={our_grade} "
                f"| Senanur={senanur_grade} "
                f"| Spread={senanur_spread:.3f}"
            )

        except Exception as exc:

            print(
                f"\nERROR: {exc}"
            )

            raise

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    our_predictions = np.asarray(
        our_predictions,
        dtype=np.int64,
    )

    senanur_predictions = np.asarray(
        senanur_predictions,
        dtype=np.int64,
    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    our_metrics = calculate_model_metrics(
        y_true,
        our_predictions,
    )

    senanur_metrics = calculate_model_metrics(
        y_true,
        senanur_predictions,
    )

    agreement = calculate_agreement(
        y_true,
        our_predictions,
        senanur_predictions,
    )

    # ========================================================
    # TIMING
    # ========================================================

    our_avg_time = float(
        np.mean(
            our_times
        )
    )

    our_total_time = float(
        np.sum(
            our_times
        )
    )

    senanur_avg_time = float(
        np.mean(
            senanur_times
        )
    )

    senanur_total_time = float(
        np.sum(
            senanur_times
        )
    )

    # ========================================================
    # UNIQUE WINS
    # ========================================================

    our_correct = (
        our_predictions
        == y_true
    )

    senanur_correct = (
        senanur_predictions
        == y_true
    )

    only_ours = (
        our_correct
        & ~senanur_correct
    )

    only_senanur = (
        ~our_correct
        & senanur_correct
    )

    # ========================================================
    # SAVE PER-IMAGE CSV
    # ========================================================

    per_image_df = pd.DataFrame(
        per_image_rows
    )

    csv_output = (
        OUTPUT_DIR
        / "per_image_predictions.csv"
    )

    per_image_df.to_csv(
        csv_output,
        index=False,
    )

    print()
    print(
        f"Saved per-image results:\n"
        f"{csv_output}"
    )

    # ========================================================
    # SAVE CONFUSION MATRICES
    # ========================================================

    our_cm = confusion_matrix(
        y_true,
        our_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    senanur_cm = confusion_matrix(
        y_true,
        senanur_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    np.savetxt(
        OUTPUT_DIR
        / "our_confusion_matrix.csv",
        our_cm,
        fmt="%d",
        delimiter=",",
    )

    np.savetxt(
        OUTPUT_DIR
        / "senanur_confusion_matrix.csv",
        senanur_cm,
        fmt="%d",
        delimiter=",",
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # --------------------------------------------------------
    # Our model
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("CURRENT DRISHTIAI MODEL")
    print("-" * 80)

    print(
        f"Accuracy                 : "
        f"{our_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision          : "
        f"{our_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall             : "
        f"{our_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1                 : "
        f"{our_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1              : "
        f"{our_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Quadratic Weighted Kappa : "
        f"{our_metrics['quadratic_weighted_kappa']:.4f}"
    )

    print(
        f"Referable Sensitivity    : "
        f"{our_metrics['referable_dr']['sensitivity']:.4f}"
    )

    print(
        f"Referable Specificity    : "
        f"{our_metrics['referable_dr']['specificity']:.4f}"
    )

    print(
        f"Average inference        : "
        f"{our_avg_time:.4f} sec/image"
    )

    print(
        f"Total inference          : "
        f"{our_total_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Senanur
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("SENANUR 5-FOLD MODEL")
    print("-" * 80)

    print(
        f"Accuracy                 : "
        f"{senanur_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision          : "
        f"{senanur_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall             : "
        f"{senanur_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1                 : "
        f"{senanur_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1              : "
        f"{senanur_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Quadratic Weighted Kappa : "
        f"{senanur_metrics['quadratic_weighted_kappa']:.4f}"
    )

    print(
        f"Referable Sensitivity    : "
        f"{senanur_metrics['referable_dr']['sensitivity']:.4f}"
    )

    print(
        f"Referable Specificity    : "
        f"{senanur_metrics['referable_dr']['specificity']:.4f}"
    )

    print(
        f"Average inference        : "
        f"{senanur_avg_time:.4f} sec/image"
    )

    print(
        f"Total inference          : "
        f"{senanur_total_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("MODEL AGREEMENT")
    print("-" * 80)

    print(
        f"Agreement rate           : "
        f"{agreement['agreement_rate']:.4f}"
    )

    print(
        f"Agreement count          : "
        f"{agreement['agreement_count']}"
    )

    print(
        f"Both correct             : "
        f"{agreement['both_correct']}"
    )

    print(
        f"Only our model correct   : "
        f"{agreement['only_our_model_correct']}"
    )

    print(
        f"Only Senanur correct     : "
        f"{agreement['only_senanur_correct']}"
    )

    print(
        f"Neither correct          : "
        f"{agreement['neither_correct']}"
    )

    # --------------------------------------------------------
    # Unique winner percentages
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("UNIQUE WINS")
    print("-" * 80)

    print(
        f"Our model only correct  : "
        f"{int(only_ours.sum())}"
        f" / {len(y_true)}"
        f" = {only_ours.mean():.4f}"
    )

    print(
        f"Senanur only correct    : "
        f"{int(only_senanur.sum())}"
        f" / {len(y_true)}"
        f" = {only_senanur.mean():.4f}"
    )

    # ========================================================
    # PER-GRADE RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("PER-GRADE RESULTS")
    print("=" * 80)

    our_report = (
        our_metrics[
            "classification_report"
        ]
    )

    senanur_report = (
        senanur_metrics[
            "classification_report"
        ]
    )

    print()

    print(
        f"{'Grade':<22}"
        f"{'Our F1':>12}"
        f"{'Senanur F1':>15}"
    )

    print("-" * 52)

    for grade in range(NUM_CLASSES):

        name = CLASS_NAMES[
            grade
        ]

        our_f1 = our_report[
            name
        ]["f1-score"]

        senanur_f1 = senanur_report[
            name
        ]["f1-score"]

        print(
            f"{name:<22}"
            f"{our_f1:>12.4f}"
            f"{senanur_f1:>15.4f}"
        )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print()
    print("=" * 80)
    print("CURRENT MODEL CONFUSION MATRIX")
    print("=" * 80)

    print(
        pd.DataFrame(
            our_cm,
            index=[
                f"True {i}"
                for i in range(NUM_CLASSES)
            ],
            columns=[
                f"Pred {i}"
                for i in range(NUM_CLASSES)
            ],
        )
    )

    print()
    print("=" * 80)
    print("SENANUR CONFUSION MATRIX")
    print("=" * 80)

    print(
        pd.DataFrame(
            senanur_cm,
            index=[
                f"True {i}"
                for i in range(NUM_CLASSES)
            ],
            columns=[
                f"Pred {i}"
                for i in range(NUM_CLASSES)
            ],
        )
    )

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    metrics_to_compare = [
        (
            "Accuracy",
            our_metrics["accuracy"],
            senanur_metrics["accuracy"],
        ),
        (
            "Macro F1",
            our_metrics["macro_f1"],
            senanur_metrics["macro_f1"],
        ),
        (
            "Weighted F1",
            our_metrics["weighted_f1"],
            senanur_metrics["weighted_f1"],
        ),
        (
            "QWK",
            our_metrics[
                "quadratic_weighted_kappa"
            ],
            senanur_metrics[
                "quadratic_weighted_kappa"
            ],
        ),
        (
            "Referable Sensitivity",
            our_metrics[
                "referable_dr"
            ]["sensitivity"],
            senanur_metrics[
                "referable_dr"
            ]["sensitivity"],
        ),
        (
            "Referable Specificity",
            our_metrics[
                "referable_dr"
            ]["specificity"],
            senanur_metrics[
                "referable_dr"
            ]["specificity"],
        ),
    ]

    print()

    print(
        f"{'Metric':<28}"
        f"{'DrishtiAI':>15}"
        f"{'Senanur':>15}"
        f"{'Winner':>15}"
    )

    print("-" * 73)

    for metric, ours, senanur in metrics_to_compare:

        if (
            np.isclose(
                ours,
                senanur,
                atol=1e-8,
            )
        ):
            winner = "Tie"

        elif senanur > ours:
            winner = "Senanur"

        else:
            winner = "DrishtiAI"

        print(
            f"{metric:<28}"
            f"{ours:>15.4f}"
            f"{senanur:>15.4f}"
            f"{winner:>15}"
        )

    # ========================================================
    # DECIDE PRIMARY MODEL
    # ========================================================

    our_qwk = our_metrics[
        "quadratic_weighted_kappa"
    ]

    senanur_qwk = senanur_metrics[
        "quadratic_weighted_kappa"
    ]

    our_f1 = our_metrics[
        "macro_f1"
    ]

    senanur_f1 = senanur_metrics[
        "macro_f1"
    ]

    if (
        senanur_qwk > our_qwk
        and senanur_f1 > our_f1
    ):
        recommended_model = (
            "Senanur 5-fold ONNX"
        )

    elif (
        our_qwk > senanur_qwk
        and our_f1 > senanur_f1
    ):
        recommended_model = (
            "Current DrishtiAI model"
        )

    else:
        recommended_model = (
            "Needs further analysis"
        )

    print()
    print(
        f"Recommended based on QWK + Macro F1: "
        f"{recommended_model}"
    )

    # ========================================================
    # SAVE FINAL REPORT
    # ========================================================

    report = {
        "dataset": {
            "name": "APTOS 2019",
            "total_available_images": int(
                len(val_df)
            ),
            "evaluation_images": int(
                len(y_true)
            ),
            "validation_split": {
                "method": "80/20 stratified",
                "random_state": RANDOM_STATE,
            },
        },

        "current_drishtiai_model": {
            "checkpoint": str(
                OUR_MODEL_PATH
            ),
            "architecture": (
                "torchvision EfficientNet-B0"
            ),
            "metrics": our_metrics,
            "average_inference_seconds": (
                our_avg_time
            ),
            "total_inference_seconds": (
                our_total_time
            ),
        },

        "senanur_model": {
            "directory": str(
                SENANUR_DIR
            ),
            "architecture": (
                "5-fold EfficientNet-B0 ordinal ONNX"
            ),
            "thresholds": (
                senanur_thresholds
            ),
            "metrics": senanur_metrics,
            "average_inference_seconds": (
                senanur_avg_time
            ),
            "total_inference_seconds": (
                senanur_total_time
            ),
        },

        "agreement": agreement,

        "unique_wins": {
            "only_our_model_correct": int(
                only_ours.sum()
            ),
            "only_senanur_correct": int(
                only_senanur.sum()
            ),
        },

        "recommended_model": (
            recommended_model
        ),

        "important_note": (
            "Senanur confidence is an ordinal "
            "score-distance estimate and is not "
            "a calibrated class probability."
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "comparison_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved comparison report:\n"
        f"{report_path}"
    )

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print()
    print(
        "Generated files:"
    )

    print(
        f"  {OUTPUT_DIR / 'comparison_report.json'}"
    )

    print(
        f"  {OUTPUT_DIR / 'per_image_predictions.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'our_confusion_matrix.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'senanur_confusion_matrix.csv'}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()