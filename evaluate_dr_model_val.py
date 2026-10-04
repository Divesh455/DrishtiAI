from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

VAL_CSV = ROOT / "data" / "splits" / "aptos_clean_val.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "train_images"
MODEL_PATH = ROOT / "weights" / "dr_model_clean.pth"

OUTPUT_DIR = ROOT / "dr_model_clean_val_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# DATASET
# ============================================================

class AptosDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        image_path = None

        for ext in [".png", ".jpg", ".jpeg",
                    ".PNG", ".JPG", ".JPEG"]:

            candidate = IMAGE_DIR / f"{image_id}{ext}"

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(
                f"Image not found: {image_id}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize exactly for model input
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # float [0,1]
        image = image.astype(
            np.float32
        ) / 255.0

        # ImageNet normalization
        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        return image, label, image_id


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model):

    print()
    print("=" * 70)
    print("LOADING DRISHTIAI MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    print(
        "Checkpoint type:",
        type(checkpoint)
    )

    # --------------------------------------------------------
    # Case 1: checkpoint dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        print(
            "Checkpoint keys:",
            list(checkpoint.keys())
        )

        state_dict = None

        for key in [
            "model_state_dict",
            "state_dict",
            "model"
        ]:

            if key in checkpoint:

                candidate = checkpoint[key]

                if isinstance(candidate, dict):

                    state_dict = candidate
                    print(
                        f"Using state dict key: {key}"
                    )
                    break

        # Sometimes checkpoint itself is state_dict
        if state_dict is None:

            tensor_values = [
                v for v in checkpoint.values()
                if torch.is_tensor(v)
            ]

            if len(tensor_values) > 0:

                state_dict = checkpoint

                print(
                    "Using checkpoint directly "
                    "as state_dict."
                )

        if state_dict is None:

            raise RuntimeError(
                "Could not find model state_dict "
                "inside checkpoint."
            )

    # --------------------------------------------------------
    # Case 2: raw state_dict
    # --------------------------------------------------------

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        print()
        print("WARNING - Missing keys:")
        for key in missing:
            print(key)

    if unexpected:
        print()
        print("WARNING - Unexpected keys:")
        for key in unexpected:
            print(key)

    model.to(DEVICE)
    model.eval()

    print()
    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model, checkpoint


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DRISHTIAI CLEAN MODEL VALIDATION")
    print("=" * 80)

    print()
    print("Validation CSV:")
    print(VAL_CSV)

    print()
    print("Model:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not VAL_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV not found:\n{VAL_CSV}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Read validation CSV
    # --------------------------------------------------------

    df = pd.read_csv(VAL_CSV)

    print()
    print(
        f"Validation images: {len(df)}"
    )

    print()
    print("Class distribution:")
    print(
        df["diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = AptosDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_checkpoint(
        build_model()
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_true = []
    all_pred = []
    all_confidence = []
    all_ids = []
    all_inference = []

    total_start = time.perf_counter()

    with torch.no_grad():

        for batch_idx, (
            images,
            labels,
            image_ids
        ) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            start = time.perf_counter()

            logits = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, predictions = torch.max(
                probabilities,
                dim=1
            )

            all_true.extend(
                labels.cpu().numpy().tolist()
            )

            all_pred.extend(
                predictions.cpu().numpy().tolist()
            )

            all_confidence.extend(
                confidence.cpu().numpy().tolist()
            )

            all_ids.extend(
                list(image_ids)
            )

            batch_image_time = (
                elapsed / len(images)
            )

            all_inference.extend(
                [batch_image_time] * len(images)
            )

            processed = min(
                (batch_idx + 1) * BATCH_SIZE,
                len(dataset)
            )

            if (
                batch_idx == 0
                or processed % 50 == 0
                or processed == len(dataset)
            ):

                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    y_true = np.asarray(
        all_true,
        dtype=np.int64
    )

    y_pred = np.asarray(
        all_pred,
        dtype=np.int64
    )

    confidence = np.asarray(
        all_confidence,
        dtype=np.float32
    )

    inference_times = np.asarray(
        all_inference,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    macro_precision = (
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    macro_recall = (
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Referable DR
    # Grade >= 2
    # --------------------------------------------------------

    true_referable = (
        y_true >= 2
    ).astype(int)

    pred_referable = (
        y_pred >= 2
    ).astype(int)

    referable_sensitivity = (
        recall_score(
            true_referable,
            pred_referable,
            zero_division=0
        )
    )

    referable_specificity = (
        recall_score(
            1 - true_referable,
            1 - pred_referable,
            zero_division=0
        )
    )

    referable_precision = (
        precision_score(
            true_referable,
            pred_referable,
            zero_division=0
        )
    )

    referable_f1 = (
        f1_score(
            true_referable,
            pred_referable,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Grade distance
    # --------------------------------------------------------

    absolute_error = np.abs(
        y_true - y_pred
    )

    exact_correct = int(
        np.sum(
            absolute_error == 0
        )
    )

    off_by_1 = int(
        np.sum(
            absolute_error == 1
        )
    )

    off_by_2_or_more = int(
        np.sum(
            absolute_error >= 2
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative"
        ],
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "True_0",
            "True_1",
            "True_2",
            "True_3",
            "True_4"
        ],
        columns=[
            "Pred_0",
            "Pred_1",
            "Pred_2",
            "Pred_3",
            "Pred_4"
        ]
    )

    # --------------------------------------------------------
    # Predictions dataframe
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "id_code": all_ids,
            "true_grade": y_true,
            "predicted_grade": y_pred,
            "confidence": confidence,
            "absolute_error": absolute_error
        }
    )

    predictions_path = (
        OUTPUT_DIR
        / "dr_model_clean_val_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        OUTPUT_DIR
        / "dr_model_clean_val_confusion_matrix.csv"
    )

    cm_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    mean_inference = float(
        np.mean(inference_times)
    )

    median_inference = float(
        np.median(inference_times)
    )

    mean_confidence = float(
        np.mean(confidence)
    )

    median_confidence = float(
        np.median(confidence)
    )

    min_confidence = float(
        np.min(confidence)
    )

    max_confidence = float(
        np.max(confidence)
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    final_report = {

        "dataset": "APTOS clean validation set",

        "images": int(len(y_true)),

        "model": "DrishtiAI clean EfficientNet-B0",

        "device": str(DEVICE),

        "metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "macro_precision": float(
                macro_precision
            ),
            "macro_recall": float(
                macro_recall
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
            "mean_absolute_error": float(
                mae
            )
        },

        "referable_dr": {
            "definition": "Grade >= 2",
            "sensitivity": float(
                referable_sensitivity
            ),
            "specificity": float(
                referable_specificity
            ),
            "precision": float(
                referable_precision
            ),
            "f1": float(
                referable_f1
            )
        },

        "grade_error": {
            "exact_correct": exact_correct,
            "exact_correct_rate": float(
                exact_correct / len(y_true)
            ),
            "off_by_1": off_by_1,
            "off_by_1_rate": float(
                off_by_1 / len(y_true)
            ),
            "off_by_2_or_more": off_by_2_or_more,
            "off_by_2_or_more_rate": float(
                off_by_2_or_more / len(y_true)
            )
        },

        "confidence": {
            "mean": mean_confidence,
            "median": median_confidence,
            "min": min_confidence,
            "max": max_confidence
        },

        "inference": {
            "mean_sec_per_image": mean_inference,
            "median_sec_per_image": median_inference,
            "total_sec": float(total_elapsed)
        },

        "per_grade": report
    }

    report_path = (
        OUTPUT_DIR
        / "dr_model_clean_val_evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=4
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("DRISHTIAI CLEAN MODEL VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"Images evaluated          : {len(y_true)}"
    )

    print(
        f"Accuracy                  : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision           : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall              : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1                  : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1               : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Quadratic Weighted Kappa  : "
        f"{qwk:.4f}"
    )

    print(
        f"Mean Absolute Error       : "
        f"{mae:.4f}"
    )

    print()
    print("Referable DR (Grade >= 2)")

    print(
        f"Sensitivity               : "
        f"{referable_sensitivity:.4f}"
    )

    print(
        f"Specificity               : "
        f"{referable_specificity:.4f}"
    )

    print(
        f"Precision                 : "
        f"{referable_precision:.4f}"
    )

    print(
        f"F1                        : "
        f"{referable_f1:.4f}"
    )

    print()
    print("Grade error distance")

    print(
        f"Exact correct             : "
        f"{exact_correct}/{len(y_true)} "
        f"({exact_correct / len(y_true):.4f})"
    )

    print(
        f"Off by 1 grade            : "
        f"{off_by_1}/{len(y_true)} "
        f"({off_by_1 / len(y_true):.4f})"
    )

    print(
        f"Off by >=2 grades         : "
        f"{off_by_2_or_more}/{len(y_true)} "
        f"({off_by_2_or_more / len(y_true):.4f})"
    )

    print()
    print("Confidence")

    print(
        f"Mean                      : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Median                    : "
        f"{median_confidence:.4f}"
    )

    print(
        f"Minimum                   : "
        f"{min_confidence:.4f}"
    )

    print(
        f"Maximum                   : "
        f"{max_confidence:.4f}"
    )

    print()
    print("Inference")

    print(
        f"Average sec/image         : "
        f"{mean_inference:.6f}"
    )

    print(
        f"Median sec/image          : "
        f"{median_inference:.6f}"
    )

    print()
    print("Per-grade metrics")

    per_grade_df = pd.DataFrame(
        report
    ).T[
        [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
    ]

    print(
        per_grade_df.to_string()
    )

    print()
    print("Confusion Matrix")

    print(cm_df)

    print()
    print("Saved:")
    print(predictions_path)
    print(confusion_path)
    print(report_path)

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()