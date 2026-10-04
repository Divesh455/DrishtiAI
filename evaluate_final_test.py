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

TEST_CSV = (
    ROOT
    / "data"
    / "splits"
    / "aptos_clean_test.csv"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train_images"
)

MODEL_PATH = (
    ROOT
    / "weights"
    / "dr_model_final.pth"
)

OUTPUT_DIR = (
    ROOT
    / "dr_model_final_test_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

class APTOSDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_dir = Path(
            image_dir
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = str(
            row["id_code"]
        )

        label = int(
            row["diagnosis"]
        )

        image_path = None

        for ext in [
            ".png",
            ".jpg",
            ".jpeg",
            ".PNG",
            ".JPG",
            ".JPEG"
        ]:

            candidate = (
                self.image_dir
                / f"{image_id}{ext}"
            )

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
                f"Could not read image: "
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            ),
            image_id
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING FINAL DRISHTIAI MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    print(
        "Checkpoint type:",
        type(checkpoint)
    )

    if isinstance(
        checkpoint,
        dict
    ):

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

                if isinstance(
                    candidate,
                    dict
                ):

                    state_dict = candidate

                    print(
                        f"Using state dict key: {key}"
                    )

                    break

        if state_dict is None:

            tensor_items = {
                k: v
                for k, v in checkpoint.items()
                if torch.is_tensor(v)
            }

            if tensor_items:

                state_dict = checkpoint

                print(
                    "Using checkpoint directly "
                    "as state_dict."
                )

    else:

        state_dict = checkpoint

    if state_dict is None:

        raise RuntimeError(
            "Could not locate model state_dict."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[7:]

        cleaned_state_dict[
            new_key
        ] = value

    model = build_model()

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    if missing:

        print()
        print(
            "WARNING - Missing keys:"
        )

        for key in missing:

            print(key)

    if unexpected:

        print()
        print(
            "WARNING - Unexpected keys:"
        )

        for key in unexpected:

            print(key)

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "Final model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL DRISHTIAI TEST EVALUATION")
    print("=" * 80)

    print()
    print(
        "Test CSV:",
        TEST_CSV
    )

    print()
    print(
        "Model:",
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Final model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Read test CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        TEST_CSV
    )

    print()
    print(
        f"Test images: {len(df)}"
    )

    if len(df) != 354:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected 354 images, "
            f"found {len(df)}."
        )

    required = {
        "id_code",
        "diagnosis"
    }

    missing_columns = (
        required
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}"
        )

    print()
    print(
        "Test class distribution:"
    )

    print(
        df[
            "diagnosis"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = APTOSDataset(
        df,
        IMAGE_DIR
    )

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
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_true = []

    all_pred = []

    all_confidence = []

    all_ids = []

    inference_times = []

    total_start = time.perf_counter()

    with torch.no_grad():

        for batch_index, (
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

            if DEVICE.type == "cuda":

                torch.cuda.synchronize()

            start = time.perf_counter()

            logits = model(
                images
            )

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

            confidence, predictions = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            batch_time_per_image = (
                elapsed
                / len(images)
            )

            all_true.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_pred.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            all_confidence.extend(
                confidence
                .cpu()
                .numpy()
                .tolist()
            )

            all_ids.extend(
                list(image_ids)
            )

            inference_times.extend(
                [
                    batch_time_per_image
                ]
                * len(images)
            )

            processed = min(
                (batch_index + 1)
                * BATCH_SIZE,
                len(dataset)
            )

            if (
                batch_index == 0
                or processed
                == len(dataset)
                or processed % 64 == 0
            ):

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(dataset)}"
                )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Convert
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
        inference_times,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Metrics
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
    # Error distance
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

    off_by_2_plus = int(
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
        labels=[
            0,
            1,
            2,
            3,
            4
        ],
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
        labels=[
            0,
            1,
            2,
            3,
            4
        ]
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
    # Predictions CSV
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
        / "dr_model_final_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Confusion CSV
    # --------------------------------------------------------

    confusion_path = (
        OUTPUT_DIR
        / "dr_model_final_test_confusion_matrix.csv"
    )

    cm_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    mean_inference = float(
        np.mean(
            inference_times
        )
    )

    median_inference = float(
        np.median(
            inference_times
        )
    )

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    final_report = {

        "dataset":
            "APTOS clean untouched test set",

        "images":
            int(len(y_true)),

        "model":
            "DrishtiAI final EfficientNet-B0",

        "model_path":
            str(MODEL_PATH),

        "metrics": {

            "accuracy":
                float(accuracy),

            "balanced_accuracy":
                float(
                    balanced_accuracy
                ),

            "macro_precision":
                float(
                    macro_precision
                ),

            "macro_recall":
                float(
                    macro_recall
                ),

            "macro_f1":
                float(
                    macro_f1
                ),

            "weighted_f1":
                float(
                    weighted_f1
                ),

            "quadratic_weighted_kappa":
                float(qwk),

            "mean_absolute_error":
                float(mae)
        },

        "referable_dr": {

            "definition":
                "Grade >= 2",

            "sensitivity":
                float(
                    referable_sensitivity
                ),

            "specificity":
                float(
                    referable_specificity
                ),

            "precision":
                float(
                    referable_precision
                ),

            "f1":
                float(
                    referable_f1
                )
        },

        "grade_error": {

            "exact_correct":
                exact_correct,

            "exact_correct_rate":
                float(
                    exact_correct
                    / len(y_true)
                ),

            "off_by_1":
                off_by_1,

            "off_by_1_rate":
                float(
                    off_by_1
                    / len(y_true)
                ),

            "off_by_2_plus":
                off_by_2_plus,

            "off_by_2_plus_rate":
                float(
                    off_by_2_plus
                    / len(y_true)
                )
        },

        "confidence": {

            "mean":
                float(
                    np.mean(confidence)
                ),

            "median":
                float(
                    np.median(confidence)
                ),

            "minimum":
                float(
                    np.min(confidence)
                ),

            "maximum":
                float(
                    np.max(confidence)
                )
        },

        "inference": {

            "average_sec_per_image":
                mean_inference,

            "median_sec_per_image":
                median_inference,

            "total_sec":
                float(total_elapsed)
        },

        "per_grade":
            report
    }

    report_path = (
        OUTPUT_DIR
        / "dr_model_final_test_report.json"
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
    print("FINAL DRISHTIAI TEST RESULTS")
    print("=" * 80)

    print(
        f"Test images               : "
        f"{len(y_true)}"
    )

    print(
        f"Accuracy                  : "
        f"{accuracy:.4f}"
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
    print(
        "Referable DR (Grade >= 2)"
    )

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
    print(
        "Grade error distance"
    )

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
        f"{off_by_2_plus}/{len(y_true)} "
        f"({off_by_2_plus / len(y_true):.4f})"
    )

    print()
    print(
        "Confidence"
    )

    print(
        f"Mean                      : "
        f"{np.mean(confidence):.4f}"
    )

    print(
        f"Median                    : "
        f"{np.median(confidence):.4f}"
    )

    print(
        f"Minimum                   : "
        f"{np.min(confidence):.4f}"
    )

    print(
        f"Maximum                   : "
        f"{np.max(confidence):.4f}"
    )

    print()
    print(
        "Inference"
    )

    print(
        f"Average sec/image         : "
        f"{mean_inference:.6f}"
    )

    print(
        f"Median sec/image          : "
        f"{median_inference:.6f}"
    )

    print()
    print(
        "Per-grade metrics"
    )

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
    print(
        "Confusion Matrix"
    )

    print(
        cm_df
    )

    print()
    print(
        "Saved:"
    )

    print(
        predictions_path
    )

    print(
        confusion_path
    )

    print(
        report_path
    )

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()