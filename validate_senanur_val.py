from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

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

SENANUR_DIR = ROOT / "external_models" / "senanur_dr"

OUTPUT_DIR = ROOT / "senanur_val_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLDS = [0.668, 1.132, 2.324, 3.3]

INPUT_SIZE = 384

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".PNG",
    ".JPG",
    ".JPEG",
]


# ============================================================
# FUNDUS PREPROCESSING
# ============================================================

def crop_image_from_gray(img, tol=7):
    """
    Remove dark/black borders around fundus image.

    Senanur preprocessing uses tolerance=7.
    """

    if img is None:
        raise ValueError("Image is None.")

    if len(img.shape) == 2:
        mask = img > tol
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > tol

    if not np.any(mask):
        return img

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img

    return img[
        rows[0]:rows[-1] + 1,
        cols[0]:cols[-1] + 1
    ]


def preprocess_senanur(image_path):
    """
    Senanur-style preprocessing:

        1. Read BGR
        2. Auto crop using tol=7
        3. Resize preserving aspect ratio so max dimension = 512
        4. Pad to 512x512
        5. Resize to 384x384
        6. BGR -> RGB
        7. /255
        8. ImageNet normalization
        9. CHW
        10. batch dimension
    """

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # --------------------------------------------------------
    # 1. Crop dark borders
    # --------------------------------------------------------
    img = crop_image_from_gray(img, tol=7)

    if img.size == 0:
        raise ValueError(f"Empty cropped image: {image_path}")

    # --------------------------------------------------------
    # 2. Resize preserving aspect ratio to max dimension 512
    # --------------------------------------------------------
    h, w = img.shape[:2]

    scale = 512.0 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 3. Pad to 512x512
    # --------------------------------------------------------
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)

    y_offset = (512 - new_h) // 2
    x_offset = (512 - new_w) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = img

    # --------------------------------------------------------
    # 4. Resize to model input
    # --------------------------------------------------------
    img = cv2.resize(
        canvas,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 5. BGR -> RGB
    # --------------------------------------------------------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # 6. Convert to float
    # --------------------------------------------------------
    img = img.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # 7. ImageNet normalization
    # --------------------------------------------------------
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img = (img - mean) / std

    # --------------------------------------------------------
    # 8. HWC -> CHW
    # --------------------------------------------------------
    img = np.transpose(img, (2, 0, 1))

    # --------------------------------------------------------
    # 9. Add batch dimension
    # --------------------------------------------------------
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(image_id):
    """
    Find image by id_code.
    """

    image_id = str(image_id)

    # Direct common extensions first
    for ext in IMAGE_EXTENSIONS:
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path

    # Fallback: search filename
    matches = list(IMAGE_DIR.glob(f"{image_id}.*"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Image not found for id_code={image_id}"
        )

    return matches[0]


# ============================================================
# LOAD SENANUR MODELS
# ============================================================

def load_models():
    sessions = []

    for fold in range(1, 6):

        model_path = SENANUR_DIR / f"fold{fold}.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print()
        print("=" * 70)
        print(f"Fold {fold}")
        print("=" * 70)

        print("Input name :", inputs[0].name)
        print("Input shape:", inputs[0].shape)
        print("Input type :", inputs[0].type)

        print("Output name :", outputs[0].name)
        print("Output shape:", outputs[0].shape)
        print("Output type :", outputs[0].type)

        sessions.append(session)

    return sessions


# ============================================================
# CONVERT RAW SCORE -> GRADE
# ============================================================

def score_to_grade(score):
    """
    Convert Senanur raw ordinal score to grade 0-4.
    """

    if score < THRESHOLDS[0]:
        return 0

    if score < THRESHOLDS[1]:
        return 1

    if score < THRESHOLDS[2]:
        return 2

    if score < THRESHOLDS[3]:
        return 3

    return 4


# ============================================================
# RUN ONE IMAGE
# ============================================================

def predict_one(image_path, sessions):
    """
    Run all 5 Senanur folds.

    Returns:
        mean raw score
        predicted grade
        fold spread
    """

    image = preprocess_senanur(image_path)

    fold_scores = []

    input_name = sessions[0].get_inputs()[0].name

    for session in sessions:

        outputs = session.run(
            None,
            {
                input_name: image
            }
        )

        if not outputs:
            raise RuntimeError(
                "ONNX model returned no outputs."
            )

        raw_output = np.asarray(outputs[0])

        # Expected published model behavior:
        # one scalar raw score per image
        raw_output = raw_output.reshape(-1)

        if raw_output.size == 0:
            raise RuntimeError(
                f"Empty output from model: {image_path}"
            )

        if raw_output.size > 1:
            raise RuntimeError(
                "Unexpected ONNX output shape.\n"
                f"Output shape: {raw_output.shape}\n"
                f"Values: {raw_output}"
            )

        score = float(raw_output[0])

        fold_scores.append(score)

    fold_scores = np.array(
        fold_scores,
        dtype=np.float32
    )

    mean_score = float(np.mean(fold_scores))
    fold_spread = float(np.std(fold_scores))

    predicted_grade = score_to_grade(mean_score)

    return (
        mean_score,
        predicted_grade,
        fold_spread,
        fold_scores
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SENANUR VALIDATION ON CLEAN APTOS VALIDATION SET")
    print("=" * 80)

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

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    df = pd.read_csv(VAL_CSV)

    print()
    print("Validation CSV:", VAL_CSV)
    print("Rows:", len(df))

    required_columns = {"id_code", "diagnosis"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print()
    print("Validation class distribution:")
    print(
        df["diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    sessions = load_models()

    print()
    print("Loaded 5 Senanur ONNX folds successfully.")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    records = []

    total_start = time.perf_counter()

    for idx, row in df.iterrows():

        image_id = row["id_code"]
        true_grade = int(row["diagnosis"])

        try:
            image_path = find_image(image_id)

            start = time.perf_counter()

            (
                raw_score,
                predicted_grade,
                fold_spread,
                fold_scores
            ) = predict_one(
                image_path,
                sessions
            )

            elapsed = time.perf_counter() - start

            records.append(
                {
                    "id_code": image_id,
                    "image_path": str(image_path),
                    "true_grade": true_grade,
                    "predicted_grade": predicted_grade,
                    "raw_score": raw_score,
                    "fold_spread": fold_spread,
                    "fold1_score": float(fold_scores[0]),
                    "fold2_score": float(fold_scores[1]),
                    "fold3_score": float(fold_scores[2]),
                    "fold4_score": float(fold_scores[3]),
                    "fold5_score": float(fold_scores[4]),
                    "inference_sec": elapsed,
                    "error": ""
                }
            )

        except Exception as exc:

            print(
                f"\nERROR: {image_id}: {exc}"
            )

            records.append(
                {
                    "id_code": image_id,
                    "image_path": "",
                    "true_grade": true_grade,
                    "predicted_grade": -1,
                    "raw_score": np.nan,
                    "fold_spread": np.nan,
                    "fold1_score": np.nan,
                    "fold2_score": np.nan,
                    "fold3_score": np.nan,
                    "fold4_score": np.nan,
                    "fold5_score": np.nan,
                    "inference_sec": np.nan,
                    "error": str(exc)
                }
            )

        if (idx + 1) % 25 == 0 or idx == 0:
            print(
                f"Processed {idx + 1}/{len(df)}"
            )

    total_elapsed = time.perf_counter() - total_start

    results = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save raw predictions
    # --------------------------------------------------------

    predictions_path = (
        OUTPUT_DIR /
        "senanur_val_predictions.csv"
    )

    results.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Keep valid predictions only
    # --------------------------------------------------------

    valid = results[
        results["predicted_grade"] >= 0
    ].copy()

    if len(valid) == 0:
        raise RuntimeError(
            "No valid predictions were produced."
        )

    y_true = valid["true_grade"].astype(int).to_numpy()
    y_pred = valid["predicted_grade"].astype(int).to_numpy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
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
    # Referable DR: Grade >= 2
    # --------------------------------------------------------

    y_true_ref = (y_true >= 2).astype(int)
    y_pred_ref = (y_pred >= 2).astype(int)

    ref_sensitivity = recall_score(
        y_true_ref,
        y_pred_ref,
        zero_division=0
    )

    ref_specificity = recall_score(
        1 - y_true_ref,
        1 - y_pred_ref,
        zero_division=0
    )

    ref_precision = precision_score(
        y_true_ref,
        y_pred_ref,
        zero_division=0
    )

    ref_f1 = f1_score(
        y_true_ref,
        y_pred_ref,
        zero_division=0
    )

    # --------------------------------------------------------
    # Error distance
    # --------------------------------------------------------

    abs_error = np.abs(
        y_true - y_pred
    )

    exact_correct = int(
        np.sum(abs_error == 0)
    )

    off_by_1 = int(
        np.sum(abs_error == 1)
    )

    off_by_2_or_more = int(
        np.sum(abs_error >= 2)
    )

    # --------------------------------------------------------
    # Per-grade report
    # --------------------------------------------------------

    report_dict = classification_report(
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

    confusion_path = (
        OUTPUT_DIR /
        "senanur_val_confusion_matrix.csv"
    )

    cm_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Confidence / fold spread
    # --------------------------------------------------------

    mean_fold_spread = float(
        valid["fold_spread"].mean()
    )

    median_fold_spread = float(
        valid["fold_spread"].median()
    )

    max_fold_spread = float(
        valid["fold_spread"].max()
    )

    mean_inference = float(
        valid["inference_sec"].mean()
    )

    median_inference = float(
        valid["inference_sec"].median()
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    evaluation_report = {
        "dataset": "APTOS clean validation set",
        "validation_images_expected": int(len(df)),
        "validation_images_evaluated": int(len(valid)),
        "failed_images": int(len(df) - len(valid)),

        "model": "Senanur 5-fold EfficientNet-B0 ONNX ensemble",

        "thresholds": THRESHOLDS,

        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "quadratic_weighted_kappa": float(qwk),
        "mean_absolute_error": float(mae),

        "referable_definition": "Grade >= 2",
        "referable_sensitivity": float(ref_sensitivity),
        "referable_specificity": float(ref_specificity),
        "referable_precision": float(ref_precision),
        "referable_f1": float(ref_f1),

        "exact_correct": exact_correct,
        "exact_correct_rate": float(
            exact_correct / len(valid)
        ),

        "off_by_1": off_by_1,
        "off_by_1_rate": float(
            off_by_1 / len(valid)
        ),

        "off_by_2_or_more": off_by_2_or_more,
        "off_by_2_or_more_rate": float(
            off_by_2_or_more / len(valid)
        ),

        "mean_fold_spread": mean_fold_spread,
        "median_fold_spread": median_fold_spread,
        "max_fold_spread": max_fold_spread,

        "mean_inference_sec_per_image": mean_inference,
        "median_inference_sec_per_image": median_inference,
        "total_inference_sec": float(total_elapsed),

        "per_grade": report_dict
    }

    report_path = (
        OUTPUT_DIR /
        "senanur_val_evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation_report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SENANUR VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"Images evaluated          : {len(valid)}"
    )

    print(
        f"Accuracy                  : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy         : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro Precision           : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall              : {macro_recall:.4f}"
    )

    print(
        f"Macro F1                  : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1               : {weighted_f1:.4f}"
    )

    print(
        f"Quadratic Weighted Kappa  : {qwk:.4f}"
    )

    print(
        f"Mean Absolute Error       : {mae:.4f}"
    )

    print()
    print("Referable DR (Grade >= 2)")
    print(
        f"Sensitivity               : {ref_sensitivity:.4f}"
    )

    print(
        f"Specificity               : {ref_specificity:.4f}"
    )

    print(
        f"Precision                 : {ref_precision:.4f}"
    )

    print(
        f"F1                        : {ref_f1:.4f}"
    )

    print()
    print("Grade error distance")

    print(
        f"Exact correct             : "
        f"{exact_correct}/{len(valid)} "
        f"({exact_correct / len(valid):.4f})"
    )

    print(
        f"Off by 1 grade            : "
        f"{off_by_1}/{len(valid)} "
        f"({off_by_1 / len(valid):.4f})"
    )

    print(
        f"Off by >=2 grades         : "
        f"{off_by_2_or_more}/{len(valid)} "
        f"({off_by_2_or_more / len(valid):.4f})"
    )

    print()
    print("Fold agreement")

    print(
        f"Mean fold spread          : "
        f"{mean_fold_spread:.6f}"
    )

    print(
        f"Median fold spread        : "
        f"{median_fold_spread:.6f}"
    )

    print(
        f"Max fold spread           : "
        f"{max_fold_spread:.6f}"
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
    print(
        pd.DataFrame(report_dict).T[
            [
                "precision",
                "recall",
                "f1-score",
                "support"
            ]
        ].to_string()
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