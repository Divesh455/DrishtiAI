from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DR_PATH = (
    ROOT
    / "dr_model_clean_val_evaluation"
    / "dr_model_clean_val_predictions.csv"
)

SENANUR_PATH = (
    ROOT
    / "senanur_val_evaluation"
    / "senanur_val_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "ensemble_val_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def round_half_up(x):
    """
    Normal rounding for positive grade values.

    Example:
        1.5 -> 2
        2.5 -> 3
    """
    return np.floor(x + 0.5).astype(int)


def calculate_grade_metrics(
    name,
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64
    )

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

    true_ref = (
        y_true >= 2
    ).astype(int)

    pred_ref = (
        y_pred >= 2
    ).astype(int)

    sensitivity = recall_score(
        true_ref,
        pred_ref,
        zero_division=0
    )

    specificity = recall_score(
        1 - true_ref,
        1 - pred_ref,
        zero_division=0
    )

    precision = precision_score(
        true_ref,
        pred_ref,
        zero_division=0
    )

    ref_f1 = f1_score(
        true_ref,
        pred_ref,
        zero_division=0
    )

    # --------------------------------------------------------
    # Error distance
    # --------------------------------------------------------

    absolute_error = np.abs(
        y_true - y_pred
    )

    exact = int(
        np.sum(absolute_error == 0)
    )

    off_by_1 = int(
        np.sum(absolute_error == 1)
    )

    off_by_2_plus = int(
        np.sum(absolute_error >= 2)
    )

    return {
        "model": name,
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
        "qwk": float(qwk),
        "mae": float(mae),

        "referable_sensitivity": float(
            sensitivity
        ),
        "referable_specificity": float(
            specificity
        ),
        "referable_precision": float(
            precision
        ),
        "referable_f1": float(
            ref_f1
        ),

        "exact_correct": exact,
        "exact_correct_rate": float(
            exact / len(y_true)
        ),

        "off_by_1": off_by_1,
        "off_by_1_rate": float(
            off_by_1 / len(y_true)
        ),

        "off_by_2_plus": off_by_2_plus,
        "off_by_2_plus_rate": float(
            off_by_2_plus / len(y_true)
        ),
    }


def print_metrics(row):
    print(
        f"{row['model']:<35}"
        f"Acc={row['accuracy']:.4f}  "
        f"MacroF1={row['macro_f1']:.4f}  "
        f"QWK={row['qwk']:.4f}  "
        f"RefSens={row['referable_sensitivity']:.4f}  "
        f"RefSpec={row['referable_specificity']:.4f}  "
        f"MAE={row['mae']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("DRISHTIAI + SENANUR ENSEMBLE VALIDATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DR_PATH.exists():
        raise FileNotFoundError(
            f"DrishtiAI predictions not found:\n{DR_PATH}"
        )

    if not SENANUR_PATH.exists():
        raise FileNotFoundError(
            f"Senanur predictions not found:\n{SENANUR_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dr = pd.read_csv(
        DR_PATH
    )

    sen = pd.read_csv(
        SENANUR_PATH
    )

    print()
    print("DrishtiAI rows :", len(dr))
    print("Senanur rows   :", len(sen))

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    dr_required = {
        "id_code",
        "true_grade",
        "predicted_grade"
    }

    sen_required = {
        "id_code",
        "true_grade",
        "predicted_grade",
        "raw_score"
    }

    if not dr_required.issubset(
        dr.columns
    ):
        raise ValueError(
            "DrishtiAI prediction CSV is missing "
            f"required columns: "
            f"{dr_required - set(dr.columns)}"
        )

    if not sen_required.issubset(
        sen.columns
    ):
        raise ValueError(
            "Senanur prediction CSV is missing "
            f"required columns: "
            f"{sen_required - set(sen.columns)}"
        )

    # --------------------------------------------------------
    # Merge by image ID
    # --------------------------------------------------------

    merged = pd.merge(
        dr[
            [
                "id_code",
                "true_grade",
                "predicted_grade",
                "confidence"
            ]
        ],
        sen[
            [
                "id_code",
                "true_grade",
                "predicted_grade",
                "raw_score"
            ]
        ],
        on="id_code",
        how="inner",
        suffixes=(
            "_dr",
            "_sen"
        )
    )

    print(
        "Matched images:",
        len(merged)
    )

    if len(merged) != 354:
        print()
        print(
            "WARNING: Expected 354 matched images."
        )

    # --------------------------------------------------------
    # Check labels agree
    # --------------------------------------------------------

    label_match = (
        merged["true_grade_dr"]
        ==
        merged["true_grade_sen"]
    )

    if not label_match.all():

        bad = merged[
            ~label_match
        ]

        raise RuntimeError(
            "True labels do not match between "
            f"prediction files.\n"
            f"Mismatched rows: {len(bad)}"
        )

    y_true = (
        merged["true_grade_dr"]
        .astype(int)
        .to_numpy()
    )

    dr_pred = (
        merged["predicted_grade_dr"]
        .astype(int)
        .to_numpy()
    )

    sen_pred = (
        merged["predicted_grade_sen"]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreement = np.mean(
        dr_pred == sen_pred
    )

    print(
        f"Model grade agreement: "
        f"{agreement:.4f} "
        f"({agreement * 100:.2f}%)"
    )

    # ========================================================
    # BUILD ENSEMBLE PREDICTIONS
    # ========================================================

    predictions = {}

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    predictions[
        "DrishtiAI"
    ] = dr_pred

    predictions[
        "Senanur"
    ] = sen_pred

    # --------------------------------------------------------
    # 50/50 average
    # --------------------------------------------------------

    avg_50 = (
        0.50 * dr_pred
        +
        0.50 * sen_pred
    )

    predictions[
        "50% DrishtiAI + 50% Senanur"
    ] = round_half_up(
        avg_50
    )

    # --------------------------------------------------------
    # 75/25
    # --------------------------------------------------------

    avg_75_dr = (
        0.75 * dr_pred
        +
        0.25 * sen_pred
    )

    predictions[
        "75% DrishtiAI + 25% Senanur"
    ] = round_half_up(
        avg_75_dr
    )

    # --------------------------------------------------------
    # 25/75
    # --------------------------------------------------------

    avg_75_sen = (
        0.25 * dr_pred
        +
        0.75 * sen_pred
    )

    predictions[
        "25% DrishtiAI + 75% Senanur"
    ] = round_half_up(
        avg_75_sen
    )

    # --------------------------------------------------------
    # Conservative maximum
    # --------------------------------------------------------

    predictions[
        "MAX grade (conservative)"
    ] = np.maximum(
        dr_pred,
        sen_pred
    )

    # --------------------------------------------------------
    # Conservative minimum
    # --------------------------------------------------------

    predictions[
        "MIN grade"
    ] = np.minimum(
        dr_pred,
        sen_pred
    )

    # ========================================================
    # METRICS
    # ========================================================

    metric_rows = []

    for name, pred in predictions.items():

        row = calculate_grade_metrics(
            name,
            y_true,
            pred
        )

        metric_rows.append(
            row
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    # --------------------------------------------------------
    # Screening rules
    # --------------------------------------------------------

    true_referable = (
        y_true >= 2
    ).astype(int)

    # DrishtiAI screening
    dr_ref = (
        dr_pred >= 2
    ).astype(int)

    # Senanur screening
    sen_ref = (
        sen_pred >= 2
    ).astype(int)

    # Either model says referable
    or_ref = (
        (dr_pred >= 2)
        |
        (sen_pred >= 2)
    ).astype(int)

    # Both models must say referable
    and_ref = (
        (dr_pred >= 2)
        &
        (sen_pred >= 2)
    ).astype(int)

    screening_rules = {
        "DrishtiAI screening": dr_ref,
        "Senanur screening": sen_ref,
        "OR safety screening": or_ref,
        "AND consensus screening": and_ref,
    }

    screening_rows = []

    for name, pred_ref in screening_rules.items():

        sensitivity = recall_score(
            true_referable,
            pred_ref,
            zero_division=0
        )

        specificity = recall_score(
            1 - true_referable,
            1 - pred_ref,
            zero_division=0
        )

        precision = precision_score(
            true_referable,
            pred_ref,
            zero_division=0
        )

        f1 = f1_score(
            true_referable,
            pred_ref,
            zero_division=0
        )

        cm = confusion_matrix(
            true_referable,
            pred_ref
        )

        tn, fp, fn, tp = cm.ravel()

        screening_rows.append(
            {
                "screening_rule": name,
                "sensitivity": float(
                    sensitivity
                ),
                "specificity": float(
                    specificity
                ),
                "precision": float(
                    precision
                ),
                "f1": float(f1),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    screening_df = pd.DataFrame(
        screening_rows
    )

    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print()
    print("=" * 90)
    print("GRADE MODEL COMPARISON")
    print("=" * 90)

    for _, row in metrics_df.iterrows():

        print_metrics(row)

    print()
    print("=" * 90)
    print("REFERABLE SCREENING COMPARISON")
    print("=" * 90)

    for _, row in screening_df.iterrows():

        print(
            f"{row['screening_rule']:<30}"
            f"Sensitivity={row['sensitivity']:.4f}  "
            f"Specificity={row['specificity']:.4f}  "
            f"Precision={row['precision']:.4f}  "
            f"F1={row['f1']:.4f}  "
            f"FN={int(row['false_negative'])}"
        )

    # ========================================================
    # FIND BEST CANDIDATES
    # ========================================================

    best_qwk = metrics_df.loc[
        metrics_df["qwk"].idxmax()
    ]

    best_macro_f1 = metrics_df.loc[
        metrics_df["macro_f1"].idxmax()
    ]

    best_accuracy = metrics_df.loc[
        metrics_df["accuracy"].idxmax()
    ]

    best_ref_sensitivity = screening_df.loc[
        screening_df["sensitivity"].idxmax()
    ]

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_output = merged[
        [
            "id_code",
            "true_grade_dr",
            "predicted_grade_dr",
            "predicted_grade_sen",
            "confidence",
            "raw_score"
        ]
    ].copy()

    prediction_output = prediction_output.rename(
        columns={
            "true_grade_dr": "true_grade",
            "predicted_grade_dr": "drishti_grade",
            "predicted_grade_sen": "senanur_grade",
            "raw_score": "senanur_raw_score"
        }
    )

    prediction_output[
        "ensemble_50_50_grade"
    ] = predictions[
        "50% DrishtiAI + 50% Senanur"
    ]

    prediction_output[
        "ensemble_75_dr_grade"
    ] = predictions[
        "75% DrishtiAI + 25% Senanur"
    ]

    prediction_output[
        "ensemble_75_sen_grade"
    ] = predictions[
        "25% DrishtiAI + 75% Senanur"
    ]

    prediction_output[
        "ensemble_max_grade"
    ] = predictions[
        "MAX grade (conservative)"
    ]

    prediction_output[
        "drishti_referable"
    ] = (
        dr_pred >= 2
    ).astype(int)

    prediction_output[
        "senanur_referable"
    ] = (
        sen_pred >= 2
    ).astype(int)

    prediction_output[
        "or_referable"
    ] = or_ref

    prediction_output[
        "and_referable"
    ] = and_ref

    prediction_output[
        "models_agree"
    ] = (
        dr_pred == sen_pred
    ).astype(int)

    prediction_output.to_csv(
        OUTPUT_DIR
        / "ensemble_val_predictions.csv",
        index=False
    )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    for name, pred in predictions.items():

        safe_name = (
            name
            .lower()
            .replace("%", "pct")
            .replace("+", "plus")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        cm = confusion_matrix(
            y_true,
            pred,
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

        cm_df.to_csv(
            OUTPUT_DIR
            / f"{safe_name}_confusion_matrix.csv"
        )

    # ========================================================
    # DETAILED REPORT FOR EACH ENSEMBLE
    # ========================================================

    detailed_reports = {}

    for name, pred in predictions.items():

        report = classification_report(
            y_true,
            pred,
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

        detailed_reports[name] = report

    # ========================================================
    # FINAL JSON
    # ========================================================

    final_report = {
        "dataset": "APTOS clean validation",
        "images": int(len(y_true)),
        "model_agreement": float(agreement),

        "grade_metrics": (
            metrics_df.to_dict(
                orient="records"
            )
        ),

        "screening_metrics": (
            screening_df.to_dict(
                orient="records"
            )
        ),

        "best_qwk": (
            best_qwk.to_dict()
        ),

        "best_macro_f1": (
            best_macro_f1.to_dict()
        ),

        "best_accuracy": (
            best_accuracy.to_dict()
        ),

        "best_referable_sensitivity": (
            best_ref_sensitivity.to_dict()
        ),

        "per_grade": detailed_reports
    }

    report_path = (
        OUTPUT_DIR
        / "ensemble_val_evaluation_report.json"
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
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 90)
    print("BEST RESULTS")
    print("=" * 90)

    print()
    print("Best QWK:")
    print(
        f"{best_qwk['model']} "
        f"-> {best_qwk['qwk']:.4f}"
    )

    print()
    print("Best Macro F1:")
    print(
        f"{best_macro_f1['model']} "
        f"-> {best_macro_f1['macro_f1']:.4f}"
    )

    print()
    print("Best Accuracy:")
    print(
        f"{best_accuracy['model']} "
        f"-> {best_accuracy['accuracy']:.4f}"
    )

    print()
    print("Best Referable Sensitivity:")
    print(
        f"{best_ref_sensitivity['screening_rule']} "
        f"-> "
        f"{best_ref_sensitivity['sensitivity']:.4f}"
    )

    print()
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(
        OUTPUT_DIR
        / "ensemble_val_predictions.csv"
    )

    print(
        OUTPUT_DIR
        / "ensemble_val_evaluation_report.json"
    )

    print()
    print("=" * 90)
    print("ENSEMBLE VALIDATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()