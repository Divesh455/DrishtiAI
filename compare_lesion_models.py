from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
import torch

from backend.models.lesion_detector import LesionDetector
from fundus_lesions_toolkit.models import segment
from fundus_lesions_toolkit.constants import Dataset


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path(
    "data/raw/idrid/images/training"
)

ANNOTATION_DIR = Path(
    "data/raw/idrid/annotations/training"
)

OUTPUT_DIR = Path(
    "lesion_model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IDRiD MASK DIRECTORIES
# ============================================================

MASK_DIRS = {
    "MA": ANNOTATION_DIR / "microaneurysms",
    "HE": ANNOTATION_DIR / "haemorrhages",
    "EX": ANNOTATION_DIR / "hard_exudates",
    "SE": ANNOTATION_DIR / "soft_exudates",
}


# ============================================================
# ORIGINAL HELD-OUT TEST SET
# ============================================================

ORIGINAL_TEST_IDS = {
    "IDRiD_08",
    "IDRiD_23",
    "IDRiD_25",
    "IDRiD_39",
}


# ============================================================
# CLEMENTP
# ============================================================

CLEMENTP_CLASSES = {
    "CWS": 1,
    "EX": 2,
    "HE": 3,
    "MA": 4,
}

CLEMENTP_SIZE = 1024


# ============================================================
# OUR MODEL
# ============================================================

OUR_CLASSES = {
    "MA": 1,
    "HE": 2,
    "EX": 3,
    "SE": 4,
}

OUR_SIZE = 512


# ============================================================
# FIND ALL 26 COMPLETE IMAGES
# ============================================================

def get_complete_ids():
    ids = []

    for image_path in sorted(
        IMAGE_DIR.glob("*.jpg")
    ):
        image_id = image_path.stem

        required_masks = [
            MASK_DIRS["MA"]
            / f"{image_id}_MA.tif",

            MASK_DIRS["HE"]
            / f"{image_id}_HE.tif",

            MASK_DIRS["EX"]
            / f"{image_id}_EX.tif",

            MASK_DIRS["SE"]
            / f"{image_id}_SE.tif",
        ]

        if all(
            p.exists()
            for p in required_masks
        ):
            ids.append(image_id)

    return sorted(ids)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(
    image_id,
    lesion_name,
):
    mask_path = (
        MASK_DIRS[lesion_name]
        / f"{image_id}_{lesion_name}.tif"
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    return mask > 0


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = int(
        np.logical_and(
            prediction,
            ground_truth,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            np.logical_not(ground_truth),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(prediction),
            ground_truth,
        ).sum()
    )

    pred_pixels = int(
        prediction.sum()
    )

    gt_pixels = int(
        ground_truth.sum()
    )

    # Dice
    dice_denominator = (
        pred_pixels + gt_pixels
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * tp
            / dice_denominator
        )

    # IoU
    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            tp / precision_denominator
        )

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            tp / recall_denominator
        )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_pixels": pred_pixels,
        "ground_truth_pixels": gt_pixels,
    }


# ============================================================
# CLEMENTP IMAGE PREPARATION
# ============================================================

def resize_with_padding(
    image,
    target_size,
):
    h, w = image.shape[:2]

    scale = min(
        target_size / w,
        target_size / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            target_size,
            target_size,
            3,
        ),
        dtype=np.uint8,
    )

    pad_x = (
        target_size - new_w
    ) // 2

    pad_y = (
        target_size - new_h
    ) // 2

    canvas[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = resized

    return (
        canvas,
        scale,
        pad_x,
        pad_y,
        new_w,
        new_h,
    )


def restore_clementp_mask(
    mask,
    original_shape,
    pad_x,
    pad_y,
    new_w,
    new_h,
):
    cropped = mask[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ]

    restored = cv2.resize(
        cropped.astype(np.uint8),
        (
            original_shape[1],
            original_shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return restored


# ============================================================
# OUR MODEL INFERENCE
# ============================================================

def predict_our_model(
    model,
    device,
    image_rgb,
):
    image_512 = cv2.resize(
        image_rgb,
        (OUR_SIZE, OUR_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    tensor = (
        torch.from_numpy(
            image_512
        )
        .float()
        / 255.0
    )

    tensor = (
        tensor
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        output = model(
            tensor
        )

    if isinstance(output, tuple):
        output = output[0]

    elif isinstance(output, dict):

        if "out" in output:
            output = output["out"]

        elif "logits" in output:
            output = output["logits"]

        else:
            raise RuntimeError(
                f"Unknown model output keys: "
                f"{output.keys()}"
            )

    if not torch.is_tensor(output):
        raise RuntimeError(
            "Model output is not a tensor."
        )

    predicted_mask_512 = (
        torch.argmax(
            output,
            dim=1,
        )[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return predicted_mask_512


# ============================================================
# CLEMENTP INFERENCE
# ============================================================

def predict_clementp(
    image_rgb,
):

    (
        prepared,
        scale,
        pad_x,
        pad_y,
        new_w,
        new_h,
    ) = resize_with_padding(
        image_rgb,
        CLEMENTP_SIZE,
    )

    prediction = segment(
        prepared,
        arch="unet",
        encoder="seresnext50_32x4d",
        train_datasets=Dataset.ALL,
        image_resolution=CLEMENTP_SIZE,
        autofit_resolution=False,
        reverse_autofit=False,
        device="cpu",
    )

    predicted_mask_1024 = (
        prediction
        .argmax(dim=0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return restore_clementp_mask(
        predicted_mask_1024,
        image_rgb.shape,
        pad_x,
        pad_y,
        new_w,
        new_h,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    path,
):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MEAN METRICS
# ============================================================

def average_metrics(
    records,
):
    if not records:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    return {
        "dice": float(
            np.mean([
                x["dice"]
                for x in records
            ])
        ),
        "iou": float(
            np.mean([
                x["iou"]
                for x in records
            ])
        ),
        "precision": float(
            np.mean([
                x["precision"]
                for x in records
            ])
        ),
        "recall": float(
            np.mean([
                x["recall"]
                for x in records
            ])
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "DRISHTIAI LESION MODEL COMPARISON"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Find complete images
    # --------------------------------------------------------

    image_ids = get_complete_ids()

    print(
        f"\nComplete annotated images: "
        f"{len(image_ids)}"
    )

    if len(image_ids) != 26:
        raise RuntimeError(
            f"Expected 26 complete images, "
            f"found {len(image_ids)}."
        )

    print("\nImages:")
    for image_id in image_ids:
        print(f"  {image_id}")

    # --------------------------------------------------------
    # Load OUR model
    # --------------------------------------------------------

    print(
        "\nLoading current DrishtiAI U-Net..."
    )

    detector = LesionDetector()

    our_model = detector.model

    our_model.eval()

    # Force CPU for consistent timing
    our_device = torch.device("cpu")

    our_model.to(
        our_device
    )

    print(
        "Our model device:",
        our_device,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    our_results = {
        "MA": [],
        "HE": [],
        "EX": [],
        "SE": [],
    }

    clementp_results = {
        "MA": [],
        "HE": [],
        "EX": [],
    }

    per_image_rows = []

    our_times = []
    clementp_times = []

    # ========================================================
    # LOOP THROUGH ALL 26
    # ========================================================

    for index, image_id in enumerate(
        image_ids,
        start=1,
    ):

        print("\n" + "-" * 80)
        print(
            f"[{index}/{len(image_ids)}] "
            f"{image_id}"
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.jpg"
        )

        image_bgr = cv2.imread(
            str(image_path)
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Could not read: "
                f"{image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        original_shape = (
            image_rgb.shape
        )

        # ====================================================
        # OUR MODEL
        # ====================================================

        start = time.perf_counter()

        our_mask_512 = predict_our_model(
            our_model,
            our_device,
            image_rgb,
        )

        our_time = (
            time.perf_counter()
            - start
        )

        our_times.append(
            our_time
        )

        our_mask = cv2.resize(
            our_mask_512,
            (
                original_shape[1],
                original_shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        print(
            f"Our model: "
            f"{our_time:.2f}s"
        )

        # ====================================================
        # CLEMENTP
        # ====================================================

        start = time.perf_counter()

        clementp_mask = (
            predict_clementp(
                image_rgb
            )
        )

        clementp_time = (
            time.perf_counter()
            - start
        )

        clementp_times.append(
            clementp_time
        )

        print(
            f"ClementP: "
            f"{clementp_time:.2f}s"
        )

        # ====================================================
        # EVALUATE OUR MODEL
        # ====================================================

        image_row = {
            "image_id": image_id,
            "our_inference_seconds":
                round(our_time, 4),
            "clementp_inference_seconds":
                round(clementp_time, 4),
        }

        for lesion_name, class_id in (
            OUR_CLASSES.items()
        ):

            gt = load_ground_truth(
                image_id,
                lesion_name,
            )

            prediction = (
                our_mask == class_id
            )

            metrics = calculate_metrics(
                prediction,
                gt,
            )

            our_results[
                lesion_name
            ].append(metrics)

            image_row[
                f"our_{lesion_name}_dice"
            ] = round(
                metrics["dice"],
                6,
            )

            print(
                f"  OUR {lesion_name}: "
                f"Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f}"
            )

        # ====================================================
        # EVALUATE CLEMENTP
        # ====================================================

        for lesion_name, class_id in (
            CLEMENTP_CLASSES.items()
        ):

            if lesion_name == "CWS":
                continue

            gt = load_ground_truth(
                image_id,
                lesion_name,
            )

            prediction = (
                clementp_mask == class_id
            )

            metrics = calculate_metrics(
                prediction,
                gt,
            )

            clementp_results[
                lesion_name
            ].append(metrics)

            image_row[
                f"clementp_{lesion_name}_dice"
            ] = round(
                metrics["dice"],
                6,
            )

            print(
                f"  CLEMENTP {lesion_name}: "
                f"Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f}"
            )

        per_image_rows.append(
            image_row
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 80)
    print(
        "FULL 26-IMAGE RESULTS"
    )
    print("=" * 80)

    our_summary = {}
    clementp_summary = {}

    print("\nCURRENT DRISHTIAI U-NET")

    for lesion_name in [
        "MA",
        "HE",
        "EX",
        "SE",
    ]:

        metrics = average_metrics(
            our_results[
                lesion_name
            ]
        )

        our_summary[
            lesion_name
        ] = metrics

        print(
            f"\n{lesion_name}"
        )

        print(
            f"  Dice      : "
            f"{metrics['dice']:.4f}"
        )

        print(
            f"  IoU       : "
            f"{metrics['iou']:.4f}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.4f}"
        )

    print(
        "\nCLEMENTP"
    )

    for lesion_name in [
        "MA",
        "HE",
        "EX",
    ]:

        metrics = average_metrics(
            clementp_results[
                lesion_name
            ]
        )

        clementp_summary[
            lesion_name
        ] = metrics

        print(
            f"\n{lesion_name}"
        )

        print(
            f"  Dice      : "
            f"{metrics['dice']:.4f}"
        )

        print(
            f"  IoU       : "
            f"{metrics['iou']:.4f}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.4f}"
        )

    # ========================================================
    # SHARED LESION MEANS
    # ========================================================

    our_shared_dice = float(
        np.mean([
            our_summary["MA"]["dice"],
            our_summary["HE"]["dice"],
            our_summary["EX"]["dice"],
        ])
    )

    our_shared_iou = float(
        np.mean([
            our_summary["MA"]["iou"],
            our_summary["HE"]["iou"],
            our_summary["EX"]["iou"],
        ])
    )

    clementp_shared_dice = float(
        np.mean([
            clementp_summary["MA"]["dice"],
            clementp_summary["HE"]["dice"],
            clementp_summary["EX"]["dice"],
        ])
    )

    clementp_shared_iou = float(
        np.mean([
            clementp_summary["MA"]["iou"],
            clementp_summary["HE"]["iou"],
            clementp_summary["EX"]["iou"],
        ])
    )

    print("\n" + "=" * 80)
    print(
        "SHARED-CLASS COMPARISON"
    )
    print("=" * 80)

    print(
        "\nMA:"
        f"\n  Our      = "
        f"{our_summary['MA']['dice']:.4f}"
        f"\n  ClementP = "
        f"{clementp_summary['MA']['dice']:.4f}"
    )

    print(
        "\nHE:"
        f"\n  Our      = "
        f"{our_summary['HE']['dice']:.4f}"
        f"\n  ClementP = "
        f"{clementp_summary['HE']['dice']:.4f}"
    )

    print(
        "\nEX:"
        f"\n  Our      = "
        f"{our_summary['EX']['dice']:.4f}"
        f"\n  ClementP = "
        f"{clementp_summary['EX']['dice']:.4f}"
    )

    print(
        "\nMean shared Dice:"
        f"\n  Our      = "
        f"{our_shared_dice:.4f}"
        f"\n  ClementP = "
        f"{clementp_shared_dice:.4f}"
    )

    print(
        "\nMean shared IoU:"
        f"\n  Our      = "
        f"{our_shared_iou:.4f}"
        f"\n  ClementP = "
        f"{clementp_shared_iou:.4f}"
    )

    # ========================================================
    # SPEED
    # ========================================================

    our_avg_time = float(
        np.mean(our_times)
    )

    clementp_avg_time = float(
        np.mean(clementp_times)
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "AVERAGE CPU INFERENCE TIME"
    )

    print(
        f"\nOur U-Net : "
        f"{our_avg_time:.2f}s/image"
    )

    print(
        f"ClementP  : "
        f"{clementp_avg_time:.2f}s/image"
    )

    # ========================================================
    # ORIGINAL 4-IMAGE TEST SUBSET
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ORIGINAL HELD-OUT 4-IMAGE SUBSET"
    )

    print(
        "These are the images previously used "
        "as the U-Net test split."
    )

    # Get rows corresponding to original test
    test_rows = [
        row
        for row in per_image_rows
        if row["image_id"]
        in ORIGINAL_TEST_IDS
    ]

    def mean_column(
        rows,
        column,
    ):
        values = [
            row[column]
            for row in rows
            if column in row
        ]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    for lesion_name in [
        "MA",
        "HE",
        "EX",
    ]:

        our_value = mean_column(
            test_rows,
            f"our_{lesion_name}_dice",
        )

        cp_value = mean_column(
            test_rows,
            f"clementp_{lesion_name}_dice",
        )

        print(
            f"\n{lesion_name}:"
            f"\n  Our      = {our_value:.4f}"
            f"\n  ClementP = {cp_value:.4f}"
        )

    our_test_mean = float(
        np.mean([
            mean_column(
                test_rows,
                "our_MA_dice",
            ),
            mean_column(
                test_rows,
                "our_HE_dice",
            ),
            mean_column(
                test_rows,
                "our_EX_dice",
            ),
        ])
    )

    cp_test_mean = float(
        np.mean([
            mean_column(
                test_rows,
                "clementp_MA_dice",
            ),
            mean_column(
                test_rows,
                "clementp_HE_dice",
            ),
            mean_column(
                test_rows,
                "clementp_EX_dice",
            ),
        ])
    )

    print(
        f"\nMean shared Dice:"
        f"\n  Our      = {our_test_mean:.4f}"
        f"\n  ClementP = {cp_test_mean:.4f}"
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "per_image_comparison.csv"
    )

    save_csv(
        per_image_rows,
        csv_path,
    )

    report = {
        "dataset": "IDRiD",
        "complete_images": image_ids,
        "image_count": len(image_ids),
        "original_held_out_test_ids": sorted(
            ORIGINAL_TEST_IDS
        ),
        "our_model": {
            "name": "DrishtiAI current U-Net",
            "metrics": our_summary,
            "mean_shared_dice": our_shared_dice,
            "mean_shared_iou": our_shared_iou,
            "average_cpu_seconds": our_avg_time,
        },
        "clementp_model": {
            "name": (
                "ClementP/"
                "fundus-lesions-segmentation-"
                "unet_seresnext50_32x4d"
            ),
            "metrics": clementp_summary,
            "mean_shared_dice": clementp_shared_dice,
            "mean_shared_iou": clementp_shared_iou,
            "average_cpu_seconds": clementp_avg_time,
        },
        "original_held_out_test_subset": {
            "our_mean_shared_dice": our_test_mean,
            "clementp_mean_shared_dice": cp_test_mean,
        },
        "class_note": (
            "ClementP CWS was not compared "
            "against our Soft Exudate class."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "comparison_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPARISON COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nPer-image CSV:"
        f"\n{csv_path}"
    )

    print(
        f"\nJSON report:"
        f"\n{json_path}"
    )


if __name__ == "__main__":
    main()