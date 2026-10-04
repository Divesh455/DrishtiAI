from pathlib import Path
import json
import time

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

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
    "clementp_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# CLEMENTP CLASS IDS
# ============================================================

# ClementP:
#
# 0 = background
# 1 = cotton wool spot
# 2 = exudates
# 3 = hemorrhages
# 4 = microaneurysms

CLEMENTP_CLASSES = {
    "CWS": 1,
    "EX": 2,
    "HE": 3,
    "MA": 4,
}


# ============================================================
# OUR IDRID GROUND TRUTH CLASS NAMES
# ============================================================

GT_CLASSES = {
    "MA": "MA",
    "HE": "HE",
    "EX": "EX",
    "SE": "SE",
}


# ============================================================
# MODEL INPUT
# ============================================================

TARGET_SIZE = 1024


# ============================================================
# FIND COMPLETE IDRID IMAGES
# ============================================================

def get_complete_ids():

    image_ids = []

    for image_path in sorted(
        IMAGE_DIR.glob("*.jpg")
    ):

        image_id = image_path.stem

        required = [
            MASK_DIRS["MA"] / f"{image_id}_MA.tif",
            MASK_DIRS["HE"] / f"{image_id}_HE.tif",
            MASK_DIRS["EX"] / f"{image_id}_EX.tif",
            MASK_DIRS["SE"] / f"{image_id}_SE.tif",
        ]

        if all(
            path.exists()
            for path in required
        ):
            image_ids.append(image_id)

    return image_ids


# ============================================================
# SAME 70/15/15 SPLIT
# ============================================================

def get_test_ids():

    ids = get_complete_ids()

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=0.30,
        random_state=42,
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42,
    )

    return sorted(test_ids)


# ============================================================
# RESIZE WITH PADDING
# ============================================================

def resize_with_padding(
    image,
    target_size=1024,
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


# ============================================================
# RESTORE PREDICTION TO ORIGINAL SIZE
# ============================================================

def restore_prediction(
    prediction,
    original_shape,
    scale,
    pad_x,
    pad_y,
    new_w,
    new_h,
):

    cropped = prediction[
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
# LOAD GROUND TRUTH
# ============================================================

def load_mask(
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
# BINARY METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(ground_truth),
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        ground_truth,
    ).sum()

    intersection = tp

    prediction_pixels = prediction.sum()
    ground_truth_pixels = ground_truth.sum()

    union = (
        prediction_pixels
        + ground_truth_pixels
        - intersection
    )

    dice_denominator = (
        prediction_pixels
        + ground_truth_pixels
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection
            / dice_denominator
        )

    if union == 0:
        iou = 1.0
    else:
        iou = (
            intersection
            / union
        )

    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = tp / recall_denominator

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_pixels": int(prediction_pixels),
        "ground_truth_pixels": int(ground_truth_pixels),
    }


# ============================================================
# COLOR VISUALIZATION
# ============================================================

def create_overlay(
    original_bgr,
    predicted_mask,
):

    overlay = original_bgr.copy()

    colors = {
        1: (255, 255, 0),   # CWS
        2: (0, 255, 255),   # EX
        3: (0, 0, 255),     # HE
        4: (255, 0, 255),   # MA
    }

    mask_color = np.zeros_like(
        original_bgr
    )

    for class_id, color in colors.items():

        mask_color[
            predicted_mask == class_id
        ] = color

    lesion_pixels = (
        predicted_mask != 0
    )

    overlay[lesion_pixels] = cv2.addWeighted(
        original_bgr[lesion_pixels],
        0.65,
        mask_color[lesion_pixels],
        0.35,
        0,
    )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ClementP Lesion Model - IDRiD Evaluation"
    )
    print("=" * 70)

    complete_ids = get_complete_ids()

    print(
        f"\nComplete IDRiD images: "
        f"{len(complete_ids)}"
    )

    test_ids = get_test_ids()

    print(
        f"Test images: {len(test_ids)}"
    )

    print("\nTest IDs:")

    for image_id in test_ids:
        print(f"  {image_id}")

    print("\n" + "-" * 70)

    # --------------------------------------------------------
    # Accumulate results
    # --------------------------------------------------------

    results = {
        "MA": [],
        "HE": [],
        "EX": [],
    }

    total_time = 0.0

    # --------------------------------------------------------
    # Evaluate each image
    # --------------------------------------------------------

    for index, image_id in enumerate(
        test_ids,
        start=1,
    ):

        print(
            f"\n[{index}/{len(test_ids)}] "
            f"{image_id}"
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.jpg"
        )

        original_bgr = cv2.imread(
            str(image_path)
        )

        if original_bgr is None:
            print(
                "ERROR: Could not read image."
            )
            continue

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB,
        )

        original_shape = (
            original_rgb.shape
        )

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        (
            prepared_image,
            scale,
            pad_x,
            pad_y,
            new_w,
            new_h,
        ) = resize_with_padding(
            original_rgb,
            TARGET_SIZE,
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        start = time.perf_counter()

        prediction = segment(
            prepared_image,
            arch="unet",
            encoder="seresnext50_32x4d",
            train_datasets=Dataset.ALL,
            image_resolution=TARGET_SIZE,
            autofit_resolution=False,
            reverse_autofit=False,
            device="cpu",
        )

        inference_time = (
            time.perf_counter() - start
        )

        total_time += inference_time

        print(
            f"  Inference time: "
            f"{inference_time:.2f}s"
        )

        # ----------------------------------------------------
        # Prediction -> class mask
        # ----------------------------------------------------

        predicted_mask_1024 = (
            prediction
            .argmax(dim=0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        predicted_mask = restore_prediction(
            predicted_mask_1024,
            original_shape,
            scale,
            pad_x,
            pad_y,
            new_w,
            new_h,
        )

        # ----------------------------------------------------
        # Shared classes
        # ----------------------------------------------------

        # ClementP:
        # EX=2
        # HE=3
        # MA=4

        # IDRiD:
        # EX
        # HE
        # MA

        mapping = {
            "MA": 4,
            "HE": 3,
            "EX": 2,
        }

        for lesion_name, class_id in mapping.items():

            gt_mask = load_mask(
                image_id,
                lesion_name,
            )

            if gt_mask.shape != predicted_mask.shape[:2]:

                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8),
                    (
                        predicted_mask.shape[1],
                        predicted_mask.shape[0],
                    ),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            pred_mask = (
                predicted_mask == class_id
            )

            metrics = calculate_metrics(
                pred_mask,
                gt_mask,
            )

            results[
                lesion_name
            ].append(metrics)

            print(
                f"  {lesion_name}: "
                f"Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f}"
            )

        # ----------------------------------------------------
        # Save overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            original_bgr,
            predicted_mask,
        )

        overlay_path = (
            OUTPUT_DIR
            / f"{image_id}_overlay.png"
        )

        cv2.imwrite(
            str(overlay_path),
            overlay,
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL CLEMENTP RESULTS")
    print("=" * 70)

    summary = {}

    for lesion_name in [
        "MA",
        "HE",
        "EX",
    ]:

        lesion_results = (
            results[lesion_name]
        )

        if not lesion_results:
            continue

        dice = np.mean([
            x["dice"]
            for x in lesion_results
        ])

        iou = np.mean([
            x["iou"]
            for x in lesion_results
        ])

        precision = np.mean([
            x["precision"]
            for x in lesion_results
        ])

        recall = np.mean([
            x["recall"]
            for x in lesion_results
        ])

        summary[lesion_name] = {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        }

        print(
            f"\n{lesion_name}"
        )

        print(
            f"  Dice      : {dice:.4f}"
        )

        print(
            f"  IoU       : {iou:.4f}"
        )

        print(
            f"  Precision : {precision:.4f}"
        )

        print(
            f"  Recall    : {recall:.4f}"
        )

    # --------------------------------------------------------
    # Mean shared lesion score
    # --------------------------------------------------------

    mean_dice = np.mean([
        summary[x]["dice"]
        for x in ["MA", "HE", "EX"]
    ])

    mean_iou = np.mean([
        summary[x]["iou"]
        for x in ["MA", "HE", "EX"]
    ])

    print(
        "\nMean shared-lesion Dice: "
        f"{mean_dice:.4f}"
    )

    print(
        "Mean shared-lesion IoU:  "
        f"{mean_iou:.4f}"
    )

    mean_inference = (
        total_time / len(test_ids)
        if test_ids
        else 0.0
    )

    print(
        "\nAverage CPU inference time: "
        f"{mean_inference:.2f}s/image"
    )

    # --------------------------------------------------------
    # CWS / SE note
    # --------------------------------------------------------

    print("\nClass compatibility:")
    print(
        "  ClementP CWS is NOT evaluated "
        "against IDRiD Soft Exudates."
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    report = {
        "model": (
            "ClementP/"
            "fundus-lesions-segmentation-"
            "unet_seresnext50_32x4d"
        ),
        "dataset": "IDRiD",
        "test_images": test_ids,
        "metrics": summary,
        "mean_shared_lesion_dice": float(
            mean_dice
        ),
        "mean_shared_lesion_iou": float(
            mean_iou
        ),
        "average_cpu_inference_seconds": float(
            mean_inference
        ),
        "class_note": (
            "CWS was not compared with "
            "Soft Exudates because they "
            "are different lesion classes."
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print(
        f"\nReport saved to: "
        f"{report_path}"
    )

    print("\n" + "=" * 70)
    print(
        "ClementP evaluation completed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()