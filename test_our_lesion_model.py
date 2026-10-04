from pathlib import Path
import json
import time

import cv2
import numpy as np
import torch

from backend.models.lesion_detector import LesionDetector


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
    "our_model_evaluation"
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
# EXACT SAME TEST SET USED FOR CLEMENTP
# ============================================================

TEST_IDS = [
    "IDRiD_08",
    "IDRiD_23",
    "IDRiD_25",
    "IDRiD_39",
]


# ============================================================
# OUR MODEL CLASS IDs
# ============================================================

# 0 = background
# 1 = microaneurysm
# 2 = hemorrhage
# 3 = hard exudate
# 4 = soft exudate

MODEL_CLASSES = {
    "MA": 1,
    "HE": 2,
    "EX": 3,
    "SE": 4,
}


# ============================================================
# MODEL INPUT SIZE
# ============================================================

IMAGE_SIZE = 512


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(image_id, lesion_name):
    mask_path = (
        MASK_DIRS[lesion_name]
        / f"{image_id}_{lesion_name}.tif"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing mask: {mask_path}"
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
# CALCULATE METRICS
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

    pred_pixels = prediction.sum()
    gt_pixels = ground_truth.sum()

    # Dice
    denominator = (
        pred_pixels + gt_pixels
    )

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * tp / denominator
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_pixels": int(pred_pixels),
        "ground_truth_pixels": int(gt_pixels),
    }


# ============================================================
# CREATE COLOR OVERLAY
# ============================================================

def create_overlay(
    original_bgr,
    predicted_mask,
):
    color_mask = np.zeros_like(
        original_bgr
    )

    # BGR colors
    colors = {
        1: (255, 0, 255),   # MA
        2: (0, 0, 255),     # HE
        3: (0, 255, 255),   # EX
        4: (255, 255, 0),   # SE
    }

    for class_id, color in colors.items():
        color_mask[
            predicted_mask == class_id
        ] = color

    lesion_area = (
        predicted_mask != 0
    )

    overlay = original_bgr.copy()

    if np.any(lesion_area):
        overlay[lesion_area] = cv2.addWeighted(
            original_bgr[lesion_area],
            0.65,
            color_mask[lesion_area],
            0.35,
            0,
        )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OUR LESION MODEL - IDRiD TEST EVALUATION")
    print("=" * 70)

    print("\nTest images:")

    for image_id in TEST_IDS:
        print(f"  {image_id}")

    # ========================================================
    # LOAD CURRENT MODEL
    # ========================================================

    print("\nLoading current DrishtiAI lesion model...")

    detector = LesionDetector()

    model = getattr(
        detector,
        "model",
        None,
    )

    if model is None:
        raise RuntimeError(
            "Could not find 'model' inside "
            "LesionDetector. Please check "
            "backend/models/lesion_detector.py"
        )

    model.eval()

    # Try to determine device
    device = getattr(
        detector,
        "device",
        None,
    )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)

    model.to(device)

    print(
        f"Model device: {device}"
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    all_results = {
        "MA": [],
        "HE": [],
        "EX": [],
        "SE": [],
    }

    inference_times = []

    # ========================================================
    # EVALUATE EACH IMAGE
    # ========================================================

    for index, image_id in enumerate(
        TEST_IDS,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(TEST_IDS)}] "
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
            raise RuntimeError(
                f"Could not read image: "
                f"{image_path}"
            )

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ====================================================
        # PREPROCESS
        # ====================================================

        image_512 = cv2.resize(
            original_rgb,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        image_tensor = (
            torch.from_numpy(
                image_512
            )
            .float()
            / 255.0
        )

        image_tensor = (
            image_tensor
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        # ====================================================
        # INFERENCE
        # ====================================================

        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(
                image_tensor
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        inference_times.append(
            elapsed
        )

        # ====================================================
        # HANDLE MODEL OUTPUT
        # ====================================================

        if isinstance(output, tuple):
            output = output[0]

        elif isinstance(output, dict):

            if "out" in output:
                output = output["out"]

            elif "logits" in output:
                output = output["logits"]

            else:
                raise RuntimeError(
                    "Unknown dictionary "
                    "output keys: "
                    f"{output.keys()}"
                )

        if not torch.is_tensor(output):
            raise RuntimeError(
                "Model output is not a tensor."
            )

        print(
            f"Output shape: "
            f"{tuple(output.shape)}"
        )

        # ====================================================
        # ARGMAX
        # ====================================================

        predicted_mask = (
            torch.argmax(
                output,
                dim=1,
            )[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        print(
            f"Inference time: "
            f"{elapsed:.2f}s"
        )

        # ====================================================
        # EVALUATE EACH LESION
        # ====================================================

        for lesion_name, class_id in (
            MODEL_CLASSES.items()
        ):

            gt_mask = load_mask(
                image_id,
                lesion_name,
            )

            # Match model resolution
            gt_mask_512 = cv2.resize(
                gt_mask.astype(np.uint8),
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            prediction_mask = (
                predicted_mask == class_id
            )

            metrics = calculate_metrics(
                prediction_mask,
                gt_mask_512,
            )

            all_results[
                lesion_name
            ].append(metrics)

            print(
                f"  {lesion_name}: "
                f"Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f}"
            )

        # ====================================================
        # SAVE OVERLAY
        # ====================================================

        overlay_512 = create_overlay(
            image_512[:, :, ::-1],
            predicted_mask,
        )

        overlay_path = (
            OUTPUT_DIR
            / f"{image_id}_overlay.png"
        )

        cv2.imwrite(
            str(overlay_path),
            overlay_512,
        )

        print(
            f"Overlay saved: "
            f"{overlay_path}"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL OUR MODEL RESULTS")
    print("=" * 70)

    summary = {}

    for lesion_name in [
        "MA",
        "HE",
        "EX",
        "SE",
    ]:

        values = (
            all_results[
                lesion_name
            ]
        )

        dice = np.mean([
            x["dice"]
            for x in values
        ])

        iou = np.mean([
            x["iou"]
            for x in values
        ])

        precision = np.mean([
            x["precision"]
            for x in values
        ])

        recall = np.mean([
            x["recall"]
            for x in values
        ])

        summary[
            lesion_name
        ] = {
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

    # ========================================================
    # MEAN ALL FOUR LESIONS
    # ========================================================

    mean_dice_all = np.mean([
        summary["MA"]["dice"],
        summary["HE"]["dice"],
        summary["EX"]["dice"],
        summary["SE"]["dice"],
    ])

    mean_iou_all = np.mean([
        summary["MA"]["iou"],
        summary["HE"]["iou"],
        summary["EX"]["iou"],
        summary["SE"]["iou"],
    ])

    # ClementP does not have SE.
    # Therefore this comparison should use
    # MA + HE + EX only.

    mean_shared_dice = np.mean([
        summary["MA"]["dice"],
        summary["HE"]["dice"],
        summary["EX"]["dice"],
    ])

    mean_shared_iou = np.mean([
        summary["MA"]["iou"],
        summary["HE"]["iou"],
        summary["EX"]["iou"],
    ])

    average_inference_time = (
        np.mean(inference_times)
    )

    print(
        "\nMean all 4 lesion Dice: "
        f"{mean_dice_all:.4f}"
    )

    print(
        "Mean all 4 lesion IoU:  "
        f"{mean_iou_all:.4f}"
    )

    print(
        "\nMean shared-lesion Dice "
        "(MA + HE + EX): "
        f"{mean_shared_dice:.4f}"
    )

    print(
        "Mean shared-lesion IoU "
        "(MA + HE + EX): "
        f"{mean_shared_iou:.4f}"
    )

    print(
        "\nAverage CPU inference time: "
        f"{average_inference_time:.2f}s/image"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    report = {
        "model": "DrishtiAI current lesion_model.pth",
        "dataset": "IDRiD",
        "test_images": TEST_IDS,
        "metrics": summary,
        "mean_all_4_lesions_dice": float(
            mean_dice_all
        ),
        "mean_all_4_lesions_iou": float(
            mean_iou_all
        ),
        "mean_shared_lesions_dice": float(
            mean_shared_dice
        ),
        "mean_shared_lesions_iou": float(
            mean_shared_iou
        ),
        "average_cpu_inference_seconds": float(
            average_inference_time
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
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nReport saved to: "
        f"{report_path}"
    )

    print("\n" + "=" * 70)
    print(
        "OUR MODEL EVALUATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()