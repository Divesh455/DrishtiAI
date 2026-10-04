from pathlib import Path
import json
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path(
    "external_models/senanur_dr"
)

IMAGE_PATH = Path(
    "data/raw/train_images/000c1434d8d7.png"
)

EXPORT_PATH = (
    MODEL_DIR / "export.json"
)

PREPROCESS_SIZE = 512
MODEL_INPUT_SIZE = 384


# ============================================================
# IMAGENET NORMALIZATION
# EXACTLY AS THE OFFICIAL ONNX SERVER
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# OFFICIAL THRESHOLDS
# ============================================================

THRESHOLDS = np.array(
    [
        0.668,
        1.132,
        2.324,
        3.300,
    ],
    dtype=np.float64,
)


GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ============================================================
# AUTO-CROP
# MATCHES SENANUR SOURCE
# ============================================================

def auto_crop(
    img,
    tol=7,
):
    """
    Remove the black frame surrounding the fundus.

    This matches the repository's preprocessing.py.
    """

    if img.ndim == 2:

        mask = img > tol

        if not mask.any():
            return img

        return img[
            np.ix_(
                mask.any(axis=1),
                mask.any(axis=0),
            )
        ]

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

    mask = gray > tol

    if not mask.any():
        return img

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)

    cropped = img[
        np.ix_(rows, cols)
    ]

    if 0 in cropped.shape[:2]:
        return img

    return cropped


# ============================================================
# PAD TO SQUARE
# MATCHES SENANUR SOURCE
# ============================================================

def pad_to_square(img):

    h, w = img.shape[:2]

    if h == w:
        return img

    size = max(h, w)

    top = (
        size - h
    ) // 2

    left = (
        size - w
    ) // 2

    output = np.zeros(
        (
            size,
            size,
            3,
        ),
        dtype=img.dtype,
    )

    output[
        top:top + h,
        left:left + w,
    ] = img

    return output


# ============================================================
# EXACT PREPROCESSING
# ============================================================

def preprocess_for_model(
    bgr,
):
    """
    Reproduce the official ONNX serving preprocessing.

    Official sequence:

        quality check
        -> auto crop
        -> no CLAHE
        -> square pad
        -> 512x512
        -> RGB
        -> 384x384
        -> /255
        -> ImageNet normalize
        -> CHW
        -> batch
    """

    # --------------------------------------------------------
    # Basic quality gate
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        gray.mean()
    )

    if brightness < 8.0:
        raise ValueError(
            "Image rejected: too dark"
        )

    if brightness > 250.0:
        raise ValueError(
            "Image rejected: too bright"
        )

    # --------------------------------------------------------
    # Auto crop
    # --------------------------------------------------------

    cropped = auto_crop(
        bgr,
        tol=7,
    )

    # --------------------------------------------------------
    # NO CLAHE
    # --------------------------------------------------------

    squared = pad_to_square(
        cropped
    )

    # --------------------------------------------------------
    # 512 x 512
    # --------------------------------------------------------

    processed = cv2.resize(
        squared,
        (
            PREPROCESS_SIZE,
            PREPROCESS_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # PIL resize
    # Official ONNX server uses PIL BILINEAR
    # --------------------------------------------------------

    pil = Image.fromarray(
        rgb
    )

    pil = pil.resize(
        (
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
        ),
        Image.BILINEAR,
    )

    x = np.asarray(
        pil,
        dtype=np.float32,
    ) / 255.0

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    x = (
        x - IMAGENET_MEAN
    ) / IMAGENET_STD

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    x = np.transpose(
        x,
        (2, 0, 1),
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    x = np.expand_dims(
        x,
        axis=0,
    ).astype(
        np.float32
    )

    return x, processed


# ============================================================
# SCORE -> GRADE
# ============================================================

def score_to_grade(
    score,
):

    return int(
        np.digitize(
            score,
            THRESHOLDS,
        )
    )


# ============================================================
# LOAD FIVE ONNX FOLDS
# ============================================================

def load_models():

    sessions = []

    for fold in range(1, 6):

        path = (
            MODEL_DIR
            / f"fold{fold}.onnx"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        options = ort.SessionOptions()

        # Match official deployment
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False

        session = (
            ort.InferenceSession(
                str(path),
                options,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        sessions.append(
            session
        )

    return sessions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SENANUR APTOS GRADER "
        "- CORRECT OFFICIAL PREPROCESSING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Export configuration
    # --------------------------------------------------------

    with open(
        EXPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        export_config = json.load(
            file
        )

    print(
        "\nExport configuration:"
    )

    print(
        json.dumps(
            export_config,
            indent=4,
        )
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{IMAGE_PATH}"
        )

    bgr = cv2.imread(
        str(IMAGE_PATH)
    )

    if bgr is None:
        raise RuntimeError(
            f"Could not decode image: "
            f"{IMAGE_PATH}"
        )

    print(
        "\nTesting:"
    )

    print(
        f"  {IMAGE_PATH}"
    )

    print(
        f"Original shape: "
        f"{bgr.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print(
        "\nApplying official "
        "preprocessing..."
    )

    model_input, processed = (
        preprocess_for_model(
            bgr
        )
    )

    print(
        f"Processed image shape: "
        f"{processed.shape}"
    )

    print(
        f"Model tensor shape: "
        f"{model_input.shape}"
    )

    print(
        f"Model tensor dtype: "
        f"{model_input.dtype}"
    )

    print(
        f"Normalized input range: "
        f"{model_input.min():.4f} "
        f"to "
        f"{model_input.max():.4f}"
    )

    # --------------------------------------------------------
    # Load folds
    # --------------------------------------------------------

    print(
        "\nLoading 5 ONNX folds..."
    )

    sessions = load_models()

    print(
        "All folds loaded."
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    scores = []

    print(
        "\nFold predictions:"
    )

    total_start = (
        time.perf_counter()
    )

    for index, session in enumerate(
        sessions,
        start=1,
    ):

        input_name = (
            session
            .get_inputs()[0]
            .name
        )

        start = (
            time.perf_counter()
        )

        output = session.run(
            None,
            {
                input_name:
                    model_input
            },
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        score = float(
            np.asarray(
                output[0]
            ).reshape(-1)[0]
        )

        scores.append(
            score
        )

        print(
            f"  Fold {index}: "
            f"{score:.6f} "
            f"({elapsed:.3f}s)"
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    scores_np = np.asarray(
        scores,
        dtype=np.float64,
    )

    mean_score = float(
        np.mean(scores_np)
    )

    fold_spread = float(
        np.std(scores_np)
    )

    grade = score_to_grade(
        mean_score
    )

    referable = (
        grade >= 2
    )

    # --------------------------------------------------------
    # Fold grades
    # --------------------------------------------------------

    fold_grades = [
        score_to_grade(
            score
        )
        for score in scores
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL CORRECTED RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "\nFold scores:"
    )

    for index, (
        score,
        fold_grade,
    ) in enumerate(
        zip(
            scores,
            fold_grades,
        ),
        start=1,
    ):

        print(
            f"  Fold {index}: "
            f"score={score:.6f} "
            f"grade={fold_grade}"
        )

    print(
        f"\nMean raw score: "
        f"{mean_score:.6f}"
    )

    print(
        f"Fold spread: "
        f"{fold_spread:.6f}"
    )

    print(
        f"\nFinal Grade: "
        f"{grade}"
    )

    print(
        f"Label: "
        f"{GRADE_NAMES[grade]}"
    )

    print(
        f"Referable: "
        f"{referable}"
    )

    print(
        f"\nTotal inference time: "
        f"{total_time:.3f}s"
    )

    print(
        f"Average per fold: "
        f"{total_time / 5:.3f}s"
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    unique_grades = set(
        fold_grades
    )

    print(
        "\nFold agreement:"
    )

    print(
        f"  Unique fold grades: "
        f"{sorted(unique_grades)}"
    )

    if len(unique_grades) == 1:

        print(
            "  Strong agreement: "
            "all folds predicted "
            "the same grade."
        )

    else:

        print(
            "  Folds disagree "
            "on the predicted grade."
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {
        "image": str(
            IMAGE_PATH
        ),
        "fold_scores": [
            float(x)
            for x in scores
        ],
        "fold_grades": fold_grades,
        "mean_raw_score": mean_score,
        "fold_spread": fold_spread,
        "grade": grade,
        "grade_label": GRADE_NAMES[
            grade
        ],
        "referable": referable,
        "preprocessing": {
            "quality_check": True,
            "auto_crop": True,
            "crop_tol": 7,
            "clahe": False,
            "square_mode": "pad",
            "preprocess_size": 512,
            "model_input_size": 384,
            "imagenet_normalization": True,
        },
        "inference_seconds": (
            total_time
        ),
    }

    output_path = Path(
        "senanur_corrected_test.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    print(
        f"\nResult saved to: "
        f"{output_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CORRECTED TEST COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()