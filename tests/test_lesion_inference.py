import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import cv2
import numpy as np
import torch

from training.config import (
    IDRID_IMAGE_DIR,
    WEIGHTS_DIR,
    NUM_LESION_CLASSES,
)

from backend.models.lesion_detector import (
    create_lesion_model
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512

MODEL_PATH = (
    WEIGHTS_DIR /
    "lesion_model.pth"
)

OUTPUT_DIR = Path(
    "backend/outputs/lesion_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLASS_NAMES = {
    1: "Microaneurysm",
    2: "Hemorrhage",
    3: "Hard Exudate",
    4: "Soft Exudate",
}


# ============================================================
# FIND IMAGE
# ============================================================

def get_test_image():

    images = sorted(
        Path(IDRID_IMAGE_DIR).glob("*.jpg")
    )

    if not images:

        raise FileNotFoundError(
            "No IDRiD images found."
        )

    return images[0]


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# CREATE COLORED MASK
# ============================================================

def create_colored_mask(prediction):

    output = np.zeros(
        (
            prediction.shape[0],
            prediction.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # BGR colors

    # Microaneurysm
    output[
        prediction == 1
    ] = (0, 0, 255)

    # Hemorrhage
    output[
        prediction == 2
    ] = (0, 165, 255)

    # Hard Exudate
    output[
        prediction == 3
    ] = (0, 255, 255)

    # Soft Exudate
    output[
        prediction == 4
    ] = (0, 255, 0)

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DrishtiAI - LESION MODEL INFERENCE"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = create_lesion_model(
        num_classes=NUM_LESION_CLASSES,
        device=device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_path = get_test_image()

    print(
        f"\nImage:"
    )

    print(image_path)

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise ValueError(
            "Could not read image."
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = preprocess(
        original
    ).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0]

    prediction = (
        prediction
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # --------------------------------------------------------
    # Resize original
    # --------------------------------------------------------

    resized_original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create mask
    # --------------------------------------------------------

    colored_mask = create_colored_mask(
        prediction
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        resized_original,
        0.65,
        colored_mask,
        0.35,
        0
    )

    # --------------------------------------------------------
    # Count detected pixels
    # --------------------------------------------------------

    print("\nPredicted lesion regions:")

    for class_id, class_name in CLASS_NAMES.items():

        pixel_count = int(
            np.sum(
                prediction == class_id
            )
        )

        print(
            f"  {class_name}: "
            f"{pixel_count} pixels"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    image_id = image_path.stem

    original_path = (
        OUTPUT_DIR /
        f"{image_id}_original.jpg"
    )

    mask_path = (
        OUTPUT_DIR /
        f"{image_id}_predicted_mask.png"
    )

    overlay_path = (
        OUTPUT_DIR /
        f"{image_id}_lesion_overlay.jpg"
    )

    cv2.imwrite(
        str(original_path),
        resized_original
    )

    cv2.imwrite(
        str(mask_path),
        colored_mask
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    print("\nSaved:")

    print(
        original_path
    )

    print(
        mask_path
    )

    print(
        overlay_path
    )

    print("\nInference completed.")


if __name__ == "__main__":
    main()