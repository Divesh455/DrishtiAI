import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

import torch

from PIL import Image

from backend.models.dr_classifier import (
    create_model
)

from backend.xai.gradcam import (
    GradCAMExplainer
)

from training.config import (
    MODEL_PATH,
    DEVICE,
    IMAGE_SIZE,
    CLASS_NAMES
)


def load_trained_model():

    print(
        "Loading trained model..."
    )

    model = create_model(
        num_classes=5,
        device=DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "✓ Model loaded"
    )

    return model


def main():

    print("=" * 60)

    print(
        "DrishtiAI - Grad-CAM++ TEST"
    )

    print("=" * 60)

    # -------------------------------------------------
    # Image
    # -------------------------------------------------

    image_path = input(
        "\nEnter retinal image path: "
    ).strip()

    image = Image.open(
        image_path
    ).convert("RGB")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = load_trained_model()

    # -------------------------------------------------
    # Explainer
    # -------------------------------------------------

    explainer = GradCAMExplainer(
        model=model,
        device=DEVICE,
        image_size=IMAGE_SIZE
    )

    # -------------------------------------------------
    # Generate
    # -------------------------------------------------

    print(
        "\nGenerating Grad-CAM++..."
    )

    result = explainer.generate(
        image
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    output_name = Path(
        image_path
    ).stem

    paths = explainer.save_results(
        result,
        output_name
    )

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    predicted_class = (
        result["predicted_class"]
    )

    confidence = (
        result["confidence"]
    )

    print("\n" + "=" * 60)

    print(
        "RESULT"
    )

    print("=" * 60)

    print(
        f"Predicted Grade : "
        f"{predicted_class}"
    )

    print(
        f"Condition       : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence      : "
        f"{confidence * 100:.2f}%"
    )

    print("\nFiles generated:")

    print(
        f"Heatmap : "
        f"{paths['heatmap_path']}"
    )

    print(
        f"Overlay : "
        f"{paths['overlay_path']}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()