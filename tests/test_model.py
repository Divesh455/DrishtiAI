import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from PIL import Image

from torchvision import transforms

from backend.models.dr_classifier import create_model

from training.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    DEVICE,
    CLASS_NAMES
)


def load_model():

    model = create_model(
        num_classes=5,
        device=DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict_image(image_path):

    model = load_model()

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = (
        predicted_class.item()
    )

    confidence = (
        confidence.item()
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


if __name__ == "__main__":

    image_path = input(
        "Enter retinal image path: "
    )

    predicted_class, confidence, probabilities = (
        predict_image(image_path)
    )

    print("\n" + "=" * 50)

    print("DrishtiAI Prediction")

    print("=" * 50)

    print(
        f"Grade      : {predicted_class}"
    )

    print(
        f"Condition  : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for class_id, probability in enumerate(
        probabilities[0]
    ):

        print(
            f"Grade {class_id}: "
            f"{probability.item() * 100:.2f}%"
        )