from pathlib import Path

import cv2
import numpy as np
import torch

from backend.models.lesion_detector import LesionDetector


IMAGE_PATH = Path(
    "data/raw/idrid/images/training/IDRiD_23.jpg"
)


CLASS_NAMES = {
    0: "background",
    1: "microaneurysm",
    2: "hemorrhage",
    3: "hard_exudate",
    4: "soft_exudate",
}


def print_class_distribution(mask, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    total = mask.size

    for class_id, name in CLASS_NAMES.items():
        count = int(np.sum(mask == class_id))
        percentage = 100.0 * count / total

        print(
            f"{class_id}: {name:18s} "
            f"{count:10,d} pixels "
            f"({percentage:.4f}%)"
        )


def main():

    print("=" * 70)
    print("CURRENT LESION MODEL DIAGNOSTIC")
    print("=" * 70)

    detector = LesionDetector()

    model = detector.model

    device = getattr(
        detector,
        "device",
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    device = torch.device(device)

    model.to(device)
    model.eval()

    print("\nDevice:", device)

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    image_bgr = cv2.imread(
        str(IMAGE_PATH)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read {IMAGE_PATH}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # ---------------------------------------------------------
    # Match current model preprocessing
    # ---------------------------------------------------------

    image_512 = cv2.resize(
        image_rgb,
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )

    tensor = (
        torch.from_numpy(image_512)
        .float()
        / 255.0
    )

    tensor = (
        tensor
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # ---------------------------------------------------------
    # Direct model inference
    # ---------------------------------------------------------

    with torch.no_grad():
        output = model(tensor)

    if isinstance(output, tuple):
        output = output[0]

    if isinstance(output, dict):
        if "out" in output:
            output = output["out"]
        elif "logits" in output:
            output = output["logits"]

    print("\nOutput shape:", tuple(output.shape))

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    predicted = torch.argmax(
        probabilities,
        dim=1,
    )[0].cpu().numpy()

    print_class_distribution(
        predicted,
        "DIRECT MODEL ARGMAX"
    )

    # ---------------------------------------------------------
    # Probability statistics
    # ---------------------------------------------------------

    print("\nAverage class probabilities:")

    mean_probs = probabilities[0].mean(
        dim=(1, 2)
    ).cpu().numpy()

    max_probs = probabilities[0].amax(
        dim=(1, 2)
    ).cpu().numpy()

    for class_id, name in CLASS_NAMES.items():
        print(
            f"{class_id}: {name:18s} "
            f"mean={mean_probs[class_id]:.6f} "
            f"max={max_probs[class_id]:.6f}"
        )

    # ---------------------------------------------------------
    # Production detector
    # ---------------------------------------------------------

    print("\nRunning production detector...")

    result = detector.detect(
        image_bgr
    )

    print("\nProduction result keys:")

    if isinstance(result, dict):
        print(
            list(result.keys())
        )

        print("\nProduction lesion count:")

        if "lesions" in result:
            print(
                len(result["lesions"])
            )

            for lesion in result["lesions"][:20]:
                print(lesion)

        else:
            print(
                "No 'lesions' key found."
            )

    else:
        print(
            "Production result type:",
            type(result),
        )
        print(result)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
