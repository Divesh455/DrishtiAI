import json
from pathlib import Path

import onnxruntime as ort


MODEL_DIR = Path(
    "external_models/senanur_dr"
)


def inspect_model(model_path: Path):
    print("\n" + "=" * 70)
    print(f"MODEL: {model_path.name}")
    print("=" * 70)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    print("\nInputs:")

    for item in session.get_inputs():
        print(
            f"  name={item.name}"
            f" | shape={item.shape}"
            f" | type={item.type}"
        )

    print("\nOutputs:")

    for item in session.get_outputs():
        print(
            f"  name={item.name}"
            f" | shape={item.shape}"
            f" | type={item.type}"
        )

    print("\nProviders:")
    print(session.get_providers())


def main():

    print("=" * 70)
    print("SENANUR APTOS DR MODEL INSPECTION")
    print("=" * 70)

    # Read export.json
    export_path = MODEL_DIR / "export.json"

    with open(
        export_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    print("\nexport.json:")
    print(
        json.dumps(
            config,
            indent=4,
        )
    )

    # Inspect all five folds
    for fold in range(1, 6):
        model_path = (
            MODEL_DIR
            / f"fold{fold}.onnx"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        inspect_model(model_path)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()