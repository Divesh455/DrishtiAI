import sys

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


if str(PROJECT_ROOT) not in sys.path:

    sys.path.append(
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PROJECT IMPORTS
# ============================================================

from training.config import (

    TRAIN_CSV,

    IMAGE_DIR,

    MODEL_PATH,

    IMAGE_SIZE,

    BATCH_SIZE,

    VALIDATION_SIZE,

    RANDOM_SEED,

    DEVICE,

    NUM_CLASSES,

    CLASS_NAMES

)


from training.dataset import (
    APTOSDataset
)


from backend.models.dr_classifier import (
    create_model
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model():

    print("\n")

    print("=" * 70)

    print(
        "Loading trained DR model..."
    )

    print("=" * 70)


    model = create_model(

        num_classes=NUM_CLASSES,

        device=DEVICE

    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]

        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model.eval()


    print(
        "✓ Model loaded successfully"
    )


    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "epoch"
            in checkpoint
        ):

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )


        if (
            "best_val_macro_f1"
            in checkpoint
        ):

            print(
                f"Checkpoint Macro-F1: "
                f"{checkpoint['best_val_macro_f1']:.4f}"
            )


        if (
            "best_val_accuracy"
            in checkpoint
        ):

            print(
                f"Checkpoint accuracy: "
                f"{checkpoint['best_val_accuracy']:.4f}"
            )


    return model


# ============================================================
# 2. CREATE VALIDATION DATASET
# ============================================================

def create_validation_dataset():

    print("\n")

    print("=" * 70)

    print(
        "Creating validation dataset..."
    )

    print("=" * 70)


    df = pd.read_csv(
        TRAIN_CSV
    )


    print(
        f"Total dataset images: "
        f"{len(df)}"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This uses the exact same split
    # configuration used during training.
    # --------------------------------------------------------

    _, val_df = train_test_split(

        df,

        test_size=
            VALIDATION_SIZE,

        random_state=
            RANDOM_SEED,

        stratify=
            df["diagnosis"]

    )


    print(
        f"Validation images: "
        f"{len(val_df)}"
    )


    print("\nValidation class distribution:")


    distribution = (
        val_df[
            "diagnosis"
        ]
        .value_counts()
        .sort_index()
    )


    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            distribution.get(
                class_id,
                0
            )
        )


        print(

            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<20} : "
            f"{count}"

        )


    # --------------------------------------------------------
    # Validation transform
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(

            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )

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


    dataset = APTOSDataset(

        dataframe=val_df,

        image_dir=IMAGE_DIR,

        transform=transform

    )


    return dataset


# ============================================================
# 3. RUN MODEL PREDICTIONS
# ============================================================

def get_predictions(
    model,
    loader
):

    print("\n")

    print("=" * 70)

    print(
        "Running model predictions..."
    )

    print("=" * 70)


    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(
            loader
        ):


            images = images.to(
                DEVICE
            )


            outputs = model(
                images
            )


            probabilities = torch.softmax(

                outputs,

                dim=1

            )


            predictions = torch.argmax(

                probabilities,

                dim=1

            )


            all_labels.extend(

                labels
                .cpu()
                .numpy()

            )


            all_predictions.extend(

                predictions
                .cpu()
                .numpy()

            )


            all_probabilities.extend(

                probabilities
                .cpu()
                .numpy()

            )


            print(

                f"\rProcessed "
                f"{batch_index + 1}/"
                f"{len(loader)} batches",

                end=""

            )


    print("\n")

    print(
        "✓ Prediction completed"
    )


    return (

        all_labels,

        all_predictions,

        all_probabilities

    )


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

def calculate_metrics(

    labels,

    predictions

):


    accuracy = accuracy_score(

        labels,

        predictions

    )


    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    macro_precision = (
        precision_score(

            labels,

            predictions,

            average="macro",

            zero_division=0

        )
    )


    macro_recall = (
        recall_score(

            labels,

            predictions,

            average="macro",

            zero_division=0

        )
    )


    macro_f1 = (
        f1_score(

            labels,

            predictions,

            average="macro",

            zero_division=0

        )
    )


    # --------------------------------------------------------
    # Weighted metrics
    # --------------------------------------------------------

    weighted_precision = (
        precision_score(

            labels,

            predictions,

            average="weighted",

            zero_division=0

        )
    )


    weighted_recall = (
        recall_score(

            labels,

            predictions,

            average="weighted",

            zero_division=0

        )
    )


    weighted_f1 = (
        f1_score(

            labels,

            predictions,

            average="weighted",

            zero_division=0

        )
    )


    return {

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall,

        "weighted_f1":
            weighted_f1

    }


# ============================================================
# 5. PRINT RESULTS
# ============================================================

def print_results(

    labels,

    predictions

):


    metrics = calculate_metrics(

        labels,

        predictions

    )


    print("\n")

    print("=" * 70)

    print(
        "OVERALL MODEL RESULTS"
    )

    print("=" * 70)


    print(

        f"Accuracy            : "
        f"{metrics['accuracy']:.4f}"

    )


    print("\nMacro metrics:")


    print(

        f"Macro Precision     : "
        f"{metrics['macro_precision']:.4f}"

    )


    print(

        f"Macro Recall        : "
        f"{metrics['macro_recall']:.4f}"

    )


    print(

        f"Macro F1            : "
        f"{metrics['macro_f1']:.4f}"

    )


    print("\nWeighted metrics:")


    print(

        f"Weighted Precision  : "
        f"{metrics['weighted_precision']:.4f}"

    )


    print(

        f"Weighted Recall     : "
        f"{metrics['weighted_recall']:.4f}"

    )


    print(

        f"Weighted F1         : "
        f"{metrics['weighted_f1']:.4f}"

    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "PER-CLASS CLASSIFICATION REPORT"
    )

    print("=" * 70)


    target_names = [

        CLASS_NAMES[index]

        for index in range(
            NUM_CLASSES
        )

    ]


    report = classification_report(

        labels,

        predictions,

        labels=list(
            range(NUM_CLASSES)
        ),

        target_names=
            target_names,

        digits=4,

        zero_division=0

    )


    print(report)


    return metrics


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(

    labels,

    predictions

):


    matrix = confusion_matrix(

        labels,

        predictions,

        labels=list(
            range(NUM_CLASSES)
        )

    )


    print("\n")

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)


    print(matrix)


    # --------------------------------------------------------
    # Save matrix image
    # --------------------------------------------------------

    output_path = (

        Path(__file__).resolve().parent
        /
        "confusion_matrix.png"

    )


    class_labels = [

        CLASS_NAMES[index]

        for index in range(
            NUM_CLASSES
        )

    ]


    plt.figure(

        figsize=(9, 7)

    )


    plt.imshow(
        matrix
    )


    plt.title(
        "DrishtiAI - DR Confusion Matrix"
    )


    plt.colorbar()


    plt.xticks(

        range(NUM_CLASSES),

        class_labels,

        rotation=45,

        ha="right"

    )


    plt.yticks(

        range(NUM_CLASSES),

        class_labels

    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "Actual Class"
    )


    # --------------------------------------------------------
    # Write values
    # --------------------------------------------------------

    for row in range(
        NUM_CLASSES
    ):

        for column in range(
            NUM_CLASSES
        ):

            plt.text(

                column,

                row,

                str(
                    matrix[
                        row,
                        column
                    ]
                ),

                ha="center",

                va="center"

            )


    plt.tight_layout()


    plt.savefig(

        output_path,

        dpi=200,

        bbox_inches="tight"

    )


    plt.close()


    print(

        f"\n✓ Confusion matrix saved:"
        f"\n{output_path}"

    )


    return matrix


# ============================================================
# 7. MAIN EVALUATION
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "DrishtiAI - FINAL DR MODEL EVALUATION"
    )

    print("=" * 70)


    print(
        f"\nDevice: {DEVICE}"
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        create_validation_dataset()
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (

        labels,

        predictions,

        probabilities

    ) = get_predictions(

        model,

        loader

    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = print_results(

        labels,

        predictions

    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = create_confusion_matrix(

        labels,

        predictions

    )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 70)


    print(

        f"Accuracy       : "
        f"{metrics['accuracy']:.4f}"

    )


    print(

        f"Macro F1       : "
        f"{metrics['macro_f1']:.4f}"

    )


    print(

        f"Weighted F1    : "
        f"{metrics['weighted_f1']:.4f}"

    )


    print(
        f"Samples tested : "
        f"{len(labels)}"
    )


    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()