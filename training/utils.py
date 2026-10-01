import random

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    labels,
    num_classes=5
):

    labels = np.asarray(labels)

    class_counts = np.bincount(
        labels,
        minlength=num_classes
    )

    total = len(labels)

    weights = []

    for count in class_counts:

        if count == 0:

            weights.append(0.0)

        else:

            weight = total / (
                num_classes * count
            )

            weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )