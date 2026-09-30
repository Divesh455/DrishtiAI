import random

import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(
    outputs,
    labels
):

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    return correct / total