from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

def get_train_transforms(image_size=224):

    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.VerticalFlip(
            p=0.1
        ),

        A.RandomRotate90(
            p=0.3
        ),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20
            ),

            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8)
            ),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(
                blur_limit=(3, 5)
            ),

            A.MotionBlur(
                blur_limit=3
            ),
        ], p=0.15),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),

        ToTensorV2(),
    ])


# ============================================================
# VALIDATION TRANSFORMS
# ============================================================

def get_val_transforms(image_size=224):

    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),

        ToTensorV2(),
    ])


# ============================================================
# DATASET
# ============================================================

class DRDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = Path(image_dir)

        self.transform = transform

        self.image_ids = (
            self.dataframe["id_code"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            self.dataframe["diagnosis"]
            .astype(int)
            .tolist()
        )


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        image_id = self.image_ids[index]

        label = self.labels[index]

        image_path = self.image_dir / f"{image_id}.png"

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform is not None:

            transformed = self.transform(
                image=image
            )

            image = transformed["image"]

        return image, torch.tensor(
            label,
            dtype=torch.long
        )