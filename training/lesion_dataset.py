import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class IDRiDLesionDataset(Dataset):

    """
    IDRiD multi-class lesion segmentation dataset.

    Classes:

    0 = Background
    1 = Microaneurysm
    2 = Hemorrhage
    3 = Hard Exudate
    4 = Soft Exudate
    """

    def __init__(
        self,
        image_paths,
        image_size=512,
        augment=False
    ):

        self.image_paths = [
            Path(path)
            for path in image_paths
        ]

        self.image_size = image_size
        self.augment = augment

    def __len__(self):

        return len(self.image_paths)

    def _get_mask_path(
        self,
        image_path,
        directory,
        suffix
    ):

        image_id = image_path.stem

        path = (
            Path(directory)
            / f"{image_id}_{suffix}.tif"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Mask not found: {path}"
            )

        return path

    def _load_mask(
        self,
        path,
        width,
        height
    ):

        mask = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise ValueError(
                f"Could not read mask: {path}"
            )

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        return mask > 0

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize image
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        height = self.image_size
        width = self.image_size

        # ----------------------------------------------------
        # Import configuration
        # ----------------------------------------------------

        from training.config import (
            IDRID_MICROANEURYSM_DIR,
            IDRID_HEMORRHAGE_DIR,
            IDRID_HARD_EXUDATE_DIR,
            IDRID_SOFT_EXUDATE_DIR,
        )

        # ----------------------------------------------------
        # Load masks
        # ----------------------------------------------------

        ma_path = self._get_mask_path(
            image_path,
            IDRID_MICROANEURYSM_DIR,
            "MA"
        )

        he_path = self._get_mask_path(
            image_path,
            IDRID_HEMORRHAGE_DIR,
            "HE"
        )

        ex_path = self._get_mask_path(
            image_path,
            IDRID_HARD_EXUDATE_DIR,
            "EX"
        )

        se_path = self._get_mask_path(
            image_path,
            IDRID_SOFT_EXUDATE_DIR,
            "SE"
        )

        ma_mask = self._load_mask(
            ma_path,
            width,
            height
        )

        he_mask = self._load_mask(
            he_path,
            width,
            height
        )

        ex_mask = self._load_mask(
            ex_path,
            width,
            height
        )

        se_mask = self._load_mask(
            se_path,
            width,
            height
        )

        # ----------------------------------------------------
        # Create multi-class mask
        # ----------------------------------------------------

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        mask[ma_mask] = 1
        mask[he_mask] = 2
        mask[ex_mask] = 3
        mask[se_mask] = 4

        # ----------------------------------------------------
        # Simple augmentation
        # ----------------------------------------------------

        if self.augment:

            if np.random.random() < 0.5:

                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if np.random.random() < 0.5:

                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image_tensor, mask_tensor