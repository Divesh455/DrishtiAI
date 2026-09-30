from pathlib import Path

import pandas as pd

import torch

from PIL import Image

from torch.utils.data import Dataset


class APTOSDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe
            .iloc[index]
        )

        image_id = (
            row["id_code"]
        )

        label = int(
            row["diagnosis"]
        )


        image_path = (
            self.image_dir /
            f"{image_id}.png"
        )


        if not image_path.exists():

            raise FileNotFoundError(

                f"Image not found: "
                f"{image_path}"

            )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        if self.transform:

            image = self.transform(
                image
            )


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return image, label