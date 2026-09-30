import pandas as pd

from dataset import APTOSDataset
from config import TRAIN_CSV, IMAGE_DIR

from torchvision import transforms


def main():

    df = pd.read_csv(TRAIN_CSV)

    print("Dataset size:", len(df))

    print("\nFirst rows:")
    print(df.head())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = APTOSDataset(
        dataframe=df,
        image_dir=IMAGE_DIR,
        transform=transform
    )

    print("\nDataset length:", len(dataset))

    image, label = dataset[0]

    print("\nImage shape:", image.shape)

    print("Label:", label.item())


if __name__ == "__main__":
    main()
    