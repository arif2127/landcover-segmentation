import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class LandCoverDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        """
        data_dir: path to output folder
        split_file: train.txt / val.txt / test.txt
        """
        self.data_dir = data_dir
        self.transform = transform

        with open(split_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]

        img_path = os.path.join(self.data_dir, f"{name}.jpg")
        mask_path = os.path.join(self.data_dir, f"{name}_m.png")

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (already class index)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# -----------------------------
# Transform Functions
# -----------------------------
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def get_val_transform():
    return A.Compose([])


# -----------------------------
# Main Test Function
# -----------------------------
if __name__ == "__main__":

    DATA_DIR = "data/raw/landcoverai/output"
    TRAIN_SPLIT = "data/raw/landcoverai/train.txt"

    dataset = LandCoverDataset(
        data_dir=DATA_DIR,
        split_file=TRAIN_SPLIT,
        transform=get_train_transform()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    images, masks = next(iter(dataloader))

    print("Batch Image Shape:", images.shape)
    print("Batch Mask Shape:", masks.shape)
    print("Unique mask values in batch:", torch.unique(masks))
