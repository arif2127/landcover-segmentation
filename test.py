import torch
import argparse
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import LandCoverDataset, get_val_transform
from utils import compute_confusion_matrix, compute_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LandCoverDataset(args.data_dir, args.test_split, get_val_transform())
    loader = DataLoader(dataset, batch_size=4)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        classes=5
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    total_ious = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds, masks)
            total_ious.append(ious)

    mean_ious = torch.tensor(total_ious).nanmean(dim=0)
    print("Class IoU:", mean_ious)
    print("Mean IoU:", mean_ious.mean())


if __name__ == "__main__":
    main()



# python test.py --data_dir data/raw/landcoverai/output --test_split data/raw/landcoverai/test.txt --model_path outputs/models/best_model_epochX_miouX.pth