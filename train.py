import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import LandCoverDataset, get_train_transform, get_val_transform
from utils import compute_iou, compute_confusion_matrix, save_loss_plot, save_visualization


def get_args():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)

    # Training args
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)

    # Validation control
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Run validation every N epochs")

    # Loss
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "dice"])

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/visuals", exist_ok=True)

    # -------------------
    # Datasets
    # -------------------
    train_dataset = LandCoverDataset(
        args.data_dir, args.train_split, get_train_transform()
    )
    val_dataset = LandCoverDataset(
        args.data_dir, args.val_split, get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    # -------------------
    # Model
    # -------------------
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        classes=args.num_classes,
        activation=None
    ).to(device)

    # -------------------
    # Loss
    # -------------------
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = smp.losses.DiceLoss(mode="multiclass")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    best_miou = 0

    print("\n🚀 Starting Training...\n")

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")

        # =========================
        # TRAIN
        # =========================
        model.train()
        running_loss = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (train_bar.n + 1)

            train_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{avg_loss:.4f}"
            })

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print(f"\n📊 Train Loss: {train_loss:.4f}")

        # =========================
        # VALIDATION (Interval-based)
        # =========================
        if (epoch + 1) % args.val_interval == 0:
            print("🔍 Running Validation...")

            model.eval()
            val_loss = 0
            total_ious = []
            conf_matrix = torch.zeros(args.num_classes, args.num_classes)

            val_bar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for images, masks in val_bar:
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)

                    ious = compute_iou(preds, masks)
                    total_ious.append(ious)

                    conf_matrix += compute_confusion_matrix(
                        preds, masks, args.num_classes
                    )

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            mean_ious = torch.tensor(total_ious).nanmean(dim=0)
            miou = mean_ious.mean().item()

            print(f"📊 Val Loss: {val_loss:.4f}")
            print(f"📊 Mean IoU: {miou:.4f}")

            for i, class_iou in enumerate(mean_ious):
                print(f"   Class {i} IoU: {class_iou:.4f}")

            # Save best model
            if miou > best_miou:
                best_miou = miou
                save_path = f"outputs/models/best_model_epoch{epoch+1}_miou{miou:.4f}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved: {save_path}")

            # Save visualization
            save_visualization(
                images[0].cpu(),
                preds[0].cpu(),
                masks[0].cpu(),
                f"outputs/visuals/epoch_{epoch+1}.png"
            )

        else:
            print(f"⏭️ Skipping validation (will run every {args.val_interval} epochs)")

    # =========================
    # Save Loss Curve
    # =========================
    save_loss_plot(
        train_losses,
        val_losses,
        "outputs/plots/loss_curve.png"
    )

    print("\n🎉 Training Complete!")
    print(f"🏆 Best mIoU achieved: {best_miou:.4f}")


if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=3 python train.py --data_dir data/raw/landcoverai/output --train_split data/raw/landcoverai/train.txt --val_split data/raw/landcoverai/val.txt --val_interval 1

# tmux attach  -t cover