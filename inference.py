import torch
import cv2
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from utils import decode_segmap


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image at {path}")
    return image


def preprocess(image, device):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(
        image_rgb / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_rgb, image_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--mask_path", required=False, default=None,
                        help="Optional path to ground truth mask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Load Model
    # =========================
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        classes=5
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # =========================
    # Load & Preprocess Image
    # =========================
    image = load_image(args.image_path)
    print("Input Image Shape:", image.shape)
    image = image[:4096,:4096,:]

    image_rgb, image_tensor = preprocess(image, device)

    # =========================
    # Inference
    # =========================
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)[0]

    pred_np = pred.cpu().numpy()
    pred_color = decode_segmap(pred_np)

    # =========================
    # Prediction Overlay
    # =========================
    pred_overlay = cv2.addWeighted(image_rgb, 0.6, pred_color, 0.4, 0)

    cv2.imwrite(
        "images/prediction_overlay.png",
        cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR)
    )

    print("Saved: images/prediction_overlay.png")

    # =========================
    # Optional Ground Truth Overlay
    # =========================
    if args.mask_path is not None:
        mask = load_image(args.mask_path)
        
        # If mask is 3-channel, convert to single channel
        if len(mask.shape) == 3:
            
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask[:4096,:4096]
        mask_color = decode_segmap(mask)

        gt_overlay = cv2.addWeighted(image_rgb, 0.6, mask_color, 0.4, 0)

        cv2.imwrite(
            "images/groundtruth_overlay.png",
            cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR)
        )

        print("Saved: images/groundtruth_overlay.png")


if __name__ == "__main__":
    main()




# CUDA_VISIBLE_DEVICES=2 python inference.py --image_path data/raw/landcoverai/images/N-34-97-C-b-1-2.tif --model_path outputs/models/best_model_epoch320_miou0.8173.pth --mask_path data/raw/landcoverai/masks/N-34-97-C-b-1-2.tif