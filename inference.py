import torch
import cv2
import argparse
import segmentation_models_pytorch as smp
from utils import decode_segmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        classes=5
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    image = cv2.imread(args.image_path)
    print(image.shape)
    image = image[:4096,:4096,:]
    cv2.imwrite("images/image.png", image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb / 255.0,
                                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)[0]

    pred_color = decode_segmap(pred.cpu().numpy())
    overlay = cv2.addWeighted(image_rgb, 0.6, pred_color, 0.4, 0)

    
    cv2.imwrite("images/overlay205.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=2 python inference.py --image_path data/raw/landcoverai/images/N-34-97-C-b-1-2.tif --model_path outputs/models/best_model_epoch205_miou0.8133.pth