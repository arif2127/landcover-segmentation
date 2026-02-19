import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


# -------------------------
# Color Map (5 Classes)
# -------------------------
COLOR_MAP = {
    0: (192, 192, 192),      # Background / Ground - Grey
    1: (255, 0, 0),      # Building - Red
    2: (34, 139, 34),    # Woodland - Dark Green
    3: (0, 0, 255),      # Water - Blue
    4: (255, 255, 0)     # Road - Yellow
}


def decode_segmap(mask):
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        output[mask == class_id] = color

    return output


# -------------------------
# IoU Calculation
# -------------------------
def compute_iou(pred, target, num_classes=5):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious


# -------------------------
# Confusion Matrix
# -------------------------
def compute_confusion_matrix(pred, target, num_classes=5):
    matrix = np.zeros((num_classes, num_classes))

    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()

    for p, t in zip(pred, target):
        matrix[t, p] += 1

    return matrix


# -------------------------
# Save Loss Plot
# -------------------------
def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig(save_path)
    plt.close()


# -------------------------
# Save Visualization
# -------------------------
def save_visualization(image, pred, target, save_path):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    pred_color = decode_segmap(pred.cpu().numpy())
    target_color = decode_segmap(target.cpu().numpy())

    overlay_pred = cv2.addWeighted(image, 0.6, pred_color, 0.4, 0)
    overlay_target = cv2.addWeighted(image, 0.6, target_color, 0.4, 0)

    combined = np.hstack([image, overlay_pred, overlay_target])
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
