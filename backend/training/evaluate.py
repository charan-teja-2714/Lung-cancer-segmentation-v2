import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "data/raw"
MODEL_PATH = "models/segmentation_multiclass/weights/best_multiclass.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    "ADC": 1,
    "LCC": 2,
    "SCC": 3
}

NUM_CLASSES = 4  # background + 3 tumor classes

print("Using device:", DEVICE)

# =========================================================
# DATASET (same as training)
# =========================================================
class LungMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="test", image_size=256):
        self.image_size = image_size
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        for cls_name, cls_id in CLASS_NAMES.items():
            ct_dir = os.path.join(ct_root, cls_name)
            mask_dir = os.path.join(mask_root, cls_name)

            if not os.path.isdir(ct_dir):
                continue

            for fname in os.listdir(ct_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(ct_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, cls_id))

        assert len(self.samples) > 0, "Dataset is empty"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, cls_id = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = np.clip(img, -160, 240)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * cls_id

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        return img, mask

# =========================================================
# LOAD DATA
# =========================================================
dataset = LungMultiClassDataset(
    data_root=DATA_ROOT,
    split="test",
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

print(f"Test samples: {len(dataset)}")

# =========================================================
# LOAD MODEL
# =========================================================
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=1,
    classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model.eval()

# =========================================================
# DICE COMPUTATION (GLOBAL)
# =========================================================
intersection = {cls_id: 0.0 for cls_id in CLASS_NAMES.values()}
union = {cls_id: 0.0 for cls_id in CLASS_NAMES.values()}

with torch.no_grad():
    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for cls_id in CLASS_NAMES.values():
            pred_cls = (preds == cls_id).float()
            mask_cls = (masks == cls_id).float()

            intersection[cls_id] += (pred_cls * mask_cls).sum().item()
            union[cls_id] += pred_cls.sum().item() + mask_cls.sum().item()

# =========================================================
# FINAL RESULTS
# =========================================================
print("\n================ DICE RESULTS ================")

dice_scores = {}
for cls_name, cls_id in CLASS_NAMES.items():
    dice = (2 * intersection[cls_id] + 1e-6) / (union[cls_id] + 1e-6)
    dice_scores[cls_name] = dice
    print(f"{cls_name} Dice: {dice:.4f}")

mean_dice = sum(dice_scores.values()) / len(dice_scores)
print("---------------------------------------------")
print(f"Mean Dice (ADC + LCC + SCC): {mean_dice:.4f}")
print("=============================================")
