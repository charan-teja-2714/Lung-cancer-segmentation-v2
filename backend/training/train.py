import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np


DATA_ROOT = "data/raw"
SAVE_DIR = "models/segmentation_multiclass/weights"

IMAGE_SIZE = 256
BATCH_SIZE = 4  
MAX_EPOCHS = 120
PATIENCE = 10
LR = 1e-4

RESUME = True  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
BEST_CKPT = os.path.join(SAVE_DIR, "best_multiclass.pth")

CLASS_NAMES = {
    1: "ADC",
    2: "LCC",
    3: "SCC"
}

print("Using device:", DEVICE)

# =========================================================
# DATASET
# =========================================================
class LungMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", image_size=256):
        self.image_size = image_size
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        for cls_id, cls_name in CLASS_NAMES.items():
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

        assert len(self.samples) > 0, "Dataset empty"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, cls_id = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = np.clip(img, -160, 240)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * cls_id

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        return img, mask

# =========================================================
# PER-CLASS DICE
# =========================================================
def per_class_dice(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1)
    dice = {}

    for cls in range(1, num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()

        inter = (p * t).sum()
        union = p.sum() + t.sum()

        dice[cls] = (2 * inter + 1e-6) / (union + 1e-6)

    return dice

# =========================================================
# DATA LOADERS
# =========================================================
train_dataset = LungMultiClassDataset(DATA_ROOT, split="train")
val_dataset   = LungMultiClassDataset(DATA_ROOT, split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, pin_memory=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, pin_memory=True, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

# =========================================================
# MODEL
# =========================================================
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4   # background + ADC + LCC + SCC
).to(DEVICE)

# =========================================================
# LOSS
# =========================================================
ce_loss = nn.CrossEntropyLoss()
dice_loss = smp.losses.DiceLoss(mode="multiclass")

def loss_fn(pred, target):
    return ce_loss(pred, target) + dice_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

# =========================================================
# RESUME
# =========================================================
start_epoch = 1
best_mean_dice = 0.0
patience_counter = 0

if RESUME and os.path.exists(LAST_CKPT):
    ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_mean_dice = ckpt["best_mean_dice"]
    patience_counter = 0  # Reset patience counter

    print(f"🔄 Resumed from epoch {start_epoch}")

# =========================================================
# TRAINING LOOP
# =========================================================
for epoch in range(start_epoch, MAX_EPOCHS + 1):

    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for imgs, masks in loop:
        try:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"Error in batch: {e}")
            torch.cuda.empty_cache()
            continue

    # ---------------- VALIDATION ----------------
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            all_preds.append(torch.argmax(preds, dim=1).cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    dice_avg = {}
    for cls in range(1, 4):
        p = (all_preds == cls).float()
        t = (all_masks == cls).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice_avg[cls] = ((2 * inter + 1e-6) / (union + 1e-6)).item()

    mean_dice = sum(dice_avg.values()) / 3

    print("\n------------------------------------")
    for k, v in dice_avg.items():
        print(f"{CLASS_NAMES[k]} Dice: {v:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("------------------------------------")

    scheduler.step(mean_dice)

    # ---------------- SAVE LAST ----------------
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_mean_dice": best_mean_dice,
        "patience_counter": patience_counter
    }, LAST_CKPT)

    # ---------------- SAVE BEST ----------------
    if mean_dice > best_mean_dice:
        best_mean_dice = mean_dice
        patience_counter = 0

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_mean_dice": best_mean_dice
        }, BEST_CKPT)

        print(f"🔥 New Best Mean Dice: {best_mean_dice:.4f}")

    else:
        patience_counter += 1
        print(f"⏳ No improvement ({patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print("🛑 Early stopping triggered")
        break

print(f"\n✅ FINAL BEST MEAN DICE: {best_mean_dice:.4f}")
