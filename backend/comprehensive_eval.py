import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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

print("=" * 70)
print("COMPREHENSIVE EVALUATION METRICS")
print("=" * 70)
print(f"Device: {DEVICE}\n")

# =========================================================
# DATASET
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

        assert len(self.samples) > 0

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
# METRICS
# =========================================================
def dice_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_recall(pred, target, smooth=1e-6):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision, recall

def hausdorff_distance_95(pred, target):
    """Calculate 95th percentile Hausdorff Distance"""
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    
    # Convert to numpy
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)
    
    # Get surface points
    pred_surface = pred_np ^ np.roll(pred_np, 1, axis=0) | pred_np ^ np.roll(pred_np, 1, axis=1)
    target_surface = target_np ^ np.roll(target_np, 1, axis=0) | target_np ^ np.roll(target_np, 1, axis=1)
    
    if not pred_surface.any() or not target_surface.any():
        return float('inf')
    
    # Distance transforms
    pred_dist = distance_transform_edt(~pred_surface)
    target_dist = distance_transform_edt(~target_surface)
    
    # Get distances
    pred_to_target = pred_dist[target_surface]
    target_to_pred = target_dist[pred_surface]
    
    # 95th percentile
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    hd95 = np.percentile(all_distances, 95)
    
    return hd95

# =========================================================
# LOAD DATA & MODEL
# =========================================================
dataset = LungMultiClassDataset(DATA_ROOT, split="test", image_size=IMAGE_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print(f"Test samples: {len(dataset)}\n")

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=1,
    classes=4
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# =========================================================
# EVALUATION
# =========================================================
metrics = {
    0: {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": []},
    1: {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": []},
    2: {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": []},
    3: {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": []}
}

all_preds = []
all_targets = []

print("Evaluating...")
with torch.no_grad():
    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        # Store for confusion matrix
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(masks.cpu().numpy().flatten())

        # Calculate metrics per class
        for cls_id in range(4):
            pred_cls = (preds == cls_id).float()
            mask_cls = (masks == cls_id).float()

            # Dice & IoU
            dice = dice_score(pred_cls, mask_cls)
            iou = iou_score(pred_cls, mask_cls)
            
            # Precision & Recall
            prec, rec = precision_recall(pred_cls, mask_cls)
            
            # HD95 (only for tumor classes)
            if cls_id > 0:
                try:
                    hd95 = hausdorff_distance_95(pred_cls[0], mask_cls[0])
                    if not np.isinf(hd95):
                        metrics[cls_id]["hd95"].append(hd95)
                except:
                    pass
            
            metrics[cls_id]["dice"].append(dice.item())
            metrics[cls_id]["iou"].append(iou.item())
            metrics[cls_id]["precision"].append(prec.item())
            metrics[cls_id]["recall"].append(rec.item())

# =========================================================
# RESULTS
# =========================================================
print("\n" + "=" * 70)
print("DETAILED METRICS PER CLASS")
print("=" * 70)

class_labels = {0: "Background", 1: "ADC", 2: "LCC", 3: "SCC"}

for cls_id in range(4):
    print(f"\n{class_labels[cls_id]} (Class {cls_id}):")
    print(f"  Dice Score:  {np.mean(metrics[cls_id]['dice']):.4f} ± {np.std(metrics[cls_id]['dice']):.4f}")
    print(f"  IoU:         {np.mean(metrics[cls_id]['iou']):.4f} ± {np.std(metrics[cls_id]['iou']):.4f}")
    print(f"  Precision:   {np.mean(metrics[cls_id]['precision']):.4f} ± {np.std(metrics[cls_id]['precision']):.4f}")
    print(f"  Recall:      {np.mean(metrics[cls_id]['recall']):.4f} ± {np.std(metrics[cls_id]['recall']):.4f}")
    
    if cls_id > 0 and len(metrics[cls_id]['hd95']) > 0:
        print(f"  HD95:        {np.mean(metrics[cls_id]['hd95']):.2f} ± {np.std(metrics[cls_id]['hd95']):.2f} pixels")

# Mean metrics (excluding background)
print("\n" + "=" * 70)
print("MEAN METRICS (ADC + LCC + SCC)")
print("=" * 70)

mean_dice = np.mean([np.mean(metrics[i]['dice']) for i in [1, 2, 3]])
mean_iou = np.mean([np.mean(metrics[i]['iou']) for i in [1, 2, 3]])
mean_precision = np.mean([np.mean(metrics[i]['precision']) for i in [1, 2, 3]])
mean_recall = np.mean([np.mean(metrics[i]['recall']) for i in [1, 2, 3]])

print(f"Mean Dice:      {mean_dice:.4f}")
print(f"Mean IoU:       {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall:    {mean_recall:.4f}")

# HD95 for tumor classes
hd95_values = []
for i in [1, 2, 3]:
    if len(metrics[i]['hd95']) > 0:
        hd95_values.extend(metrics[i]['hd95'])

if len(hd95_values) > 0:
    print(f"Mean HD95:      {np.mean(hd95_values):.2f} ± {np.std(hd95_values):.2f} pixels")

# =========================================================
# CONFUSION MATRIX
# =========================================================
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
print("\nRows: True Labels | Columns: Predicted Labels")
print("Classes: [Background, ADC, LCC, SCC]\n")
print(cm)

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nNormalized Confusion Matrix (%):")
print((cm_normalized * 100).astype(int))

# =========================================================
# STATISTICAL VALIDATION
# =========================================================
print("\n" + "=" * 70)
print("STATISTICAL VALIDATION")
print("=" * 70)

print("\nValidation Strategy:")
print("  - Single Train/Test Split (80/20)")
print("  - Training Set: 4,693 slices from 148 patients")
print("  - Test Set: 1,140 slices from 147 patients")
print("  - No cross-validation (patient-level split)")
print("  - Early stopping with patience=10 epochs")
print("  - Best model selected based on validation Dice score")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
