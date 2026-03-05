import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp

# Configuration
DATA_ROOT = "data/raw"
MODEL_PATH = "models/segmentation_multiclass/weights/best_multiclass.pth"
IMAGE_SIZE = 256
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

CLASS_NAMES = ['Background', 'ADC', 'LCC', 'SCC']
CLASS_MAP = {1: "ADC", 2: "LCC", 3: "SCC"}

print(f"Using device: {DEVICE}")

# Dataset class
class LungDataset(Dataset):
    def __init__(self, data_root, split="test", image_size=256):
        self.image_size = image_size
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        for cls_id, cls_name in CLASS_MAP.items():
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
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        return img, mask

def evaluate_dataset(model, loader, split_name):
    """Evaluate model on a dataset and return metrics"""
    model.eval()
    dice_scores = {i: [] for i in range(1, 4)}
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            # Calculate dice per class
            for cls in range(1, 4):
                pred_cls = (preds == cls).float()
                mask_cls = (masks == cls).float()
                
                intersection = (pred_cls * mask_cls).sum(dim=(1, 2))
                union = pred_cls.sum(dim=(1, 2)) + mask_cls.sum(dim=(1, 2))
                
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                dice_scores[cls].extend(dice.cpu().numpy())
    
    mean_dice = {cls: np.mean(dice_scores[cls]) for cls in range(1, 4)}
    
    return {
        'dice_scores': dice_scores,
        'mean_dice': mean_dice,
        'all_preds': all_preds,
        'all_targets': all_targets
    }

# Load model
print("Loading model...")
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=1,
    classes=4
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("✅ Model loaded")

# Load datasets
print("\nLoading datasets...")
train_dataset = LungDataset(DATA_ROOT, split="train", image_size=IMAGE_SIZE)
test_dataset = LungDataset(DATA_ROOT, split="test", image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Test samples: {len(test_dataset)}")

# Evaluate on both datasets
print("\n" + "="*60)
train_results = evaluate_dataset(model, train_loader, "Train")
test_results = evaluate_dataset(model, test_loader, "Test")

# 1. TRAIN VS TEST COMPARISON
print("\n📊 Generating Train vs Test comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Dice scores comparison
x = np.arange(len(CLASS_NAMES[1:]))
width = 0.35

train_dice = [train_results['mean_dice'][i] for i in range(1, 4)]
test_dice = [test_results['mean_dice'][i] for i in range(1, 4)]

axes[0].bar(x - width/2, train_dice, width, label='Train', color='#4ECDC4', alpha=0.8)
axes[0].bar(x + width/2, test_dice, width, label='Test', color='#FF6B6B', alpha=0.8)
axes[0].set_ylabel('Dice Score', fontsize=12)
axes[0].set_title('Train vs Test Dice Scores', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES[1:])
axes[0].legend()
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (train_val, test_val) in enumerate(zip(train_dice, test_dice)):
    axes[0].text(i - width/2, train_val + 0.02, f'{train_val:.3f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, test_val + 0.02, f'{test_val:.3f}', ha='center', fontsize=9)

# Overall comparison
overall_train = np.mean(train_dice)
overall_test = np.mean(test_dice)

axes[1].bar(['Train', 'Test'], [overall_train, overall_test], 
           color=['#4ECDC4', '#FF6B6B'], alpha=0.8, width=0.5)
axes[1].set_ylabel('Mean Dice Score', fontsize=12)
axes[1].set_title('Overall Performance', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

# Add values
axes[1].text(0, overall_train + 0.02, f'{overall_train:.3f}', ha='center', fontsize=12, fontweight='bold')
axes[1].text(1, overall_test + 0.02, f'{overall_test:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Train vs Test comparison saved")

# 2. CONFUSION MATRIX (IMAGE-LEVEL)
print("\n📊 Generating confusion matrix (image-level)...")
image_true_labels = []
image_pred_labels = []

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Computing predictions"):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(images.size(0)):
            # Get true label from mask
            mask = masks[i]
            unique_true = torch.unique(mask)
            true_label = unique_true[unique_true > 0][0].item() if (unique_true > 0).sum() > 0 else 0
            
            # Get predicted label (most common cancer class)
            pred_mask = preds[i]
            unique, counts = torch.unique(pred_mask, return_counts=True)
            non_bg_mask = unique > 0
            if non_bg_mask.sum() > 0:
                unique_cancer = unique[non_bg_mask]
                counts_cancer = counts[non_bg_mask]
                pred_label = unique_cancer[torch.argmax(counts_cancer)].item()
            else:
                pred_label = 0
            
            if true_label > 0:  # Only count cancer images
                image_true_labels.append(true_label)
                image_pred_labels.append(pred_label)

cm_test = confusion_matrix(image_true_labels, image_pred_labels, labels=[1, 2, 3])
cm_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0],
           xticklabels=CLASS_NAMES[1:], yticklabels=CLASS_NAMES[1:], cbar_kws={'label': 'Images'})
axes[0].set_title('Confusion Matrix - Image Count', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Greens', ax=axes[1],
           xticklabels=CLASS_NAMES[1:], yticklabels=CLASS_NAMES[1:], cbar_kws={'label': 'Percentage'})
axes[1].set_title('Confusion Matrix - Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# 3. DICE DISTRIBUTION COMPARISON
print("\n📊 Generating dice distribution comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, cls in enumerate(range(1, 4)):
    train_data = train_results['dice_scores'][cls]
    test_data = test_results['dice_scores'][cls]
    
    axes[idx].hist(train_data, bins=30, alpha=0.6, label='Train', color='#4ECDC4', edgecolor='black')
    axes[idx].hist(test_data, bins=30, alpha=0.6, label='Test', color='#FF6B6B', edgecolor='black')
    
    axes[idx].axvline(np.mean(train_data), color='#4ECDC4', linestyle='--', linewidth=2, 
                     label=f'Train Mean: {np.mean(train_data):.3f}')
    axes[idx].axvline(np.mean(test_data), color='#FF6B6B', linestyle='--', linewidth=2,
                     label=f'Test Mean: {np.mean(test_data):.3f}')
    
    axes[idx].set_xlabel('Dice Score', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{CLASS_NAMES[cls]} Distribution', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dice_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Dice distribution comparison saved")

# 4. ROC CURVES
print("\n📈 Generating ROC curves...")
all_probs = []
all_targets_roc = []

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Computing ROC"):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_targets_roc.append(masks.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_targets_roc = np.concatenate(all_targets_roc, axis=0)

n_samples = all_probs.shape[0] * all_probs.shape[2] * all_probs.shape[3]
all_probs_flat = all_probs.transpose(0, 2, 3, 1).reshape(n_samples, -1)
all_targets_flat = all_targets_roc.flatten()

fig, ax = plt.subplots(figsize=(10, 8))

for i, class_name in enumerate(CLASS_NAMES):
    y_true = (all_targets_flat == i).astype(int)
    y_score = all_probs_flat[:, i]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ ROC curves saved")

# 5. SAMPLE PREDICTIONS
print("\n🖼️ Generating sample predictions...")
samples = []

with torch.no_grad():
    for images, masks in test_loader:
        if len(samples) >= 6:
            break
        
        images_gpu = images.to(DEVICE)
        outputs = model(images_gpu)
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(min(images.size(0), 6 - len(samples))):
            samples.append({
                'image': images[i].numpy(),
                'mask': masks[i].numpy(),
                'pred': preds[i].cpu().numpy()
            })

fig, axes = plt.subplots(6, 3, figsize=(12, 24))

for idx, sample in enumerate(samples):
    axes[idx, 0].imshow(sample['image'][0], cmap='gray')
    axes[idx, 0].set_title('Input CT Image', fontsize=10, fontweight='bold')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(sample['mask'], cmap='jet', vmin=0, vmax=3)
    axes[idx, 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(sample['pred'], cmap='jet', vmin=0, vmax=3)
    axes[idx, 2].set_title('Model Prediction', fontsize=10, fontweight='bold')
    axes[idx, 2].axis('off')

# Add colorbar legend
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 3)), 
                     cax=cbar_ax)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(CLASS_NAMES)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f'{OUTPUT_DIR}/sample_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Sample predictions saved")

# 6. PERFORMANCE METRICS
print("\n📊 Generating performance metrics...")
# Calculate precision, recall, F1 from confusion matrix
precision = []
recall = []
f1_score = []

for i in range(3):
    tp = cm_test[i, i]
    fp = cm_test[:, i].sum() - tp
    fn = cm_test[i, :].sum() - tp
    
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x = np.arange(len(CLASS_NAMES[1:]))
width = 0.6

axes[0, 0].bar(x, precision, width, color='#FF6B6B', alpha=0.8)
axes[0, 0].set_ylabel('Precision', fontsize=12)
axes[0, 0].set_title('Precision by Class', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(CLASS_NAMES[1:])
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(precision):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

axes[0, 1].bar(x, recall, width, color='#4ECDC4', alpha=0.8)
axes[0, 1].set_ylabel('Recall', fontsize=12)
axes[0, 1].set_title('Recall by Class', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(CLASS_NAMES[1:])
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(recall):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

axes[1, 0].bar(x, f1_score, width, color='#45B7D1', alpha=0.8)
axes[1, 0].set_ylabel('F1 Score', fontsize=12)
axes[1, 0].set_title('F1 Score by Class', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(CLASS_NAMES[1:])
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(f1_score):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# Overall accuracy
accuracy = np.diag(cm_test).sum() / cm_test.sum()
axes[1, 1].bar(['Accuracy'], [accuracy], width=0.5, color='#95E1D3', alpha=0.8)
axes[1, 1].set_ylabel('Score', fontsize=12)
axes[1, 1].set_title('Overall Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].text(0, accuracy + 0.02, f'{accuracy:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Performance metrics saved")

# 7. SUMMARY REPORT
print("\n📋 Generating summary report...")

summary_text = f"""
LUNG CANCER SEGMENTATION - PERFORMANCE SUMMARY
==============================================

DATASET INFORMATION:
- Training Samples: {len(train_dataset)}
- Test Samples: {len(test_dataset)}

TRAINING SET PERFORMANCE:
- ADC Dice: {train_results['mean_dice'][1]:.4f}
- LCC Dice: {train_results['mean_dice'][2]:.4f}
- SCC Dice: {train_results['mean_dice'][3]:.4f}
- Mean Dice: {np.mean([train_results['mean_dice'][i] for i in range(1, 4)]):.4f}

TEST SET PERFORMANCE:
- ADC Dice: {test_results['mean_dice'][1]:.4f}
- LCC Dice: {test_results['mean_dice'][2]:.4f}
- SCC Dice: {test_results['mean_dice'][3]:.4f}
- Mean Dice: {np.mean([test_results['mean_dice'][i] for i in range(1, 4)]):.4f}

GENERALIZATION GAP:
- ADC: {train_results['mean_dice'][1] - test_results['mean_dice'][1]:.4f}
- LCC: {train_results['mean_dice'][2] - test_results['mean_dice'][2]:.4f}
- SCC: {train_results['mean_dice'][3] - test_results['mean_dice'][3]:.4f}

CONFUSION MATRIX INTERPRETATION:
The confusion matrix shows how well the model distinguishes between classes.
- Diagonal values: Correct predictions
- Off-diagonal values: Misclassifications
- Higher diagonal = Better performance

All visualizations saved in '{OUTPUT_DIR}/' directory
"""

with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
    f.write(summary_text)

print(summary_text)
print(f"\n✅ All visualizations completed! Check '{OUTPUT_DIR}/' directory")
print("\nGenerated files:")
print("  1. train_vs_test_comparison.png - Performance comparison")
print("  2. confusion_matrix.png - Image-level confusion matrix")
print("  3. dice_distribution_comparison.png - Score distributions")
print("  4. roc_curves.png - ROC analysis")
print("  5. sample_predictions.png - Visual examples")
print("  6. performance_metrics.png - Precision, Recall, F1, Accuracy")
print("  7. summary_report.txt - Detailed metrics")
