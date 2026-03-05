import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_ROOT = "data/raw"
CLASS_MAP = {1: "ADC", 2: "LCC", 3: "SCC"}

def analyze_dataset(split="train"):
    """Analyze dataset distribution"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {split.upper()} SET")
    print(f"{'='*60}")
    
    ct_root = os.path.join(DATA_ROOT, split, "CT")
    mask_root = os.path.join(DATA_ROOT, split, "MASK")
    
    # Image-level counts
    image_counts = {cls_name: 0 for cls_name in CLASS_MAP.values()}
    
    # Pixel-level counts
    pixel_counts = {'Background': 0, 'ADC': 0, 'LCC': 0, 'SCC': 0}
    
    # Analyze each class
    for cls_id, cls_name in CLASS_MAP.items():
        ct_dir = os.path.join(ct_root, cls_name)
        mask_dir = os.path.join(mask_root, cls_name)
        
        if not os.path.isdir(ct_dir):
            continue
        
        files = [f for f in os.listdir(ct_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_counts[cls_name] = len(files)
        
        print(f"\n{cls_name}:")
        print(f"  Images: {len(files)}")
        
        # Sample a few masks to count pixels
        sample_size = min(10, len(files))
        for fname in files[:sample_size]:
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Count pixels
                bg_pixels = np.sum(mask == 0)
                cancer_pixels = np.sum(mask > 0)
                
                pixel_counts['Background'] += bg_pixels
                pixel_counts[cls_name] += cancer_pixels
    
    # Print summary
    print(f"\n{'='*60}")
    print("IMAGE-LEVEL DISTRIBUTION:")
    print(f"{'='*60}")
    total_images = sum(image_counts.values())
    for cls_name, count in image_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{cls_name:15s}: {count:5d} images ({percentage:5.1f}%)")
    print(f"{'Total':15s}: {total_images:5d} images")
    
    print(f"\n{'='*60}")
    print("PIXEL-LEVEL DISTRIBUTION (sampled):")
    print(f"{'='*60}")
    total_pixels = sum(pixel_counts.values())
    for cls_name, count in pixel_counts.items():
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"{cls_name:15s}: {count:12d} pixels ({percentage:5.1f}%)")
    print(f"{'Total':15s}: {total_pixels:12d} pixels")
    
    return image_counts, pixel_counts

# Analyze both splits
train_img, train_pix = analyze_dataset("train")
test_img, test_pix = analyze_dataset("test")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Train - Image level
axes[0, 0].bar(train_img.keys(), train_img.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
axes[0, 0].set_title('Training Set - Image Count', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Images', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(train_img.items()):
    axes[0, 0].text(i, v + max(train_img.values())*0.02, str(v), ha='center', fontsize=11, fontweight='bold')

# Test - Image level
axes[0, 1].bar(test_img.keys(), test_img.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
axes[0, 1].set_title('Test Set - Image Count', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Number of Images', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(test_img.items()):
    axes[0, 1].text(i, v + max(test_img.values())*0.02, str(v), ha='center', fontsize=11, fontweight='bold')

# Train - Pixel level
axes[1, 0].bar(train_pix.keys(), train_pix.values(), 
              color=['#95E1D3', '#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
axes[1, 0].set_title('Training Set - Pixel Count (Sampled)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Number of Pixels', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Test - Pixel level
axes[1, 1].bar(test_pix.keys(), test_pix.values(),
              color=['#95E1D3', '#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
axes[1, 1].set_title('Test Set - Pixel Count (Sampled)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Number of Pixels', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/dataset_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Dataset distribution visualization saved to 'visualizations/dataset_distribution.png'")

print(f"\n{'='*60}")
print("EXPLANATION:")
print(f"{'='*60}")
print("""
The confusion matrix shows PIXEL counts, not image counts!

For example, if you have:
- 1000 images of ADC
- Each image is 256x256 = 65,536 pixels
- Total pixels = 1000 × 65,536 = 65,536,000 pixels

That's why confusion matrix values are in millions!

Your dataset has:
- Train: {train_total} images
- Test: {test_total} images

But confusion matrix counts every pixel in every image.
""".format(
    train_total=sum(train_img.values()),
    test_total=sum(test_img.values())
))
