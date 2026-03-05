import os
import cv2
import numpy as np
from collections import defaultdict

DATA_ROOT = "data/raw"
IMAGE_SIZE = 256

CLASS_MAP = {
    "ADC": 1,
    "LCC": 2,
    "SCC": 3
}

def analyze_tumor_statistics():
    stats = {
        "ADC": {"areas": [], "pixel_counts": []},
        "LCC": {"areas": [], "pixel_counts": []},
        "SCC": {"areas": [], "pixel_counts": []},
        "ALL": {"areas": [], "pixel_counts": []}
    }
    
    total_pixels = 0
    tumor_pixels = 0
    background_pixels = 0
    
    print("=" * 70)
    print("TUMOR REGION SIZE STATISTICS")
    print("=" * 70)
    print("\nAnalyzing masks...")
    
    # Analyze both train and test sets
    for split in ["train", "test"]:
        mask_root = os.path.join(DATA_ROOT, split, "MASK")
        
        for cancer_type in CLASS_MAP.keys():
            mask_dir = os.path.join(mask_root, cancer_type)
            
            if not os.path.exists(mask_dir):
                continue
            
            for fname in os.listdir(mask_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    mask_path = os.path.join(mask_dir, fname)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if mask is None:
                        continue
                    
                    # Resize to standard size
                    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), 
                                     interpolation=cv2.INTER_NEAREST)
                    
                    # Count pixels
                    tumor_pixel_count = np.sum(mask > 0)
                    total_pixel_count = mask.size
                    
                    # Calculate percentage
                    tumor_percentage = (tumor_pixel_count / total_pixel_count) * 100
                    
                    # Store statistics
                    if tumor_pixel_count > 0:
                        stats[cancer_type]["areas"].append(tumor_percentage)
                        stats[cancer_type]["pixel_counts"].append(tumor_pixel_count)
                        stats["ALL"]["areas"].append(tumor_percentage)
                        stats["ALL"]["pixel_counts"].append(tumor_pixel_count)
                    
                    # Global counts
                    total_pixels += total_pixel_count
                    tumor_pixels += tumor_pixel_count
                    background_pixels += (total_pixel_count - tumor_pixel_count)
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("PER-CLASS TUMOR REGION STATISTICS")
    print("=" * 70)
    
    for cancer_type in ["ADC", "LCC", "SCC"]:
        if len(stats[cancer_type]["areas"]) > 0:
            areas = np.array(stats[cancer_type]["areas"])
            pixel_counts = np.array(stats[cancer_type]["pixel_counts"])
            
            print(f"\n{cancer_type} (Samples: {len(areas)}):")
            print(f"  Average Tumor Area:    {np.mean(areas):.2f}% ± {np.std(areas):.2f}%")
            print(f"  Median Tumor Area:     {np.median(areas):.2f}%")
            print(f"  Minimum Tumor Area:    {np.min(areas):.2f}% ({np.min(pixel_counts)} pixels)")
            print(f"  Maximum Tumor Area:    {np.max(areas):.2f}% ({np.max(pixel_counts)} pixels)")
            print(f"  25th Percentile:       {np.percentile(areas, 25):.2f}%")
            print(f"  75th Percentile:       {np.percentile(areas, 75):.2f}%")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL TUMOR STATISTICS (ALL CLASSES)")
    print("=" * 70)
    
    all_areas = np.array(stats["ALL"]["areas"])
    all_pixel_counts = np.array(stats["ALL"]["pixel_counts"])
    
    print(f"\nTotal Samples Analyzed: {len(all_areas)}")
    print(f"Average Tumor Area:     {np.mean(all_areas):.2f}% ± {np.std(all_areas):.2f}%")
    print(f"Median Tumor Area:      {np.median(all_areas):.2f}%")
    print(f"Minimum Tumor Area:     {np.min(all_areas):.2f}% ({np.min(all_pixel_counts)} pixels)")
    print(f"Maximum Tumor Area:     {np.max(all_areas):.2f}% ({np.max(all_pixel_counts)} pixels)")
    
    # Class imbalance
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)
    
    total_pixels_analyzed = total_pixels
    tumor_percentage = (tumor_pixels / total_pixels_analyzed) * 100
    background_percentage = (background_pixels / total_pixels_analyzed) * 100
    imbalance_ratio = background_pixels / tumor_pixels if tumor_pixels > 0 else 0
    
    print(f"\nTotal Pixels Analyzed:  {total_pixels_analyzed:,}")
    print(f"Tumor Pixels:           {tumor_pixels:,} ({tumor_percentage:.2f}%)")
    print(f"Background Pixels:      {background_pixels:,} ({background_percentage:.2f}%)")
    print(f"Imbalance Ratio:        {imbalance_ratio:.2f}:1 (Background:Tumor)")
    
    # Size categories
    print("\n" + "=" * 70)
    print("TUMOR SIZE DISTRIBUTION")
    print("=" * 70)
    
    small = np.sum(all_areas < 1.0)
    medium = np.sum((all_areas >= 1.0) & (all_areas < 5.0))
    large = np.sum(all_areas >= 5.0)
    
    print(f"\nSmall Tumors (<1% of slice):    {small} ({small/len(all_areas)*100:.1f}%)")
    print(f"Medium Tumors (1-5% of slice):  {medium} ({medium/len(all_areas)*100:.1f}%)")
    print(f"Large Tumors (>5% of slice):    {large} ({large/len(all_areas)*100:.1f}%)")
    
    # Additional statistics
    print("\n" + "=" * 70)
    print("ADDITIONAL STATISTICS")
    print("=" * 70)
    
    print(f"\nImage Dimensions:       {IMAGE_SIZE} x {IMAGE_SIZE} pixels")
    print(f"Pixels per Image:       {IMAGE_SIZE * IMAGE_SIZE:,}")
    print(f"Average Tumor Pixels:   {np.mean(all_pixel_counts):.0f} ± {np.std(all_pixel_counts):.0f}")
    print(f"Median Tumor Pixels:    {np.median(all_pixel_counts):.0f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_tumor_statistics()
