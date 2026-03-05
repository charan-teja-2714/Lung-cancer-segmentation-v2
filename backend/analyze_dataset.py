import os
from collections import defaultdict

DATA_ROOT = "data/raw"

def analyze_dataset():
    stats = {
        "train": defaultdict(int),
        "test": defaultdict(int)
    }
    
    for split in ["train", "test"]:
        ct_root = os.path.join(DATA_ROOT, split, "CT")
        
        for cancer_type in ["ADC", "LCC", "SCC"]:
            ct_dir = os.path.join(ct_root, cancer_type)
            
            if os.path.exists(ct_dir):
                files = [f for f in os.listdir(ct_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                stats[split][cancer_type] = len(files)
    
    # Calculate totals
    train_total = sum(stats["train"].values())
    test_total = sum(stats["test"].values())
    total_images = train_total + test_total
    
    # Extract unique patient IDs
    train_patients = set()
    test_patients = set()
    
    for split, patient_set in [("train", train_patients), ("test", test_patients)]:
        ct_root = os.path.join(DATA_ROOT, split, "CT")
        for cancer_type in ["ADC", "LCC", "SCC"]:
            ct_dir = os.path.join(ct_root, cancer_type)
            if os.path.exists(ct_dir):
                for fname in os.listdir(ct_dir):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        # Extract patient ID (e.g., LUNG1-143 from LUNG1-143.nii_slice_116.png)
                        patient_id = fname.split('.nii')[0]
                        patient_set.add(patient_id)
    
    total_patients = len(train_patients | test_patients)
    
    print("=" * 60)
    print("LUNG CANCER SEGMENTATION DATASET STATISTICS")
    print("=" * 60)
    
    print("\nOVERALL STATISTICS")
    print(f"Total CT Images: {total_images}")
    print(f"Total Patients: {total_patients}")
    print(f"Total Slices: {total_images}")
    
    print("\nCANCER TYPE DISTRIBUTION")
    total_adc = stats["train"]["ADC"] + stats["test"]["ADC"]
    total_lcc = stats["train"]["LCC"] + stats["test"]["LCC"]
    total_scc = stats["train"]["SCC"] + stats["test"]["SCC"]
    
    print(f"ADC (Adenocarcinoma): {total_adc} slices ({total_adc/total_images*100:.1f}%)")
    print(f"LCC (Large Cell Carcinoma): {total_lcc} slices ({total_lcc/total_images*100:.1f}%)")
    print(f"SCC (Squamous Cell Carcinoma): {total_scc} slices ({total_scc/total_images*100:.1f}%)")
    
    print("\nTRAIN/TEST SPLIT")
    print(f"Training Set: {train_total} slices ({train_total/total_images*100:.1f}%)")
    print(f"  - ADC: {stats['train']['ADC']}")
    print(f"  - LCC: {stats['train']['LCC']}")
    print(f"  - SCC: {stats['train']['SCC']}")
    print(f"  - Patients: {len(train_patients)}")
    
    print(f"\nTest Set: {test_total} slices ({test_total/total_images*100:.1f}%)")
    print(f"  - ADC: {stats['test']['ADC']}")
    print(f"  - LCC: {stats['test']['LCC']}")
    print(f"  - SCC: {stats['test']['SCC']}")
    print(f"  - Patients: {len(test_patients)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_dataset()
