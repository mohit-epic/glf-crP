"""
Generate data.csv for Kaggle SEN12MS-CR-Winter dataset.
Handles the double-nested folder structure and maps S1, S2, S2_cloudy files.
Run this on Kaggle to create data.csv in /kaggle/working/
"""
import os
import csv
from pathlib import Path
from collections import defaultdict

# Base path (adjust if running locally)
root = Path("/kaggle/input/sen12ms-cr-winter")
out_csv = Path("/kaggle/working/data.csv")

print(f"Scanning dataset from: {root}")
print(f"Will write CSV to: {out_csv}")

# Find all S1, S2, and S2_cloudy triplets
# Expected structure:
# /kaggle/input/sen12ms-cr-winter/
#   ROIs2017_winter_s1/ROIs2017_winter_s1/s1_XXX/ROIs2017_winter_s1_XXX_pYYY.tif
#   ROIs2017_winter_s2/ROIs2017_winter_s2/s2_XXX/ROIs2017_winter_s2_XXX_pYYY.tif
#   ROIs2017_winter_s2_cloudy/ROIs2017_winter_s2_cloudy/s2_cloudy_XXX/ROIs2017_winter_s2_cloudy_XXX_pYYY.tif

# Index files by (folder_id, patch_id) for easy matching
s1_files = defaultdict(dict)
s2_files = defaultdict(dict)
s2_cloudy_files = defaultdict(dict)

print("\nIndexing S1 files...")
s1_base = root / "ROIs2017_winter_s1" / "ROIs2017_winter_s1" / "ROIs2017_winter_s1"
if s1_base.exists():
    for folder in sorted(s1_base.iterdir()):
        if folder.is_dir() and folder.name.startswith("s1_"):
            folder_id = folder.name.replace("s1_", "")
            for tif_file in folder.glob("*.tif"):
                # Extract patch_id from filename, e.g., "ROIs2017_winter_s1_102_p614.tif" -> "p614"
                fname = tif_file.name
                patch_id = fname.split("_")[-1].replace(".tif", "")  # e.g., "p614.tif" -> "p614"
                rel_path = tif_file.relative_to(root)
                s1_files[folder_id][patch_id] = {
                    'name': fname,
                    'rel_folder': str(rel_path.parent.relative_to(root)),
                    'full_path': tif_file
                }
    print(f"  Found {sum(len(v) for v in s1_files.values())} S1 files")

print("Indexing S2 (cloud-free) files...")
s2_base = root / "ROIs2017_winter_s2" / "ROIs2017_winter_s2" / "ROIs2017_winter_s2"
if s2_base.exists():
    for folder in sorted(s2_base.iterdir()):
        if folder.is_dir() and folder.name.startswith("s2_"):
            folder_id = folder.name.replace("s2_", "")
            for tif_file in folder.glob("*.tif"):
                fname = tif_file.name
                patch_id = fname.split("_")[-1].replace(".tif", "")
                rel_path = tif_file.relative_to(root)
                s2_files[folder_id][patch_id] = {
                    'name': fname,
                    'rel_folder': str(rel_path.parent.relative_to(root)),
                    'full_path': tif_file
                }
    print(f"  Found {sum(len(v) for v in s2_files.values())} S2 files")

print("Indexing S2_cloudy files...")
s2_cloudy_base = root / "ROIs2017_winter_s2_cloudy" / "ROIs2017_winter_s2_cloudy" / "ROIs2017_winter_s2_cloudy"
if s2_cloudy_base.exists():
    for folder in sorted(s2_cloudy_base.iterdir()):
        if folder.is_dir() and folder.name.startswith("s2_cloudy_"):
            folder_id = folder.name.replace("s2_cloudy_", "")
            for tif_file in folder.glob("*.tif"):
                fname = tif_file.name
                patch_id = fname.split("_")[-1].replace(".tif", "")
                rel_path = tif_file.relative_to(root)
                s2_cloudy_files[folder_id][patch_id] = {
                    'name': fname,
                    'rel_folder': str(rel_path.parent.relative_to(root)),
                    'full_path': tif_file
                }
    print(f"  Found {sum(len(v) for v in s2_cloudy_files.values())} S2_cloudy files")

# Match triplets
print("\nMatching triplets...")
entries = []
matched = 0
skipped = 0

for folder_id in sorted(s2_files.keys()):
    for patch_id in sorted(s2_files[folder_id].keys()):
        if folder_id in s1_files and patch_id in s1_files[folder_id]:
            if folder_id in s2_cloudy_files and patch_id in s2_cloudy_files[folder_id]:
                s1_info = s1_files[folder_id][patch_id]
                s2_info = s2_files[folder_id][patch_id]
                s2_cloudy_info = s2_cloudy_files[folder_id][patch_id]
                
                # CSV format: [split_id, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
                entries.append([
                    '3',  # split_id = 3 (test)
                    s1_info['rel_folder'],
                    s2_info['rel_folder'],
                    s2_cloudy_info['rel_folder'],
                    s2_info['name'],
                    s1_info['name'],
                    s2_cloudy_info['name']
                ])
                matched += 1
            else:
                skipped += 1
        else:
            skipped += 1

print(f"  Matched: {matched}")
print(f"  Skipped: {skipped}")

# Write CSV
if entries:
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(entries)
    print(f"\n✓ CSV written to {out_csv}")
    print(f"  Total entries: {len(entries)}")
    print("\nFirst 3 entries:")
    for i, entry in enumerate(entries[:3]):
        print(f"  {i+1}. {entry}")
else:
    print("\n✗ No triplets found! Check folder structure and naming patterns.")

# Verify paths (sample check)
print("\nVerifying sample paths...")
if entries:
    sample = entries[0]
    s1_check = root / sample[1] / sample[5]
    s2_check = root / sample[2] / sample[4]
    s2c_check = root / sample[3] / sample[6]
    
    print(f"\nSample entry: {sample[4]}")
    print(f"  S1 path exists: {s1_check.exists()} ({s1_check})")
    print(f"  S2 path exists: {s2_check.exists()} ({s2_check})")
    print(f"  S2_cloudy path exists: {s2c_check.exists()} ({s2c_check})")
