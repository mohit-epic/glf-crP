"""
generate_kaggle_data_csv.py
Scan SEN12MS-CR-Winter dataset and produce a 70/15/15 split.
Writes the following files to /kaggle/working:
  - train.csv
  - val.csv
  - test.csv
  - data.csv  (combined without split id)

Run on Kaggle:
  python generate_kaggle_data_csv.py
"""

import csv
import random
from pathlib import Path
from collections import defaultdict


ROOT = Path("/kaggle/input/sen12ms-cr-winter")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def index_files(nested_parts):
    files = defaultdict(dict)
    base = ROOT
    for p in nested_parts:
        base = base / p
    if not base.exists():
        return files
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        parts = folder_name.split('_')
        if len(parts) < 2:
            continue
        folder_id = parts[-1]
        for tif in sorted(folder.glob('*.tif')):
            fname = tif.name
            patch = fname.split('_')[-1].replace('.tif','')
            rel_folder = str(tif.parent.relative_to(ROOT))
            files[folder_id][patch] = {'name': fname, 'rel_folder': rel_folder}
    return files


def write_rows(path, rows, split_id=None):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for r in rows:
            if split_id is not None:
                # Write with split_id as first column
                writer.writerow([split_id, r['s1_folder'], r['s2_folder'], r['s2_cloudy_folder'], r['s2_filename'], r['s1_filename'], r['s2_cloudy_filename']])
            else:
                # Write without split_id (for combined data.csv with split_id already in r)
                writer.writerow([r['split_id'], r['s1_folder'], r['s2_folder'], r['s2_cloudy_folder'], r['s2_filename'], r['s1_filename'], r['s2_cloudy_filename']])


def main():
    print(f"Scanning dataset at: {ROOT}")

    s1 = index_files(["ROIs2017_winter_s1", "ROIs2017_winter_s1", "ROIs2017_winter_s1"])
    s2 = index_files(["ROIs2017_winter_s2", "ROIs2017_winter_s2", "ROIs2017_winter_s2"])
    s2c = index_files(["ROIs2017_winter_s2_cloudy", "ROIs2017_winter_s2_cloudy", "ROIs2017_winter_s2_cloudy"])

    print(f"Indexed folders -> S1: {len(s1)}, S2: {len(s2)}, S2_cloudy: {len(s2c)}")

    triplets = []
    for fid, patches in s2.items():
        for pid, info in patches.items():
            s1_info = s1.get(fid, {}).get(pid)
            s2c_info = s2c.get(fid, {}).get(pid)
            if s1_info and s2c_info:
                triplets.append({
                    's1_folder': s1_info['rel_folder'],
                    's2_folder': info['rel_folder'],
                    's2_cloudy_folder': s2c_info['rel_folder'],
                    's2_filename': info['name'],
                    's1_filename': s1_info['name'],
                    's2_cloudy_filename': s2c_info['name']
                })

    total = len(triplets)
    print(f"Total matched triplets: {total}")
    if total == 0:
        print("No triplets found - check dataset layout.")
        return

    random.seed(42)
    random.shuffle(triplets)

    train_n = int(0.70 * total)
    val_n = int(0.15 * total)
    test_n = total - train_n - val_n

    train = triplets[:train_n]
    val = triplets[train_n:train_n+val_n]
    test = triplets[train_n+val_n:]

    # Attach split ids for combined CSV
    combined = []
    for r in train:
        rr = r.copy(); rr['split_id'] = 1; combined.append(rr)
    for r in val:
        rr = r.copy(); rr['split_id'] = 2; combined.append(rr)
    for r in test:
        rr = r.copy(); rr['split_id'] = 3; combined.append(rr)

    # Write separate CSVs (each with their split_id) and combined CSV
    write_rows(OUT_DIR / 'train.csv', train, split_id=1)
    write_rows(OUT_DIR / 'val.csv', val, split_id=2)
    write_rows(OUT_DIR / 'test.csv', test, split_id=3)
    write_rows(OUT_DIR / 'data.csv', combined, split_id=None)

    print(f"Wrote train/val/test and combined CSVs to {OUT_DIR}")
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == '__main__':
    main()
