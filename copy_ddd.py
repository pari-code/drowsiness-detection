import shutil, random
from pathlib import Path
from tqdm import tqdm

# ── CHANGE THIS to your DDD source folder ───────────────────
DDD_SOURCE = Path(r"C:\Users\mitta\Downloads\archive (1)\Driver Drowsiness Dataset (DDD)")

# ── Destination (your project's processed folder) ───────────
DEST  = Path("data/processed")
SEED  = 42
SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}

random.seed(SEED)

# DDD folder name → project label
LABEL_MAP = {
    "Non Drowsy": "open",    # alert   → label 0
    "Drowsy":     "closed"   # drowsy  → label 1
}

total_copied = 0

for ddd_folder, proj_label in LABEL_MAP.items():

    src_dir = DDD_SOURCE / ddd_folder
    if not src_dir.exists():
        print(f"WARNING: {src_dir} not found — check DDD_SOURCE path")
        continue

    # Collect all image files
    all_files = [
        f for f in src_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    random.shuffle(all_files)

    n       = len(all_files)
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])

    splits = {
        "train": all_files[:n_train],
        "val":   all_files[n_train : n_train + n_val],
        "test":  all_files[n_train + n_val :]
    }

    print(f"\n{ddd_folder} → {proj_label}  (total: {n:,})")
    print(f"  train: {len(splits['train']):,}  val: {len(splits['val']):,}  test: {len(splits['test']):,}")

    for split_name, files in splits.items():
        dest_dir = DEST / split_name / proj_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_file in tqdm(files, desc=f"  copying {split_name}/{proj_label}", leave=False):
            dest_file = dest_dir / ("ddd_" + src_file.name)
            shutil.copy2(src_file, dest_file)
            total_copied += 1

print(f"\nDone! Copied {total_copied:,} DDD images.")

# ── Final combined count (MRL + DDD together) ───────────────
print("\n── Combined totals (all datasets so far) ──")
for split in ["train", "val", "test"]:
    for label in ["open", "closed"]:
        d = DEST / split / label
        n = len([f for f in d.iterdir() if f.is_file()])
        print(f"  {split:5s} / {label:6s} : {n:,}")