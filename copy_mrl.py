import shutil, os
from pathlib import Path
from tqdm import tqdm

# ── CHANGE THIS to where you extracted the MRL archive ──────
MRL_SOURCE = Path(r"C:\Users\mitta\Downloads\archive\data")

# ── This stays as-is (your project's processed folder) ──────
DEST = Path("data/processed")

# awake → open,  sleepy → closed
LABEL_MAP = {
    "awake":  "open",
    "sleepy": "closed"
}

total_copied = 0

for split in ["train", "val", "test"]:
    for mrl_label, proj_label in LABEL_MAP.items():

        src_dir  = MRL_SOURCE / split / mrl_label
        dest_dir = DEST / split / proj_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found — check your MRL_SOURCE path")
            continue

        files = list(src_dir.iterdir())
        print(f"Copying {split}/{mrl_label} → {split}/{proj_label}  ({len(files):,} files)")

        for src_file in tqdm(files, desc=f"  {split}/{proj_label}", leave=False):
            if src_file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                # Prefix with "mrl_" to avoid filename clashes with other datasets
                dest_file = dest_dir / ("mrl_" + src_file.name)
                shutil.copy2(src_file, dest_file)
                total_copied += 1

print(f"\nDone! Copied {total_copied:,} MRL images total.")

# Verify final counts
print("\n── Final counts ──────────────────────────")
for split in ["train", "val", "test"]:
    for label in ["open", "closed"]:
        d = DEST / split / label
        n = len([f for f in d.iterdir() if f.is_file()])
        print(f"  {split:5s}/{label:6s}: {n:,}")