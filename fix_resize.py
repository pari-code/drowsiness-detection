import cv2
from pathlib import Path
from tqdm import tqdm

PROCESSED  = Path("data/processed")
TARGET     = (224, 224)
EXTS       = {".jpg", ".jpeg", ".png"}

# Collect every image file
all_files = [
    f for f in PROCESSED.rglob("*")
    if f.suffix.lower() in EXTS
]
print(f"Found {len(all_files):,} images to process")

already_ok = 0
resized    = 0
corrupt    = 0

for fpath in tqdm(all_files, desc="Resizing"):
    img = cv2.imread(str(fpath))

    if img is None:
        corrupt += 1
        fpath.unlink(missing_ok=True)   # delete unreadable file
        continue

    h, w = img.shape[:2]

    if (h, w) == TARGET:
        already_ok += 1
        continue

    # Resize using INTER_AREA (best quality when shrinking)
    resized_img = cv2.resize(img, TARGET, interpolation=cv2.INTER_AREA)

    # Overwrite the original file
    cv2.imwrite(str(fpath), resized_img)
    resized += 1

print(f"\nDone!")
print(f"  Already 224x224 : {already_ok:,}")
print(f"  Resized         : {resized:,}")
print(f"  Corrupt deleted : {corrupt}")