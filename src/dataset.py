import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ── Label mapping ───────────────────────────────────────────
# open   = 0  (awake / alert)
# closed = 1  (drowsy)
# This must stay consistent across dataset.py, train.py,
# evaluate.py, and realtime.py
CLASS_MAP = {"open": 0, "closed": 1}

# ImageNet stats — used because EfficientNet-B0 was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the correct transform pipeline for each split.
    CRITICAL: val and test must NEVER have random augmentations.
    Only train gets augmentation — augmenting val/test inflates
    reported accuracy because you test on modified images.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
    else:
        # val and test — deterministic, no randomness
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])


class DrowsinessDataset(Dataset):
    """
    Loads images from:
        root_dir/open/   → label 0 (alert)
        root_dir/closed/ → label 1 (drowsy)

    Works with your MRL + DDD combined structure.
    Automatically handles mrl_ and ddd_ prefixed filenames.
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = get_transforms(split)
        self.samples   = []   # list of (Path, int) tuples

        for label_name, label_idx in CLASS_MAP.items():
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}\n"
                    f"Make sure data/processed/{split}/{label_name}/ exists."
                )
            for fpath in class_dir.iterdir():
                if fpath.suffix.lower() in self.VALID_EXTS:
                    self.samples.append((fpath, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                f"Check your folder structure."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            # Corrupt file slipped through — return a black image
            # so training doesn't crash on a bad file
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(img), torch.tensor(label, dtype=torch.long)

    def get_class_counts(self):
        """Returns {label_name: count} — useful for verifying balance."""
        counts = {name: 0 for name in CLASS_MAP}
        idx_to_name = {v: k for k, v in CLASS_MAP.items()}
        for _, label_idx in self.samples:
            counts[idx_to_name[label_idx]] += 1
        return counts


def get_dataloaders(
    data_dir:   str   = "data/processed",
    batch_size: int   = 32,
    num_workers: int  = 0,
) -> dict:
    """
    Returns dict with 'train', 'val', 'test' DataLoaders.

    num_workers guide:
      0  → safest on Windows (no multiprocessing bugs)
      2  → good for Linux/Mac with SSD
      4  → good for Colab or Linux with fast disk
    Start with 0. If training feels slow increase to 2.

    pin_memory:
      True  → faster GPU training (data stays in pinned RAM)
      False → safer for CPU-only training (Intel Unnati lab)
    """
    import torch
    use_gpu    = torch.cuda.is_available()
    pin_memory = use_gpu   # auto: True for GPU, False for CPU

    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = f"{data_dir}/{split}"
        ds = DrowsinessDataset(root_dir=split_dir, split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),  # ONLY shuffle train
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),  # drop incomplete last batch
        )
        counts = ds.get_class_counts()
        print(f"  {split:5s}: {len(ds):,} images | "
              f"open={counts['open']:,} closed={counts['closed']:,}")

    return loaders
