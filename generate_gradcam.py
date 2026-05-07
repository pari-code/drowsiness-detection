import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from dataset import get_dataloaders
from model import DrowsinessDetector
from gradcam import generate_gradcam_samples

# ── Config ──────────────────────────────────────────────────
MODEL_PATH = "weights/best_model.pth"
DATA_DIR   = "data/processed"
SEQ_LEN    = 4
BATCH_SIZE = 8      # small batch — only need a few samples
N_SAMPLES  = 8      # number of Grad-CAM images to generate

# ── Load model ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt  = torch.load(MODEL_PATH, map_location=device)
model = DrowsinessDetector(seq_len=SEQ_LEN).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded — val_f1={ckpt['val_f1']:.4f}")

# ── Load test dataloader ─────────────────────────────────────
loaders = get_dataloaders(
    data_dir    = DATA_DIR,
    batch_size  = BATCH_SIZE,
    num_workers = 0,
)
test_loader = loaders["test"]
print(f"Test loader ready — {len(test_loader):,} batches")

# ── Generate Grad-CAM images ─────────────────────────────────
generate_gradcam_samples(
    model      = model,
    dataloader = test_loader,
    device     = device,
    output_dir = "outputs/gradcam",
    n_samples  = N_SAMPLES,
)

print("Done. Check outputs/gradcam/ folder.")