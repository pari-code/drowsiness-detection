import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append("src")
from model import DrowsinessDetector
from dataset import get_dataloaders
from train import make_sequence


def evaluate_model(
    checkpoint_path: str = "weights/best_model.pth",
    data_dir:        str = "/content/processed",
    output_dir:      str = "outputs",
    seq_len:         int = 16,
    batch_size:      int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(exist_ok=True)

    # ── Load checkpoint ─────────────────────────────────────────
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = DrowsinessDetector(seq_len=seq_len).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_f1={ckpt['val_f1']:.4f})")

    # ── Test loader ─────────────────────────────────────────────
    loaders = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    test_loader = loaders["test"]

    # ── Run inference on full test set ──────────────────────────
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating test set"):
            imgs   = make_sequence(imgs, seq_len).to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:,1].cpu().tolist())  # drowsy prob

    # ── Metrics ─────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    print(f"\n── Test set results ──────────────────────")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 score : {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['alert','drowsy'])}")

    # ── Confusion matrix plot ───────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["alert", "drowsy"],
        yticklabels=["alert", "drowsy"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — test set\nAcc={acc:.3f}  F1={f1:.3f}")
    plt.tight_layout()
    out_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")
    return acc, f1


if __name__ == "__main__":
    evaluate_model()