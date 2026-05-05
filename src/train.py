import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import sys

sys.path.append("src")
from model import DrowsinessDetector
from dataset import get_dataloaders

CFG = {
    "data_dir":      "/content/data/processed",
    "weights_dir":   "weights",
    "batch_size":    64,
    "num_workers":   2,
    "seq_len":       4,
    "epochs":        15,
    "lr_head":       1e-3,
    "lr_backbone":   1e-4,
    "lr_finetune":   5e-5,
    "weight_decay":  1e-4,
    "patience":      5,
    "phase2_epoch":  5,
    "phase3_epoch":  999,
}


def make_sequence(imgs, seq_len):
    return imgs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)


def train_one_epoch(model, loader, optimizer, criterion, device, seq_len):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs   = make_sequence(imgs, seq_len).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device, seq_len):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  eval", leave=False):
            imgs   = make_sequence(imgs, seq_len).to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, f1, all_preds, all_labels


def get_optimizer(model, phase, cfg):
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=cfg["lr_head"],
                           weight_decay=cfg["weight_decay"])
    elif phase == 2:
        cnn_params  = [p for p in model.cnn.parameters() if p.requires_grad]
        head_params = (list(model.se.parameters()) +
                       list(model.lstm.parameters()) +
                       list(model.classifier.parameters()))
        return optim.AdamW([
            {"params": cnn_params,  "lr": cfg["lr_backbone"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ], weight_decay=cfg["weight_decay"])
    else:
        return optim.AdamW(model.parameters(), lr=cfg["lr_finetune"],
                           weight_decay=cfg["weight_decay"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    Path(CFG["weights_dir"]).mkdir(exist_ok=True)

    wandb.init(project="drowsiness-detection", config=CFG,
               name="efficientnet-bilstm-se", resume="allow")

    loaders = get_dataloaders(
        data_dir    = CFG["data_dir"],
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
    )

    model = DrowsinessDetector(seq_len=CFG["seq_len"]).to(device)
    model.count_parameters()

    criterion = nn.CrossEntropyLoss()

    RESUME_FROM   = "weights/best_model.pth"
    start_epoch   = 1
    best_val_f1   = 0.0
    current_phase = 1

    if Path(RESUME_FROM).exists():
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["val_f1"]
        print(f"Resumed from epoch {ckpt['epoch']} — val_f1={best_val_f1:.4f}")
        if start_epoch > CFG["phase2_epoch"]:
            model.unfreeze_cnn(blocks=2)
            current_phase = 2
        else:
            current_phase = 1
        print(f"Restored to phase {current_phase}")
    else:
        print("No checkpoint — starting from scratch")

    optimizer = get_optimizer(model, current_phase, CFG)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"])

    patience_count = 0
    best_ckpt      = Path(CFG["weights_dir"]) / "best_model.pth"

    for epoch in range(start_epoch, CFG["epochs"] + 1):

        if epoch == CFG["phase2_epoch"] + 1 and current_phase == 1:
            print(f"-> Epoch {epoch}: Phase 2 — unfreezing last 2 CNN blocks")
            model.unfreeze_cnn(blocks=2)
            current_phase = 2
            optimizer = get_optimizer(model, current_phase, CFG)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["epochs"] - epoch)

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, CFG["seq_len"])

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, loaders["val"], criterion,
            device, CFG["seq_len"])

        scheduler.step()

        wandb.log({
            "epoch": epoch, "phase": current_phase,
            "tr_loss": tr_loss,  "val_loss": val_loss,
            "tr_acc":  tr_acc,   "val_acc":  val_acc,
            "tr_f1":   tr_f1,    "val_f1":   val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Ep {epoch:02d}/{CFG['epochs']} | Ph{current_phase} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
              f"val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "val_acc":     val_acc,
                "cfg":         CFG,
            }, best_ckpt)
            print(f"  -> New best saved — val_f1={val_f1:.4f}")
            patience_count = 0
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{CFG['patience']})")
            if patience_count >= CFG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.finish()
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
