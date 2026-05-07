import torch
import torch.nn as nn
import time, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import DrowsinessDetector
from dataset import get_dataloaders
from sklearn.metrics import f1_score

# ── Config ──────────────────────────────────────────────────
MODEL_PATH  = "weights/best_model.pth"
DATA_DIR    = "data/processed"
SEQ_LEN     = 4
N_WARMUP    = 5    # warmup runs before timing
N_BENCH     = 50   # timed runs for latency measurement
BATCH_SIZE  = 1    # single sample — realistic for real-time use

Path("weights").mkdir(exist_ok=True)


# ── Helper: make a single pseudo-sequence ───────────────────
def make_dummy():
    """Single sample pseudo-sequence (1, T, 3, 224, 224)."""
    return torch.randn(1, SEQ_LEN, 3, 224, 224)


# ── Helper: measure inference latency ───────────────────────
def measure_latency_pytorch(model, dummy, n_warmup, n_bench):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_bench):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def measure_latency_onnx(session, dummy_np, n_warmup, n_bench):
    for _ in range(n_warmup):
        session.run(None, {"input": dummy_np})
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy_np})
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


# ── Helper: evaluate F1 on test set ─────────────────────────
def eval_f1_pytorch(model, loader, seq_len, device, max_batches=50):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches: break
            seq    = imgs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).to(device)
            logits = model(seq)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


def eval_f1_onnx(session, loader, seq_len, max_batches=50):
    all_preds, all_labels = [], []
    for i, (imgs, labels) in enumerate(loader):
        if i >= max_batches: break
        seq = imgs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).numpy()
        out = session.run(None, {"input": seq})[0]
        all_preds.extend(out.argmax(axis=1).tolist())
        all_labels.extend(labels.tolist())
    return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
print("="*55)
print("  Model Optimization — Drowsiness Detection")
print("="*55)

device = torch.device("cpu")   # optimization benchmarked on CPU
dummy  = make_dummy()

# ── Load baseline model ─────────────────────────────────────
print("\n[1/4] Loading baseline model...")
ckpt  = torch.load(MODEL_PATH, map_location=device)
model = DrowsinessDetector(seq_len=SEQ_LEN).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

loaders = get_dataloaders(DATA_DIR, batch_size=1, num_workers=0)

baseline_size    = file_size_mb(MODEL_PATH)
baseline_latency, baseline_std = measure_latency_pytorch(
    model, dummy, N_WARMUP, N_BENCH
)
baseline_f1 = eval_f1_pytorch(model, loaders["test"], SEQ_LEN, device)
print(f"   Size     : {baseline_size:.1f} MB")
print(f"   Latency  : {baseline_latency:.1f} ± {baseline_std:.1f} ms")
print(f"   F1 score : {baseline_f1:.4f}")

# ── Step 1: PyTorch dynamic quantization ───────────────────
print("\n[2/4] Applying PyTorch dynamic quantization...")
model_quant = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},   # quantize Linear and LSTM layers
    dtype=torch.qint8
)
quant_path = "weights/model_quantized.pth"
torch.save(model_quant.state_dict(), quant_path)

quant_size    = file_size_mb(quant_path)
quant_latency, quant_std = measure_latency_pytorch(
    model_quant, dummy, N_WARMUP, N_BENCH
)
quant_f1 = eval_f1_pytorch(model_quant, loaders["test"], SEQ_LEN, device)
print(f"   Size     : {quant_size:.1f} MB")
print(f"   Latency  : {quant_latency:.1f} ± {quant_std:.1f} ms")
print(f"   F1 score : {quant_f1:.4f}")

# ── Step 2: ONNX export ────────────────────────────────────
print("\n[3/4] Exporting to ONNX...")
onnx_path = "weights/model.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names  = ["input"],
    output_names = ["output"],
    opset_version = 18,
    verbose = False
)
print(f"   Saved to: {onnx_path}")

# ── Step 3: ONNX Runtime inference ─────────────────────────
print("\n[4/4] Benchmarking ONNX Runtime...")
import onnxruntime as ort
session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)
dummy_np = dummy.numpy()

onnx_size    = file_size_mb(onnx_path)
onnx_latency, onnx_std = measure_latency_onnx(
    session, dummy_np, N_WARMUP, N_BENCH
)
onnx_f1 = eval_f1_onnx(session, loaders["test"], SEQ_LEN)
print(f"   Size     : {onnx_size:.1f} MB")
print(f"   Latency  : {onnx_latency:.1f} ± {onnx_std:.1f} ms")
print(f"   F1 score : {onnx_f1:.4f}")

# ── Print final comparison table ───────────────────────────
print("\n" + "="*65)
print(f"{'Metric':22} {'Baseline':>12} {'Quantized':>12} {'ONNX':>12}")
print("-"*65)
print(f"{'Model size (MB)':22} {baseline_size:>12.1f} {quant_size:>12.1f} {onnx_size:>12.1f}")
print(f"{'Latency (ms)':22} {baseline_latency:>12.1f} {quant_latency:>12.1f} {onnx_latency:>12.1f}")
print(f"{'F1 score':22} {baseline_f1:>12.4f} {quant_f1:>12.4f} {onnx_f1:>12.4f}")
print(f"{'Speedup vs baseline':22} {'1.00x':>12} {baseline_latency/quant_latency:>11.2f}x {baseline_latency/onnx_latency:>11.2f}x")
print(f"{'Size reduction':22} {'1.00x':>12} {baseline_size/quant_size:>11.2f}x {baseline_size/onnx_size:>11.2f}x")
print("="*65)
print("\nFiles saved:")
print(f"  Quantized model : {quant_path}")
print(f"  ONNX model      : {onnx_path}")
print("\nCopy this table into your project report.")