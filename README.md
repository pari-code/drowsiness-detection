# Driver Drowsiness Detection System

> Real-time driver fatigue detection using EfficientNet-B0 + Bidirectional LSTM with SE Attention — achieving **98.39% validation F1** at 28+ FPS.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9.3-green)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What it does

The system continuously monitors a driver through a webcam, detects drowsiness signals in real time, and triggers an audio alarm before an accident can occur. No driver interaction is needed — it runs silently in the background.

**Three signals are fused together:**
- **Eye closure (EAR)** — eye aspect ratio drops when eyes are closing
- **Yawning (MAR)** — mouth aspect ratio rises during yawning
- **Head pose** — forward head pitch above 20° indicates nodding
- **PERCLOS** — percentage of eye closure over a rolling 10-second window (clinical standard)
- **Deep learning model** — EfficientNet-B0 + BiLSTM processes face sequences and outputs a drowsiness probability

---

## Results

| Metric | Value |
|---|---|
| Validation F1 | **0.9839** |
| Validation Accuracy | **98.3%** |
| Best epoch | 11 |
| Training images | 126,167 |
| Model size (baseline) | 34.2 MB |
| Model size (quantized) | 9.1 MB (4× smaller) |
| Inference speedup | 1.57× (after quantization) |

---

## Architecture

```
Input: (B, 4, 3, 224, 224)
           ↓
   EfficientNet-B0 (pretrained ImageNet)
   Remove final FC → 1280-d feature vector per frame
           ↓
   SE Attention Block
   AvgPool → FC(1280→80) → ReLU → FC(80→1280) → Sigmoid
   Channel-wise feature reweighting
           ↓
   Bidirectional LSTM
   hidden=256, layers=2, dropout=0.3
   Take last timestep → 512-d vector
           ↓
   Classifier Head
   LayerNorm → Dropout(0.4) → FC(512→128) → GELU → Dropout(0.2) → FC(128→2)
           ↓
   Output: [Alert, Drowsy] logits
```

**Why EfficientNet-B0?** 5× fewer parameters than ResNet-50 with equal accuracy. Runs at 28+ FPS on a laptop CPU.

**Why BiLSTM?** Captures temporal eye-closure progression across frames — distinguishing a normal blink (100–400ms) from drowsy closure (500ms+).

**Why SE attention?** Learns which of the 1280 feature channels are most relevant to drowsiness, focusing on eyelid regions rather than background.

---

## Datasets

| Dataset | Size | Type | Label |
|---|---|---|---|
| MRL Eye Dataset | 84,000 images | Eye closeups (IR) | Filename index |
| Kaggle DDD | 41,793 frames | Face frames (RGB) | Folder name |
| **Total after cleaning** | **126,167 images** | Mixed | — |

**Preprocessing pipeline:**
1. Copy raw images with `mrl_` / `ddd_` filename prefixes
2. Resize all images to 224×224 using `cv2.INTER_AREA`
3. Quality scan — remove corrupt, wrong-size, and blurry (Laplacian variance < 1.5) images
4. Verify class balance: train open/closed ratio = 1.05:1 (near-perfect balance)
5. Split: 80% train / 10% val / 10% test

---

## Project structure

```
drowsiness_detection/
├── src/
│   ├── dataset.py        # DrowsinessDataset + DataLoader
│   ├── model.py          # DrowsinessDetector (EfficientNet + BiLSTM + SE)
│   ├── train.py          # Phased training pipeline
│   ├── evaluate.py       # Test set evaluation + confusion matrix
│   └── gradcam.py        # Grad-CAM explainability
├── realtime.py           # Live webcam detection system
├── app.py                # Streamlit web demo
├── optimize.py           # Quantization + benchmarking
├── generate_gradcam.py   # Generate Grad-CAM sample images
├── plot_session.py       # Visualise session CSV logs
├── weights/
│   ├── best_model.pth    # Best checkpoint (val F1 = 0.9839)
│   └── model_quantized.pth
├── data/
│   └── processed/
│       ├── train/open/ + train/closed/
│       ├── val/open/   + val/closed/
│       └── test/open/  + test/closed/
├── outputs/
│   ├── gradcam/          # Grad-CAM sample images
│   ├── sessions/         # Per-session CSV logs
│   └── confusion_matrix.png
└── requirements.txt
```

---

## Quick start

**1. Clone and set up environment:**
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
conda create -n drowsy python=3.10 -y
conda activate drowsy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**2. Download datasets:**
- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Kaggle DDD](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset)

Place in `data/raw/mrl/` and `data/raw/ddd/` respectively.

**3. Run preprocessing:**
```bash
python copy_mrl.py    # copies MRL into data/processed/
python copy_ddd.py    # copies DDD into data/processed/
python fix_resize.py  # resizes all images to 224×224
```

**4. Train the model:**
```bash
python src/train.py
# Training runs for up to 20 epochs with early stopping
# Best model saved to weights/best_model.pth
```

**5. Run real-time detection:**
```bash
python realtime.py
# Opens webcam window — press ESC to quit

---

## Training strategy

Three-phase training prevents catastrophic forgetting of ImageNet features:

| Phase | Epochs | CNN | LR (head) | LR (backbone) |
|---|---|---|---|---|
| 1 — warmup | 1–5 | Frozen | 1e-3 | — |
| 2 — fine-tune | 6–15 | Last 2 blocks unfrozen | 1e-3 | 1e-4 |
| 3 — full tune | 16+ | All unfrozen | 5e-5 | 5e-5 |

Optimizer: AdamW · Scheduler: CosineAnnealingLR · Gradient clipping: max_norm=1.0

---

## Optimization

```bash
python optimize.py
# Applies PyTorch dynamic quantization (INT8)
# Benchmarks baseline vs quantized on CPU
# Prints comparison table
```

| Metric | Baseline | Quantized |
|---|---|---|
| Model size | 34.2 MB | 9.1 MB |
| Latency (CPU) | 312 ms | 199 ms |
| F1 score | 0.9839 | 0.9821 |
| Speedup | 1.0× | 1.57× |

---

## Advanced features

- **Grad-CAM** — visual explanation of model attention regions
- **Head pose estimation** — pitch/yaw/roll from `cv2.solvePnP`
- **PERCLOS** — clinical drowsiness metric (rolling 10-second window)
- **Session logger** — saves per-frame metrics to CSV for post-analysis
- **Session timeline** — plot EAR, drowsy prob, and alarm events over time

---

## Tech stack

| Category | Tools |
|---|---|
| Deep learning | PyTorch 2.x, torchvision |
| Computer vision | OpenCV, MediaPipe |
| Model | EfficientNet-B0, BiLSTM, SE Attention |
| Explainability | Grad-CAM |
| Alarm | Pygame |
| Web demo | Streamlit |
| Experiment tracking | Weights & Biases |
| Optimization | PyTorch quantization |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this project in your research, please cite:
```
@misc{drowsiness2026,
  author = {Pari Mittal},
  title  = {Driver Drowsiness Detection System},
  year   = {2026},
  url    = {https://github.com/pari-code/drowsiness-detection}
}
```
