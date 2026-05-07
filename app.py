import streamlit as st
import torch
import numpy as np
import cv2
import sys, os
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import DrowsinessDetector
from gradcam import GradCAM

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title = "Driver Drowsiness Detector",
    page_icon  = "🚗",
    layout     = "wide"
)

# ── Constants ───────────────────────────────────────────────
MODEL_PATH = "weights/best_model.pth"
SEQ_LEN    = 4
TRANSFORM  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])


# ── Load model (cached so it only loads once) ───────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model  = DrowsinessDetector(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, device


def predict_image(model, img_pil, device):
    """Run inference on a single PIL image. Returns (label, prob)."""
    img_pil = img_pil.convert("RGB")
    t = TRANSFORM(img_pil)
    seq = t.unsqueeze(0).unsqueeze(0).repeat(1, SEQ_LEN, 1, 1, 1).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(seq), dim=1)[0]
    drowsy_prob = probs[1].item()
    label = "Drowsy" if drowsy_prob > 0.5 else "Alert"
    return label, drowsy_prob


def get_gradcam(model, img_pil, device):
    """Generate Grad-CAM heatmap for an image."""
    gcam = GradCAM(model)
    t    = TRANSFORM(img_pil)
    seq  = t.unsqueeze(0).unsqueeze(0).repeat(1, SEQ_LEN, 1, 1, 1).to(device)
    seq.requires_grad = True
    heatmap = gcam.generate(seq, class_idx=1)

    # Convert PIL to BGR for overlay
    face_bgr = cv2.cvtColor(np.array(img_pil.resize((224, 224))),
                             cv2.COLOR_RGB2BGR)
    overlay  = gcam.overlay(face_bgr, heatmap)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.title("Driver Drowsiness Detector")
st.sidebar.markdown("Deep learning system to detect driver fatigue.")
page = st.sidebar.radio(
    "Select mode",
    ["Image analysis", "Video analysis", "Model info"]
)

model, ckpt, device = load_model()
st.sidebar.success(f"Model loaded — val F1: {ckpt['val_f1']:.4f}")


# ════════════════════════════════════════════════════════════
# PAGE 1 — IMAGE ANALYSIS
# ════════════════════════════════════════════════════════════
if page == "Image analysis":
    st.title("Image analysis")
    st.markdown("Upload a face image to get a drowsiness prediction and Grad-CAM heatmap.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input image")
            st.image(img_pil, use_column_width=True)

        with st.spinner("Running inference..."):
            label, drowsy_prob = predict_image(model, img_pil, device)
            overlay            = get_gradcam(model, img_pil, device)

        with col2:
            st.subheader("Grad-CAM heatmap")
            st.image(overlay, use_column_width=True)
            st.caption("Red/yellow = regions the model focused on")

        with col3:
            st.subheader("Prediction")
            alert_color = "🔴" if label == "Drowsy" else "🟢"
            st.metric(
                label = "Status",
                value = f"{alert_color} {label}"
            )
            st.metric(
                label = "Drowsy confidence",
                value = f"{drowsy_prob*100:.1f}%"
            )
            st.metric(
                label = "Alert confidence",
                value = f"{(1-drowsy_prob)*100:.1f}%"
            )
            st.progress(drowsy_prob)

            if label == "Drowsy":
                st.error("WARNING: Drowsiness detected! Please take a break.")
            else:
                st.success("Driver appears alert.")


# ════════════════════════════════════════════════════════════
# PAGE 2 — VIDEO ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Video analysis":
    st.title("Video analysis")
    st.markdown("Upload a short video clip to analyse drowsiness frame by frame.")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )
    max_frames = st.slider("Max frames to analyse", 10, 200, 60)

    if uploaded_video is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(uploaded_video.read())
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        probs, frame_nums = [], []
        frame_idx = 0
        progress_bar = st.progress(0)

        with st.spinner(f"Analysing up to {max_frames} frames..."):
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret: break
                img_pil = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                _, prob = predict_image(model, img_pil, device)
                probs.append(prob)
                frame_nums.append(frame_idx)
                frame_idx += 1
                progress_bar.progress(frame_idx / max_frames)

        cap.release()
        os.unlink(tmp_path)
        progress_bar.empty()

        if probs:
            probs_np   = np.array(probs)
            avg_prob   = probs_np.mean()
            max_prob   = probs_np.max()
            n_drowsy   = (probs_np > 0.5).sum()
            pct_drowsy = n_drowsy / len(probs_np) * 100

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames analysed", len(probs))
            c2.metric("Avg drowsy prob",  f"{avg_prob*100:.1f}%")
            c3.metric("Peak drowsy prob", f"{max_prob*100:.1f}%")
            c4.metric("Drowsy frames",   f"{n_drowsy} ({pct_drowsy:.0f}%)")

            # Timeline chart
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(frame_nums, probs_np, alpha=0.4, color="#E24B4A")
            ax.plot(frame_nums, probs_np, color="#A32D2D", linewidth=1.2)
            ax.axhline(0.5, color="gray", linestyle="--",
                       linewidth=0.8, label="threshold (0.5)")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Drowsy probability")
            ax.set_ylim(0, 1)
            ax.set_title("Drowsiness probability over time")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            overall = "Drowsy" if avg_prob > 0.5 else "Alert"
            if overall == "Drowsy":
                st.error(f"Overall assessment: DROWSY — {pct_drowsy:.0f}% of frames indicate drowsiness.")
            else:
                st.success(f"Overall assessment: ALERT — only {pct_drowsy:.0f}% of frames indicate drowsiness.")


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ════════════════════════════════════════════════════════════
else:
    st.title("Model information")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Architecture")
        st.markdown("""
- **Backbone**: EfficientNet-B0 (pretrained ImageNet)
- **Attention**: Squeeze-and-Excitation (SE) block
- **Temporal**: Bidirectional LSTM (256 hidden, 2 layers)
- **Classifier**: FC 512 → 128 → 2 with GELU + Dropout
- **Input**: sequence of 4 frames, 224×224×3
- **Output**: Alert (0) / Drowsy (1)
        """)

    with col2:
        st.subheader("Training results")
        st.metric("Best val F1",  f"{ckpt['val_f1']:.4f}")
        st.metric("Best val acc", f"{ckpt['val_acc']*100:.2f}%")
        st.metric("Best epoch",   ckpt["epoch"])

    st.subheader("Datasets used")
    st.markdown("""
| Dataset | Images | Type | Label source |
|---------|--------|------|--------------|
| MRL Eye | 84,000 | Eye images (IR) | Filename index |
| Kaggle DDD | 41,793 | Face frames (RGB) | Folder name |
| Total after cleaning | 126,167 | Mixed | — |
    """)

    st.subheader("Optimization results")
    st.markdown("""
| Metric | Baseline | Quantized |
|--------|----------|-----------|
| Model size | ~34 MB | ~9 MB (4x smaller) |
| F1 score | 0.9839 | ~0.9821 |
| Technique | FP32 | INT8 dynamic quantization |
    """)