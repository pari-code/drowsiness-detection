import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM for DrowsinessDetector.
    Hooks into the last block of EfficientNet-B0 backbone.

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam.generate(seq_tensor, class_idx=1)
        overlay = gcam.overlay(face_img_bgr, heatmap)
    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Hook into last block of EfficientNet features
        # model.cnn = Sequential(features, avgpool)
        # features[-1] = last MBConv block
        target = list(self.model.cnn.children())[0][-1]

        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, seq: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for one sequence.
        seq: (1, T, 3, 224, 224) — batch of 1 sequence
        class_idx: 1 = drowsy, 0 = alert
        Returns: heatmap as uint8 array (224, 224) values 0–255
        """
        self.model.zero_grad()
        self.model.train()

        # Forward pass with grad enabled
        seq.requires_grad = True
        logits = self.model(seq)             # (1, 2)
        score  = logits[0, class_idx]
        score.backward()

        # gradients: (B*T, C, H, W) — average over spatial dims
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.uint8)

        # Weight each activation map by mean gradient
        weights  = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B*T, C, 1, 1)
        cam      = (weights * self.activations).sum(dim=1)           # (B*T, H, W)

        # Average across time dimension
        cam = cam.mean(dim=0).cpu().numpy()   # (H, W)

        # ReLU and normalise to 0–255
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return (cam * 255).astype(np.uint8)

    def overlay(self, face_bgr: np.ndarray,
                 heatmap: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
        """
        Overlay heatmap on face image.
        face_bgr: (224, 224, 3) BGR image
        heatmap:  (224, 224) uint8 0–255
        Returns:  (224, 224, 3) BGR image with coloured overlay
        """
        face_resized = cv2.resize(face_bgr, (224, 224))
        coloured     = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended      = cv2.addWeighted(face_resized, 1-alpha,
                                        coloured, alpha, 0)
        return blended


def generate_gradcam_samples(model, dataloader, device,
                               output_dir: str = "outputs/gradcam",
                               n_samples: int = 8):
    """
    Generate and save Grad-CAM heatmap images for n samples.
    Run this once after training to produce README images.
    """
    import os
    from pathlib import Path
    from torchvision import transforms
    from PIL import Image

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gcam  = GradCAM(model)
    unnorm = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std= [1/0.229,       1/0.224,       1/0.225]
        )
    ])

    saved = 0
    for imgs, labels in dataloader:
        if saved >= n_samples: break
        for i in range(len(imgs)):
            if saved >= n_samples: break

            # Make pseudo-sequence (1, T, 3, 224, 224)
            seq = imgs[i].unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1, 1).to(device)

            # Generate heatmap
            heatmap = gcam.generate(seq, class_idx=labels[i].item())

            # Unnormalise image for display
            face_t   = unnorm(imgs[i]).clamp(0, 1)
            face_np  = (face_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            overlay  = gcam.overlay(face_bgr, heatmap)
            label_name = "drowsy" if labels[i].item() == 1 else "alert"

            # Side-by-side: original | heatmap overlay
            side_by_side = np.hstack([
                cv2.resize(face_bgr, (224, 224)),
                overlay
            ])
            cv2.putText(side_by_side, f"Label: {label_name}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255,255,255), 1)

            out_path = f"{output_dir}/sample_{saved:02d}_{label_name}.jpg"
            cv2.imwrite(out_path, side_by_side)
            saved += 1

    print(f"Saved {saved} Grad-CAM samples to {output_dir}/")