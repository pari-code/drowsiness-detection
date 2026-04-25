import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention. Input/output: (B, T, C)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = x.mean(dim=1)      # (B, C) — squeeze across time
        s = self.fc(s)          # (B, C) — excitation
        return x * s.unsqueeze(1)  # (B, T, C) — scale


class DrowsinessDetector(nn.Module):
    """
    EfficientNet-B0 + SE attention + BiLSTM + classifier.

    Args:
        seq_len:    number of frames per sequence (default 16)
        hidden:     LSTM hidden size per direction (default 256)
        num_layers: number of LSTM layers (default 2)
        dropout:    dropout probability (default 0.3)

    Input:  (B, T, 3, 224, 224)
    Output: (B, 2) — logits for [alert, drowsy]
    """
    CNN_OUT_DIM = 1280   # EfficientNet-B0 feature dimension

    def __init__(
        self,
        seq_len:    int = 16,
        hidden:     int = 256,
        num_layers: int = 2,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── 1. CNN backbone ────────────────────────────────────
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove classifier (last child) — keep only the feature extractor
        # base.children() = [features, avgpool, classifier]
        # We keep features + avgpool → outputs (B, 1280, 1, 1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        # Freeze ALL CNN weights initially
        # We unfreeze them in train.py during Phase 2 of training
        for p in self.cnn.parameters():
            p.requires_grad = False

        # ── 2. SE attention ────────────────────────────────────
        self.se = SEBlock(self.CNN_OUT_DIM, reduction=16)

        # ── 3. Bidirectional LSTM ──────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = self.CNN_OUT_DIM,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,      # input: (B, T, features)
            bidirectional = True,      # doubles output dim: hidden*2
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden * 2     # 256 * 2 = 512 (bidirectional)

        # ── 4. Classifier head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(0.4),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)        # 2 classes: alert=0, drowsy=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # ── Reshape: merge batch+time for CNN ──────────────────
        x = x.view(B * T, C, H, W)         # (B*T, 3, 224, 224)

        # ── CNN: extract per-frame features ────────────────────
        x = self.cnn(x)                    # (B*T, 1280, 1, 1)
        x = x.squeeze(-1).squeeze(-1)    # (B*T, 1280)

        # ── Reshape: separate batch and time for LSTM ──────────
        x = x.view(B, T, self.CNN_OUT_DIM) # (B, T, 1280)

        # ── SE attention: reweight feature channels ────────────
        x = self.se(x)                     # (B, T, 1280)

        # ── LSTM: temporal encoding ────────────────────────────
        lstm_out, _ = self.lstm(x)         # (B, T, 512)
        x = lstm_out[:, -1, :]            # (B, 512) — last timestep only

        # ── Classify ───────────────────────────────────────────
        return self.classifier(x)          # (B, 2)

    def unfreeze_cnn(self, blocks: int = 2):
        """
        Unfreeze the last N blocks of EfficientNet for fine-tuning.
        Call this in train.py at epoch 5 (Phase 2 training).
        blocks=2 unfreezes blocks 6 and 7 (out of 0–8).
        """
        features = list(self.cnn.children())[0]  # EfficientNet features
        all_blocks = list(features.children())
        for block in all_blocks[-blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_all_cnn(self):
        """Unfreeze entire CNN. Call at Phase 3 training (epoch 25+)."""
        for p in self.cnn.parameters():
            p.requires_grad = True

    def count_parameters(self):
        """Print trainable vs total parameters."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"Total params     : {total:,}")
        print(f"Trainable params : {trainable:,}")
        print(f"Frozen params    : {total - trainable:,}")
        return total, trainable