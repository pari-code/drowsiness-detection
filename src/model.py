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
        s = x.mean(dim=1)           # (B, C) — squeeze across time
        s = self.fc(s)              # (B, C) — excitation
        return x * s.unsqueeze(1)  # (B, T, C) — scale


class DrowsinessDetector(nn.Module):
    """
    EfficientNet-B0 + SE attention + BiLSTM + classifier.
    Input:  (B, T, 3, 224, 224)
    Output: (B, 2) — logits for [alert, drowsy]
    """
    CNN_OUT_DIM = 1280

    def __init__(
        self,
        seq_len:    int = 4,
        hidden:     int = 256,
        num_layers: int = 2,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len

        # CNN backbone
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False

        # SE attention
        self.se = SEBlock(self.CNN_OUT_DIM, reduction=16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = self.CNN_OUT_DIM,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden * 2

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(0.4),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.view(B, T, self.CNN_OUT_DIM)
        x = self.se(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.classifier(x)

    def unfreeze_cnn(self, blocks: int = 2):
        features   = list(self.cnn.children())[0]
        all_blocks = list(features.children())
        for block in all_blocks[-blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_all_cnn(self):
        for p in self.cnn.parameters():
            p.requires_grad = True

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params     : {total:,}")
        print(f"Trainable params : {trainable:,}")
        print(f"Frozen params    : {total - trainable:,}")
        return total, trainable