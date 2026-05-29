"""PyTorch ports of the three classifiers originally written in TF/Keras.

Why a port: the upstream Neuro-Lens-AI repo's classifier weights ship as Git LFS
pointer files (134 bytes each), so a zip download from GitHub does not include
the actual binaries. The one .h5 that IS in upstream (real_eval_current/vit)
was trained with a different version of the model code (ResNet50 flattened at
the top level; our build_vit_classifier nests it as 'vit_hybrid_resnet_base')
and is topology-incompatible with src/models.py. TF 2.21 on this machine is
CPU-only (no native-Windows GPU), so CPU retraining is ~3 hrs per model.

PyTorch on the RTX 4060 cuts that to ~5 min per model. Architectures are
matched to src/models.py as closely as possible so behaviour matches what the
original paper describes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNClassifier(nn.Module):
    """Custom 3-block CNN baseline. Matches src/models.py:build_cnn_baseline.

    3 Conv2D+MaxPool blocks (32 -> 64 -> 128 channels) -> Flatten -> Dense(128)
    -> Dropout -> Dense(1, sigmoid). The TF version uses Rescaling(1/255) as the
    first layer; we expect callers to pass images already in [0,1].
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        # 224 / 2 / 2 / 2 = 28, so feature map is 128 x 28 x 28.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    @property
    def last_conv_module(self):
        """Grad-CAM target. Property avoids duplicating features[6] in state_dict."""
        return self.features[6]  # the Conv2d(64, 128, ...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TransferClassifier(nn.Module):
    """ResNet50 backbone (ImageNet pretrained) + classification head. Matches
    src/models.py:build_transfer_model with default args (resnet50, fine_tune=False).

    The backbone outputs (B, 2048, 7, 7); we GAP to (B, 2048), then
    Dropout -> Dense(256, relu) -> Dropout -> Dense(1).
    """

    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = True):
        super().__init__()
        # weights=ResNet50_Weights.IMAGENET1K_V2 gives the better pretrained weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = self.backbone.fc.in_features  # 2048
        # Replace the classification head with our own
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Keep BN layers in eval mode regardless of model.train()
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @property
    def last_conv_module(self):
        """Grad-CAM target. Not registered as a child module via this property,
        so the state_dict shape stays unchanged."""
        return self.backbone.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 2048)
        return self.head(features)


class ViTHybridClassifier(nn.Module):
    """Hybrid ResNet50 + 4 transformer-block ViT. Matches
    src/models.py:build_vit_classifier with default args.

    Frozen ResNet50 outputs (B, 2048, 7, 7) -> Conv1x1 to projection_dim (128)
    -> reshape to (B, 49, 128) token sequence -> learnable position embedding
    -> 4 transformer encoder blocks (4 heads, mlp_dim=256) -> LayerNorm ->
    GlobalAveragePool over tokens -> Dropout -> Dense(128, relu) -> Dense(1).
    """

    def __init__(self, projection_dim: int = 128, num_layers: int = 4, num_heads: int = 4,
                 mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the classification head and avgpool so we keep the feature map.
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        # Freeze
        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # 1x1 patch projection
        self.patch_projection = nn.Conv2d(2048, projection_dim, kernel_size=1)
        self.num_patches = 7 * 7  # 224 / 32 = 7
        self.projection_dim = projection_dim
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, projection_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # match TF behaviour: norm after, not before
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(projection_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @property
    def last_conv_module(self):
        """Grad-CAM target on the ResNet50 last conv block. Property, not a
        registered submodule, so state_dict layout matches the trained weights."""
        return self.backbone.layer4[-1]

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet50 forward up through layer4, keeping the (B, 2048, 7, 7) feature map."""
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return x  # (B, 2048, 7, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._run_backbone(x)                    # (B, 2048, 7, 7)
        patches = self.patch_projection(feat)            # (B, 128, 7, 7)
        tokens = patches.flatten(2).transpose(1, 2)      # (B, 49, 128)
        tokens = tokens + self.position_embedding
        tokens = self.transformer(tokens)                # (B, 49, 128)
        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)                      # (B, 128)
        return self.head(pooled)


def get_classifier(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name == 'cnn':
        return CNNClassifier()
    if name == 'transfer':
        return TransferClassifier()
    if name == 'vit':
        return ViTHybridClassifier()
    raise ValueError(f'Unknown classifier: {model_name!r}. Choose cnn / transfer / vit.')


@torch.no_grad()
def predict_probability(model: nn.Module, image_chw_01: torch.Tensor) -> float:
    """Run inference on a single image already in (C, H, W) layout, [0,1]
    range. Returns the sigmoid probability of the 'tumor' class."""
    model.eval()
    logit = model(image_chw_01.unsqueeze(0))
    return float(torch.sigmoid(logit).item())


__all__ = ['CNNClassifier', 'TransferClassifier', 'ViTHybridClassifier', 'get_classifier', 'predict_probability']
