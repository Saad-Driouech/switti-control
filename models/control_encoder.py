# models/control_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple
from torchvision import models
import timm


# ============================================================
# -------- CUSTOM ViT BACKBONE (for pretrained=False) --------
# ============================================================
class CustomViTBackbone(nn.Module):
    def __init__(
        self,
        vit_hidden: int,
        patch_size: int,
        vit_layers: int,
        heads: int,
        max_img_size: int = 1024,  # Safe upper bound
    ):
        super().__init__()
        self.patch_size = patch_size
        self.vit_hidden = vit_hidden

        # --- Patch embedding ---
        self.patch_embed = nn.Conv2d(
            3, vit_hidden,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_hidden,
            nhead=heads,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=vit_layers
        )

        # --- Pre-allocate positional embeddings ---
        max_seq_len = (max_img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, vit_hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.max_len = max_seq_len

    # --------------------------------------------------------
    #  POSITIONAL EMBEDDING EXTRACTION + 2D AWARE INTERPOLATION
    # --------------------------------------------------------
    def _get_pos_embed(self, seq_len: int) -> torch.Tensor:
        """
        Returns a new tensor (never modifies self.pos_embed).
        Uses 2D-structured interpolation where needed.
        """
        max_len = self.max_len

        # Case 1 — direct prefix slice (good for smaller inputs)
        if seq_len <= max_len:
            # Check if seq_len is a perfect square
            side = int(seq_len ** 0.5)
            if side * side == seq_len:
                max_side = int(max_len ** 0.5)

                pe = self.pos_embed.reshape(1, max_side, max_side, self.vit_hidden)
                pe = pe[:, :side, :side, :]
                return pe.reshape(1, seq_len, self.vit_hidden)

            # Else: fallback to flat prefix
            return self.pos_embed[:, :seq_len, :]

        # Case 2 — need to upsample positional embeddings
        old_side = int(max_len ** 0.5)
        new_side = int(seq_len ** 0.5)

        pe = self.pos_embed.reshape(1, old_side, old_side, self.vit_hidden)
        pe = pe.permute(0, 3, 1, 2)

        pe = F.interpolate(
            pe,
            size=(new_side, new_side),
            mode="bicubic",
            align_corners=False,
        )

        pe = pe.permute(0, 2, 3, 1)
        return pe.reshape(1, seq_len, self.vit_hidden)

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        tokens = patches.flatten(2).transpose(1, 2)
        L = tokens.size(1)

        pos = self._get_pos_embed(L)
        tokens = tokens + pos

        return self.transformer_encoder(tokens)  # (B, L, D)


# ============================================================
# ----------------------- CNN ENCODER ------------------------
# ============================================================
class CNNControlEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 512,
        mid_channels: int = 64,
        num_downsamples: int = 4,
        pretrained: bool = False,
    ):
        super().__init__()
        self.pretrained = pretrained

        if pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            in_ch = 512
            
            # Add ImageNet normalization constants
            self.register_buffer('imagenet_mean', 
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('imagenet_std', 
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            layers, in_ch, ch = [], 3, mid_channels
            for _ in range(num_downsamples):
                layers += [
                    nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2),
                ]
                in_ch = ch
                ch = min(ch * 2, 512)

            self.backbone = nn.Sequential(*layers)

        self.proj = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        if self.pretrained:
            # Convert [-1, 1] -> [0, 1]
            x = (x + 1) / 2
            # Apply ImageNet normalization
            x = (x - self.imagenet_mean) / self.imagenet_std
        feat = self.backbone(x)
        return self.proj(feat)


# ============================================================
# ----------------------- VIT ENCODER ------------------------
# ============================================================
class ViTControlEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 512,
        patch_size: int = 16,
        vit_hidden: int = 256,
        vit_layers: int = 4,
        heads: int = 8,
        pretrained: bool = False,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.patch_size = patch_size

        if pretrained:
            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=True,
                num_classes=0,
                features_only=False,
            )
            backbone_dim = self.backbone.embed_dim
            self.expected_size = self.backbone.patch_embed.img_size[0]
            self.patch_size = self.backbone.patch_embed.patch_size[0]

            # Add ImageNet normalization constants
            self.register_buffer('imagenet_mean', 
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('imagenet_std', 
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        else:
            self.backbone = CustomViTBackbone(
                vit_hidden=vit_hidden,
                patch_size=patch_size,
                vit_layers=vit_layers,
                heads=heads,
            )
            backbone_dim = vit_hidden
            self.expected_size = None

        # ViT works with 1D tokens (B, L, D), so use nn.Linear for projection
        self.proj = nn.Linear(backbone_dim, out_dim)

    def forward(self, x):
        if self.pretrained:
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)

            x = (x - self.imagenet_mean) / self.imagenet_std

            if x.shape[2] != self.expected_size:
                x = F.interpolate(
                    x,
                    (self.expected_size, self.expected_size),
                    mode="bicubic",
                    align_corners=False,
                )

            feats = self.backbone.forward_features(x) # (B, 1+L, D_backbone)

            # remove CLS if present
            if feats.shape[1] > 1 and hasattr(self.backbone, "global_pool"):
                feats = feats[:, 1:] # (B, L, D_backbone)

        else:
            feats = self.backbone(x) # (B, L, D_backbone)

        # Apply Linear projection: (B, L, D_backbone) -> (B, L, D_out)
        out = self.proj(feats) 

        # --- Convert 1D token sequence back to 2D feature map ---
        B, L, D = out.shape
        side = int(L**0.5)
        
        # Check if L is a perfect square, which it must be for a ViT token sequence
        if side * side != L:
            raise ValueError(f"ViT output sequence length {L} is not a perfect square. Cannot convert to 2D feature map.")
            
        # Reshape (B, L, D) -> (B, D, Hc, Wc)
        feat_map = out.transpose(1, 2).reshape(B, D, side, side)
        
        return feat_map # (B, D, Hc, Wc)


# ============================================================
# -------------- MULTI-SCALE WRAPPER -------------------------
# ============================================================
class MultiScaleControlEncoder(nn.Module):
    """
    Encodes control image to multiple scales matching Switti's patch_nums.
    
    Returns dict: {patch_num: (B, patch_num², control_dim)}
    """
    def __init__(
        self,
        encoder_type: Literal["cnn", "vit"] = "cnn",
        control_context_dim: int = 512,
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.patch_nums = patch_nums
        self.control_context_dim = control_context_dim

        # Base encoder (outputs spatial features: B, C, H_feat, W_feat)
        if encoder_type == "cnn":
            self.encoder = CNNControlEncoder(
                out_channels=control_context_dim,
                pretrained=pretrained,
            )
        elif encoder_type == "vit":
            self.encoder = ViTControlEncoder(
                out_dim=control_context_dim,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Invalid encoder_type: {encoder_type}")

    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W) control image
            
        Returns:
            dict[int, Tensor]: {patch_num: (B, patch_num², control_dim)}
        """
        if img is None:
            return None

        # Extract spatial features
        features = self.encoder(img)  # (B, control_dim, H_feat, W_feat)

        # Adaptively pool to each target scale
        control_per_scale = {}
        for pn in self.patch_nums:
            # Pool to target resolution
            pooled = F.adaptive_avg_pool2d(features, (pn, pn))  # (B, C, pn, pn)
            
            # Flatten to tokens (B, pn², C)
            tokens = pooled.flatten(2).transpose(1, 2)
            
            control_per_scale[pn] = tokens

        return control_per_scale


# ============================================================
# ---------------------- WRAPPER -----------------------------
# ============================================================
class ControlEncoder(nn.Module):
    """
    Main control encoder class that dispatches to multi-scale implementation.
    """
    def __init__(
        self,
        encoder_type: Literal["cnn", "vit"] = "cnn",
        control_context_dim: int = 512,
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder = MultiScaleControlEncoder(
            encoder_type=encoder_type,
            control_context_dim=control_context_dim,
            patch_nums=patch_nums,
            pretrained=pretrained,
        )

    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W) or None
            
        Returns:
            dict[int, Tensor] or None: {patch_num: (B, patch_num², control_dim)}
        """
        return self.encoder(img)