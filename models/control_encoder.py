import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
from torchvision import models


# ------------------------------
# CNN-BASED CONTROL ENCODER
# ------------------------------
class CNNControlEncoder(nn.Module):
    """
    Simple CNN backbone to produce control features.
    Returns (B, out_channels, Hc, Wc).
    """
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
            # Use ImageNet-pretrained feature extractor (light ResNet)
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            modules = list(backbone.children())[:-2]  # remove avgpool + fc
            self.encoder = nn.Sequential(*modules)
            in_ch = 512
        else:
            # Custom lightweight CNN
            layers = []
            in_ch = 3
            ch = mid_channels
            for _ in range(num_downsamples):
                layers += [
                    nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=2),
                ]
                in_ch = ch
                ch = min(ch * 2, 512)
            self.encoder = nn.Sequential(*layers)

        # final projection to desired control dimension
        self.out_proj = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) in [0,1] or normalized.
        returns: (B, out_channels, Hc, Wc)
        """
        feat = self.encoder(x)
        feat = self.out_proj(feat)
        return feat


# ------------------------------
# VIT-BASED CONTROL ENCODER
# ------------------------------
class ViTControlEncoder(nn.Module):
    """
    Lightweight ViT-like control encoder.
    patchify -> Transformer -> project.
    Returns (B, Lp, out_dim).
    """
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
        self.patch_size = patch_size
        self.pretrained = pretrained

        if pretrained:
            # use pretrained DINOv2-small or ViT-B/16 from timm
            import timm
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=True)
            vit_dim = self.vit.embed_dim
            self.proj = nn.Linear(vit_dim, out_dim)
        else:
            # small custom ViT encoder
            self.patch_embed = nn.Conv2d(
                3, vit_hidden, kernel_size=patch_size, stride=patch_size
            )
            self.pos_embed = None  # created lazily when input size known
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=vit_hidden, nhead=heads, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=vit_layers
            )
            self.proj = nn.Linear(vit_hidden, out_dim)
            self.vit = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if self.pretrained and self.vit is not None:
            # Pretrained ViT expects 224x224 usually
            expected_size = self.vit.patch_embed.img_size[0]
            if H != expected_size or W != expected_size:
                x = F.interpolate(x, size=(expected_size, expected_size), mode="bicubic", align_corners=False)
            feats = self.vit.forward_features(x)  # (B, L, vit_dim)
            return self.proj(feats)  # (B, L_ctrl, out_dim)

        # custom path
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), "H,W must be divisible by patch_size"

        patches = self.patch_embed(x)  # (B, D, Hc, Wc)
        _, D, Hc, Wc = patches.shape
        L = Hc * Wc
        tokens = patches.flatten(2).transpose(1, 2)  # (B, L, D)

        # lazy init position embedding
        if self.pos_embed is None or self.pos_embed.shape[1] != L:
            self.pos_embed = nn.Parameter(torch.zeros(1, L, D, device=tokens.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        tokens = tokens + self.pos_embed
        tokens = self.transformer_encoder(tokens)
        return self.proj(tokens)  # (B, L_ctrl, out_dim)


# ------------------------------
# WRAPPER FOR SELECTION
# ------------------------------
class ControlEncoder(nn.Module):
    """
    Wrapper that returns control tokens in shape (B, L_ctrl, ctrl_dim).
    If cnn: output is flattened from (B, Cctrl, Hc, Wc) -> (B, L_ctrl, ctrl_dim).
    If vit: already (B, L_ctrl, ctrl_dim).
    """
    def __init__(
        self,
        encoder_type: Literal["cnn", "vit"] = "cnn",
        control_context_dim: int = 512,
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.control_context_dim = control_context_dim
        self.pretrained = pretrained

        if encoder_type == "cnn":
            self.enc = CNNControlEncoder(
                out_channels=control_context_dim, pretrained=pretrained
            )
            self._is_cnn = True
        elif encoder_type == "vit":
            self.enc = ViTControlEncoder(
                out_dim=control_context_dim, pretrained=pretrained
            )
            self._is_cnn = False
        else:
            raise ValueError(f"Unknown control encoder type: {encoder_type}")

    def forward(self, img: torch.Tensor):
        """
        img: (B, 3, H, W)
        returns: (B, L_ctrl, control_context_dim)
        """
        out = self.enc(img)
        if self._is_cnn:
            B, Cc, Hc, Wc = out.shape
            tokens = out.flatten(2).transpose(1, 2)  # (B, L_ctrl, Cc)
            return tokens
        else:
            return out
