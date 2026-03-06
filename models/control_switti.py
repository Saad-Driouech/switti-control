"""
ControlNet-style spatial control for Switti.

Freezes the pretrained Switti model and trains a parallel spatial conditioning
module that injects control signals between self-attention and cross-attention
in each frozen transformer block.
"""
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.switti import Switti, get_crop_condition

__all__ = ["SpatialEncoder", "SwittiControlNet"]

MODALITY_IDS = {"canny": 0, "depth": 1, "seg": 2, "normal": 3, "hed": 4}


def _gn(num_channels: int) -> nn.GroupNorm:
    num_groups = min(num_channels // 4, 32)
    return nn.GroupNorm(num_groups, num_channels)


class SpatialEncoder(nn.Module):
    """
    Encodes a control image (canny edges, depth map, etc.) into multi-scale
    spatial tokens matching Switti's patch_nums pyramid.

    Architecture:
        backbone: Conv(3→16) → Conv(16→32, s=2) → Conv(32→64, s=2) → Conv(64→128)
                  all with GroupNorm + SiLU
        per scale: adaptive_avg_pool2d(feat, (pn, pn)) → add modality_embed → proj to C
    """

    def __init__(self, num_modalities: int = 5, out_dim: int = 1024):
        super().__init__()
        self.out_dim = out_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            _gn(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            _gn(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            _gn(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            _gn(128),
            nn.SiLU(),
        )
        # ID = num_modalities is the null/dropout modality
        self.modality_embed = nn.Embedding(num_modalities + 1, 128)
        self.proj = nn.Linear(128, out_dim)

    def extract_features(self, ctrl_image: torch.Tensor) -> torch.Tensor:
        """Run backbone once → (B, 128, H/4, W/4)."""
        return self.backbone(ctrl_image)

    def tokens_from_features(
        self,
        feat: torch.Tensor,        # (B, 128, H/4, W/4)
        modality_ids: torch.Tensor, # (B,)
        patch_nums: Tuple[int, ...],
    ) -> torch.Tensor:              # (B, L, C)
        B = feat.shape[0]
        mod_embed = self.modality_embed(modality_ids)  # (B, 128)

        tokens_list = []
        for pn in patch_nums:
            f = F.adaptive_avg_pool2d(feat, (pn, pn))          # (B, 128, pn, pn)
            f = f + mod_embed[:, :, None, None]                  # broadcast
            f = f.permute(0, 2, 3, 1).reshape(B, pn * pn, 128)  # (B, pn², 128)
            tokens_list.append(self.proj(f))                     # (B, pn², C)

        return torch.cat(tokens_list, dim=1)  # (B, L, C)

    def forward(
        self,
        ctrl_image: torch.Tensor,   # (B, 3, H, W)
        modality_ids: torch.Tensor, # (B,)
        patch_nums: Tuple[int, ...],
    ) -> torch.Tensor:              # (B, L, C)
        feat = self.extract_features(ctrl_image)
        return self.tokens_from_features(feat, modality_ids, patch_nums)


class SwittiControlNet(nn.Module):
    """
    ControlNet wrapper for Switti.

    Trainable parameters:
      - spatial_encoder: encodes ctrl_image into spatial tokens
      - control_blocks: full copy of Switti blocks, initialized from pretrained weights
      - zero_convs: nn.Linear(C, C) per block, zero-initialized

    Frozen parameters:
      - frozen_switti: the pretrained Switti model (unchanged)

    Control signals are injected between self-attention and cross-attention in
    each frozen block. Zero-init guarantees that at training start the output
    is identical to uncontrolled Switti.
    """

    def __init__(self, frozen_switti: Switti, num_modalities: int = 5):
        super().__init__()
        frozen_switti.requires_grad_(False)
        self.frozen_switti = frozen_switti

        C = frozen_switti.C
        self.spatial_encoder = SpatialEncoder(num_modalities=num_modalities, out_dim=C)
        self.control_blocks = copy.deepcopy(frozen_switti.blocks)  # trainable
        self.zero_convs = nn.ModuleList(
            [self._make_zero_conv(C) for _ in range(len(frozen_switti.blocks))]
        )

    @staticmethod
    def _make_zero_conv(C: int) -> nn.Linear:
        layer = nn.Linear(C, C)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def _frozen_block_forward(
        self,
        frz_blk,
        x: torch.Tensor,
        ctrl_signal: torch.Tensor,
        cond_BD: torch.Tensor,
        attn_bias,
        prompt_embeds: Optional[torch.Tensor],
        prompt_attn_bias: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        crop_cond: Optional[torch.Tensor],
        ctrl_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Inline forward of a frozen AdaLNSelfCrossAttn block with ctrl_signal
        injected between self-attention and cross-attention residuals.

        Mirrors AdaLNSelfCrossAttn.forward exactly, splitting at the injection point.
        """
        _cond_BD = cond_BD
        if frz_blk.use_crop_cond and crop_cond is not None:
            _cond_BD = _cond_BD + frz_blk.crop_cond_scales * crop_cond

        gamma1, gamma2, scale1, scale2, shift1, shift2 = (
            frz_blk.ada_lin(_cond_BD).view(-1, 1, 6, frz_blk.C).unbind(2)
        )

        # --- Self-attention residual ---
        x = x + frz_blk.self_attention_norm2(
            frz_blk.attn(
                frz_blk.self_attention_norm1(x).mul(scale1.add(1)).add(shift1),
                attn_bias=attn_bias,
                freqs_cis=freqs_cis,
            )
        ).mul(gamma1)

        # --- Inject control signal ---
        x = x + ctrl_strength * ctrl_signal

        # --- Cross-attention (text conditioning) ---
        if prompt_embeds is not None:
            x = x + frz_blk.cross_attention_norm2(
                frz_blk.cross_attn(
                    frz_blk.cross_attention_norm1(x),
                    frz_blk.attention_y_norm(prompt_embeds),
                    context_attn_bias=prompt_attn_bias,
                    freqs_cis=freqs_cis,
                )
            )

        # --- FFN ---
        x = x + frz_blk.ffn_norm2(
            frz_blk.ffn(frz_blk.ffn_norm1(x).mul(scale2.add(1)).add(shift2))
        ).mul(gamma2)

        return x

    def forward(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_attn_bias: torch.Tensor,
        ctrl_image: torch.Tensor,
        modality_ids: torch.Tensor,
        batch_height: Optional[list] = None,
        batch_width: Optional[list] = None,
        ctrl_strength: float = 1.0,
    ) -> torch.Tensor:  # (B, L, V)
        """
        :param x_BLCv_wo_first_l: teacher-forcing input (B, L - first_l, Cvae)
        :param prompt_embeds: (B, ctx_len, context_dim)
        :param pooled_prompt_embeds: (B, pooled_embed_size)
        :param prompt_attn_bias: (B, ctx_len) boolean mask
        :param ctrl_image: (B, 3, H, W) control image in [-1, 1]
        :param modality_ids: (B,) integer modality indices
        :param batch_height / batch_width: for crop conditioning
        :param ctrl_strength: scale applied to every ctrl_signal
        :return: logits (B, L, V)
        """
        switti = self.frozen_switti
        B = x_BLCv_wo_first_l.shape[0]
        ed = switti.L

        # ------------------------------------------------------------------
        # Step 1: Shared embedding (reuse frozen Switti sub-modules)
        # ------------------------------------------------------------------
        with torch.amp.autocast("cuda", enabled=False):
            pooled = switti.text_pooler(pooled_prompt_embeds)
            sos = cond_BD = pooled
            sos = sos.unsqueeze(1).expand(B, switti.first_l, -1) + switti.pos_start.expand(
                B, switti.first_l, -1
            )
            x_BLC = torch.cat((sos, switti.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += switti.lvl_embed(switti.lvl_1L[:, :ed].expand(B, -1))
            if not switti.rope:
                x_BLC += switti.pos_1LC[:, :ed]

        attn_bias = switti.attn_bias_for_masking[:, :, :ed, :ed]

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(batch_height, batch_width).to(cond_BD.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        # determine mixed-precision dtype (mirrors Switti.forward hack)
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # ------------------------------------------------------------------
        # Step 2: Spatial tokens  (B, L, C)
        # ------------------------------------------------------------------
        ctrl_tokens = self.spatial_encoder(
            ctrl_image.to(dtype=main_type), modality_ids, switti.patch_nums
        ).to(dtype=main_type)

        # ------------------------------------------------------------------
        # Step 3: Parallel block loop
        # ------------------------------------------------------------------
        x_frozen = x_BLC
        x_ctrl = x_BLC + ctrl_tokens
        freqs_cis = switti.freqs_cis  # (1, L, C//2) — full sequence for training

        for frz_blk, ctrl_blk, zero_conv in zip(
            switti.blocks, self.control_blocks, self.zero_convs
        ):
            # Control branch
            x_ctrl = ctrl_blk(
                x=x_ctrl,
                cond_BD=cond_BD,
                attn_bias=attn_bias,
                context=prompt_embeds,
                context_attn_bias=prompt_attn_bias,
                freqs_cis=freqs_cis,
                crop_cond=crop_cond,
            )
            ctrl_signal = zero_conv(x_ctrl)

            # Frozen branch with injection
            x_frozen = self._frozen_block_forward(
                frz_blk=frz_blk,
                x=x_frozen,
                ctrl_signal=ctrl_signal,
                cond_BD=cond_BD,
                attn_bias=attn_bias,
                prompt_embeds=prompt_embeds,
                prompt_attn_bias=prompt_attn_bias,
                freqs_cis=freqs_cis,
                crop_cond=crop_cond,
                ctrl_strength=ctrl_strength,
            )

        # ------------------------------------------------------------------
        # Step 4: Logits
        # ------------------------------------------------------------------
        with torch.amp.autocast("cuda", enabled=not self.training):
            return switti.get_logits(x_frozen, cond_BD.float())
