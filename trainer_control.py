"""
Training loop for SwittiControlNet.

Mirrors trainer.py but:
  - Dataloader yields (images, ctrl_images, captions, modality_ids)
  - The model forward call passes ctrl_images and modality_ids
  - Optimizer only updates spatial_encoder, control_blocks, zero_convs
  - Supports ctrl_dropout_prob: replaces modality_ids with null ID
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.utils import make_grid

import dist
from models.vqvae import VQVAE
from models.control_switti import SwittiControlNet
from models.control_pipeline import SwittiControlPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

EVAL_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed",
    "A sad puppy with large eyes",
    "A girl with pale blue hair and a cami tank top",
    "cute girl, Kyoto animation, 4k, high resolution",
    "A city in 4-dimensional space-time",
]


class SwittiControlTrainer:
    def __init__(
        self,
        dataloader,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        pipe: SwittiControlPipeline,
        vae_local: VQVAE,
        control_net_wo_ddp: SwittiControlNet,
        control_net: FSDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        args=None,
    ):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.args = args

        self.control_net = control_net
        self.control_net_wo_ddp = control_net_wo_ddp
        self.vae_local = vae_local
        self.quantize_local = vae_local.quantize
        self.optimizer = optimizer
        self.pipe = pipe

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")

        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.device = device
        self.grad_accum = args.grad_accum
        self.ctrl_dropout_prob = getattr(args, "ctrl_dropout_prob", 0.1)
        self.num_modalities = getattr(args, "num_modalities", 5)

    def _apply_ctrl_dropout(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """Replace modality_ids with null ID with probability ctrl_dropout_prob."""
        if self.ctrl_dropout_prob <= 0:
            return modality_ids
        mask = torch.bernoulli(
            torch.full((modality_ids.shape[0],), self.ctrl_dropout_prob)
        ).bool()
        null_id = self.num_modalities  # null = num_modalities (last entry in embedding)
        return torch.where(mask.to(modality_ids.device), null_id, modality_ids)

    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:

        self.control_net.train()
        for accum_iter in range(self.grad_accum):
            images, ctrl_images, captions, modality_ids = next(self.dataloader)

            inp_B3HW = images.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW, size=(self.resos[-1], self.resos[-1]), mode="bicubic"
            )
            ctrl_images = ctrl_images.to(self.device, non_blocking=True)
            modality_ids = modality_ids.to(self.device, non_blocking=True)

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_switti_input(gt_idx_Bl)

            # Unconditional text dropout
            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(
                    torch.full((B,), self.args.uncond_proba)
                )
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        captions[i_] = ""

            # Control modality dropout (null modality for spatial CFG)
            modality_ids = self._apply_ctrl_dropout(modality_ids)

            (
                prompt_embeds,
                pooled_prompt_embeds,
                prompt_attn_bias,
            ) = self.pipe.encode_prompt(captions, encode_null=False)

            # All training images are resized to data_load_reso; supply fixed crop condition
            batch_hw = B * [self.resos[-1]]

            with self.optimizer.amp_ctx:
                logits_BLV = self.control_net(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    ctrl_image=ctrl_images,
                    modality_ids=modality_ids,
                    batch_height=batch_hw,
                    batch_width=batch_hw,
                )
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()

            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss, is_stepping=is_stepping
            )

        # Logging
        if g_it > 0 and g_it % self.args.log_iters == 0:
            self.control_net.eval()
            with torch.no_grad(), self.optimizer.amp_ctx:
                logits_BLV = self.control_net(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    ctrl_image=ctrl_images,
                    modality_ids=modality_ids,
                    batch_height=batch_hw,
                    batch_width=batch_hw,
                )

            pred_BL = logits_BLV.data.argmax(dim=-1)
            prob_per_class = pred_BL.view(-1).bincount(minlength=V).float().cuda()
            dist.allreduce(prob_per_class)
            prob_per_class /= prob_per_class.sum()
            cluster_usage = (prob_per_class > 0.001 / V).float().mean().item() * 100

            kw = dict(z_voc_usage=cluster_usage, acc_total=0.0, L_total=0.0)
            logits_lg = {}
            for si, (bg, ed) in enumerate(self.begin_ends):
                pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                tar = gt_BL[:, bg:ed].reshape(-1)
                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                stats = torch.tensor([acc, ce], device=dist.get_device())
                dist.allreduce(stats)
                stats /= dist.get_world_size()
                acc, ce = stats.tolist()
                kw[f"acc_{self.resos[si]}"] = acc
                kw[f"L_{self.resos[si]}"] = ce
                kw["acc_total"] += acc / len(self.begin_ends)
                kw["L_total"] += ce / len(self.begin_ends)

            if dist.is_master():
                tb_lg.update(head="Control_iter_loss", **kw, step=g_it)

            self.control_net.train()
            dist.barrier()

        return grad_norm.item(), scale_log2

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }
