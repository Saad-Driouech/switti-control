"""
Training loop for SwittiControlNet.

Mirrors trainer.py but:
  - Dataloader yields (images, ctrl_images, captions, modality_ids)
  - The model forward call passes ctrl_images and modality_ids
  - Optimizer only updates spatial_encoder, control_blocks, zero_convs
  - Supports ctrl_dropout_prob: replaces modality_ids with null ID
"""
import json
import math
import os
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

import dist
from models.vqvae import VQVAE
from models.control_switti import SwittiControlNet, MODALITY_IDS
from models.control_pipeline import SwittiControlPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

# Inverse map: int id -> modality name string
ID_TO_MODALITY = {v: k for k, v in MODALITY_IDS.items()}

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


def generate_logging_prompts_captions(
    missing_files_path: str,
    captions_json_path: str,
    ctrl_maps_dir: Optional[str],
    control_modalities: Optional[List[str]],
    num_select: int = 12,
    final_reso: int = 512,
) -> Tuple[List[str], Optional[dict]]:
    """
    Select eval samples for TensorBoard logging.

    Returns:
        selected_captions: list of caption strings
        ctrl_dict: {modality: (N,3,H,W) tensor in [-1,1]} or None
    """
    # Load list of filenames to use for logging
    with open(missing_files_path, "r") as f:
        missing_filenames = [line.strip() for line in f if line.strip()]

    # Load COCO captions
    with open(captions_json_path, "r") as f:
        captions_data = json.load(f)

    file_to_id = {img["file_name"]: img["id"] for img in captions_data["images"]}
    id_to_captions: dict = defaultdict(list)
    for ann in captions_data["annotations"]:
        id_to_captions[ann["image_id"]].append(ann["caption"])

    filtered_items = []
    for fname in missing_filenames:
        image_id = file_to_id.get(fname)
        fname_png = fname.replace(".jpg", ".png")
        if image_id is None:
            continue
        caps = id_to_captions.get(image_id, [])
        if not caps:
            continue
        filtered_items.append((fname_png, caps[0]))

    selected_items = (
        random.sample(filtered_items, num_select)
        if len(filtered_items) > num_select
        else filtered_items
    )
    selected_filenames = [x[0] for x in selected_items]
    selected_captions = [x[1] for x in selected_items]
    print(f"[Trainer] Logging filenames: {selected_filenames}")

    ctrl_dict = None
    if ctrl_maps_dir and control_modalities:
        to_tensor = transforms.Compose([
            transforms.Resize((final_reso, final_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
        ])
        ctrl_dict = {}
        for modality in control_modalities:
            tensors = []
            for fname in selected_filenames:
                ctrl_fp = os.path.join(ctrl_maps_dir, modality, fname)
                if os.path.exists(ctrl_fp):
                    try:
                        img = Image.open(ctrl_fp).convert("RGB")
                        tensors.append(to_tensor(img))
                    except Exception as e:
                        print(f"[Warning] Failed to load {ctrl_fp}: {e}")
                        tensors.append(torch.zeros(3, final_reso, final_reso))
                else:
                    tensors.append(torch.zeros(3, final_reso, final_reso))
            ctrl_dict[modality] = torch.stack(tensors, dim=0)  # (N,3,H,W)

    return selected_captions, ctrl_dict


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
        self.log_modality = (getattr(args, "control_modalities", None) or ["canny"])[0]

        # Load logging data (prompts + control images for eval visualisation)
        try:
            self.log_prompts, self.log_ctrl_dict = generate_logging_prompts_captions(
                missing_files_path=os.path.join(args.data_path, "log_files.txt"),
                captions_json_path=os.path.join(
                    args.data_path, "annotations", "captions_val2014.json"
                ),
                ctrl_maps_dir=getattr(args, "ctrl_maps_dir", None),
                control_modalities=getattr(args, "control_modalities", None),
                final_reso=args.data_load_reso,
            )
        except Exception as e:
            print(f"[Warning] Could not load logging data ({e}); using EVAL_PROMPTS fallback.")
            self.log_prompts = EVAL_PROMPTS
            self.log_ctrl_dict = None

        print(f"[Trainer] Log prompts: {self.log_prompts}")

    # ------------------------------------------------------------------
    # Visualisation helpers (mirrored from SwittiTrainer)
    # ------------------------------------------------------------------

    def _prepare_vis_image(self, item) -> torch.Tensor:
        """Convert PIL image or tensor to float CHW [0,1]."""
        if isinstance(item, Image.Image):
            return TF.to_tensor(item).float()

        if torch.is_tensor(item):
            t = item.detach().cpu().float()
            if t.ndim == 4 and t.size(0) == 1:
                t = t.squeeze(0)
            if t.ndim != 3:
                raise ValueError(f"Expected CHW tensor, got {t.shape}")
            if t.min() < 0:
                t = (t + 1) * 0.5
            return t.clamp(0, 1)

        raise ValueError(f"Unsupported type in _prepare_vis_image: {type(item)}")

    def _resize_and_crop_for_vis(self, t: torch.Tensor) -> torch.Tensor:
        """Resize and centre-crop to data_load_reso."""
        mid_reso = round(self.args.data_load_reso * self.args.mid_reso)
        t = TF.resize(t, mid_reso, antialias=True)
        t = TF.center_crop(t, (self.args.data_load_reso, self.args.data_load_reso))
        return t.clamp(0, 1)

    def _combine_side_by_side(self, left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if left is None:
            return right
        if right is None:
            return left
        H = max(left.shape[1], right.shape[1])
        if left.shape[1] != H:
            left = F.pad(left, (0, 0, 0, H - left.shape[1]))
        if right.shape[1] != H:
            right = F.pad(right, (0, 0, 0, H - right.shape[1]))
        return torch.cat([left, right], dim=2)

    def _make_ctrl_grid(self, ctrl_tensor: torch.Tensor, n_show: Optional[int] = None) -> torch.Tensor:
        """Build a make_grid image from a (N,3,H,W) control tensor in [-1,1]."""
        items = ctrl_tensor[:n_show] if n_show is not None else ctrl_tensor
        tensors = []
        for i in range(items.shape[0]):
            t = self._prepare_vis_image(items[i])
            t = self._resize_and_crop_for_vis(t)
            tensors.append(t)
        imgs = torch.stack(tensors, dim=0)
        return make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))

    def _log_ctrl_and_generated(
        self,
        tb_lg: TensorboardLogger,
        tag_prefix: str,
        gen_imgs: torch.Tensor,        # (3,H,W) make_grid output
        ctrl_tensor: Optional[torch.Tensor],  # (N,3,H,W) or None
        modality: str,
        g_it: int,
        n_show: Optional[int] = None,
    ):
        """Log side-by-side control | generated panel to TensorBoard."""
        gen_grid = gen_imgs.detach().cpu().float().clamp(0, 1)

        if ctrl_tensor is None:
            tb_lg.log_image(f"{tag_prefix}_generated", gen_grid, step=g_it)
            return

        ctrl_grid = self._make_ctrl_grid(ctrl_tensor, n_show).float().clamp(0, 1)
        combined = self._combine_side_by_side(ctrl_grid, gen_grid)
        tb_lg.log_image(f"{tag_prefix}_{modality}_ctrl_and_generated", combined, step=g_it)

    # ------------------------------------------------------------------

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

            # Image logging
            if g_it % self.args.log_images_iters == 0:
                with FSDP.summon_full_params(self.control_net, writeback=False):
                    torch.cuda.empty_cache()

                    # Determine the predominant modality in this batch
                    batch_modality = ID_TO_MODALITY.get(
                        modality_ids[0].item(), self.log_modality
                    )
                    n_show = min(B, 8)

                    # --- Train batch: control + generated ---
                    train_ctrl = ctrl_images[:n_show].cpu()
                    train_prompts = list(captions[:n_show])
                    imgs = self.pipe(
                        prompt=train_prompts,
                        ctrl_image=train_ctrl,
                        modality=batch_modality,
                        cfg=self.args.guidance,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        return_pil=False,
                    )
                    imgs_grid = make_grid(imgs, nrow=math.floor(math.sqrt(n_show)))
                    if dist.is_master():
                        self._log_ctrl_and_generated(
                            tb_lg,
                            tag_prefix=f"train_cfg={self.args.guidance}",
                            gen_imgs=imgs_grid,
                            ctrl_tensor=train_ctrl,
                            modality=batch_modality,
                            g_it=g_it,
                            n_show=n_show,
                        )
                    del imgs, imgs_grid, train_ctrl

                    # --- Eval prompts: control + generated ---
                    log_ctrl = (
                        self.log_ctrl_dict.get(self.log_modality)
                        if self.log_ctrl_dict is not None
                        else None
                    )
                    if log_ctrl is not None:
                        imgs = self.pipe(
                            prompt=self.log_prompts,
                            ctrl_image=log_ctrl,
                            modality=self.log_modality,
                            cfg=self.args.guidance,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                        )
                        n_eval = len(self.log_prompts)
                        imgs_grid = make_grid(imgs, nrow=math.floor(math.sqrt(n_eval)))
                        if dist.is_master():
                            self._log_ctrl_and_generated(
                                tb_lg,
                                tag_prefix=f"eval_cfg={self.args.guidance}",
                                gen_imgs=imgs_grid,
                                ctrl_tensor=log_ctrl,
                                modality=self.log_modality,
                                g_it=g_it,
                            )
                        del imgs, imgs_grid

            if dist.is_master():
                tb_lg.update(head="Control_iter_loss", **kw, step=g_it)

            print(f"LOGGING {g_it} FINISHED")
            self.control_net.train()
            dist.barrier()

        return grad_norm.item(), scale_log2

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }
