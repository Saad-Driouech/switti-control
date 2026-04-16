import math
from typing import List, Optional, Tuple, Union
import json
import os
import random
from collections import defaultdict
from PIL import Image
from PIL.Image import Image as PILImage
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

import dist
from models import Switti, VQVAE
from models.pipeline import SwittiPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger
from utils.perceptual_loss import PerceptualLoss, compute_gate_regularization

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

EVAL_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    "A sad puppy with large eyes",
    "A girl with pale blue hair and a cami tank top",
    "cute girl, Kyoto animation, 4k, high resolution",
    "A person laying on a surfboard holding his dog",
    "Green commercial building with refrigerator and refrigeration units outside",
    "An airplane with two propellor engines flying in the sky",
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "A city in 4-dimensional space-time",
    "A black dog sitting on a wooden chair. A white cat with black ears is standing up with its paws on the chair.",
    "a cat patting a crystal ball with the number 7 written on it in black marker",
    "a barred owl peeking out from dense tree branches",
    "a cat sitting on a stairway railing",
    "a cat drinking a pint of beer",
    "a bat landing on a baseball bat",
    "a black dog sitting between a bush and a pair of green pants standing up with nobody inside them",
    "a close-up of a blue dragonfly on a daffodil",
    "A close-up of two beetles wearing karate uniforms and fighting, jumping over a waterfall."
]

def generate_logging_prompts_captions(
    missing_files_path: str,
    captions_json_path: str,
    control_path: str | None,
    control_types: list[str] | None,
    num_select: int = 12,
    final_reso: int = 512,
    mid_reso: float = 1.125,
) -> tuple[list[str], list[str], dict[str, list[Image.Image] | None]]:
    """
    Select files from missing files list, fetch captions from COCO captions JSON,
    and load control images for each control type.

    Returns:
        selected_captions: list of corresponding captions (str)
        control_dict_batch: dict mapping control_type -> Tensor (N, 3, H, W) in [-1, 1]
    """

    # Load missing filenames
    with open(missing_files_path, 'r') as f:
        missing_filenames = [line.strip() for line in f if line.strip()]

    # Load captions json
    with open(captions_json_path, 'r') as f:
        captions_data = json.load(f)

    # Map image_id -> file_name and vice versa
    id_to_file = {img['id']: img['file_name'] for img in captions_data['images']}
    file_to_id = {v: k for k, v in id_to_file.items()}

    # Map image_id -> list of captions (usually multiple captions per image)
    id_to_captions = defaultdict(list)
    for ann in captions_data['annotations']:
        id_to_captions[ann['image_id']].append(ann['caption'])

    # Filter captions for missing files only
    filtered_items = []
    for fname in missing_filenames:
        image_id = file_to_id.get(fname)
        fname = fname.replace(".jpg", ".png")
        if image_id is None:
            continue
        caps = id_to_captions.get(image_id, [])
        if not caps:
            continue
        # Pick the first caption or join all captions
        caption = caps[0] if caps else ""
        filtered_items.append((fname, caption))

    # Optional: Stratify selection by simple heuristics or categories if available
    # Here, randomly sample num_select or less from filtered list
    if len(filtered_items) > num_select:
        selected_items = random.Random(42).sample(filtered_items, num_select)
    else:
        selected_items = filtered_items

    selected_filenames = [x[0] for x in selected_items]
    selected_captions = [x[1] for x in selected_items]

    print(f"[Trainer] Selected filenames: {selected_filenames}")

    # Load control images if control_path and control_types are provided
    control_dict_batch = None
    if control_path is not None and control_types:
        control_dict_batch = {ctrl: [] for ctrl in control_types}
        
        # Create transform matching training data
        from utils.data import JointTransform
        transform = JointTransform(
            final_reso=final_reso,
            mid_reso=mid_reso,
            hflip_prob=0.0  # No flip for validation
        )
        
        for fname in selected_filenames:
            for ctrl in control_types:
                ctrl_fp = os.path.join(control_path, ctrl, fname)
                if os.path.exists(ctrl_fp):
                    try:
                        img = Image.open(ctrl_fp).convert("RGB")
                        # Apply SAME transform as training data
                        # (dummy dict because transform expects dict input)
                        _, processed = transform(img, {ctrl: img})
                        control_dict_batch[ctrl].append(processed[ctrl])
                    except Exception as e:
                        print(f"[Warning] Failed to process control image {ctrl_fp}: {e}")
                        # Append zero tensor as placeholder
                        dummy = torch.zeros(3, final_reso, final_reso)
                        control_dict_batch[ctrl].append(dummy)
                else:
                    # Missing file: zero tensor
                    dummy = torch.zeros(3, final_reso, final_reso)
                    control_dict_batch[ctrl].append(dummy)
        
        # Stack into tensors
        for ctrl in control_types:
            control_dict_batch[ctrl] = torch.stack(control_dict_batch[ctrl], dim=0)
            # Shape: (N, 3, H, W) in [-1, 1]
    
    return selected_captions, control_dict_batch


class SwittiTrainer(object):
    def __init__(
        self,
        dataloader,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        pipe: SwittiPipeline,
        vae_local: VQVAE,
        switti_wo_ddp: Switti,
        switti: DDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        args=None,
    ):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.args = args

        self.switti, self.vae_local, self.quantize_local = (
            switti,
            vae_local,
            vae_local.quantize,
        )
        self.switti_wo_ddp: Switti = switti_wo_ddp  # after torch.compile
        self.optimizer = optimizer
        self.pipe = pipe
        self.switti_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
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
        self.embed_noise_std = args.embed_noise_std
        self.log_prompts, self.log_control_dict = generate_logging_prompts_captions(
            missing_files_path=os.path.join(args.data_path, "log_files.txt"),
            captions_json_path=os.path.join(args.data_path, "annotations", "captions_val2014.json"),
            control_path=os.path.join(args.data_path, "val_control"),
            control_types=args.control_types,
            final_reso=args.data_load_reso,
            mid_reso=args.mid_reso
        )
        print(f"[Trainer] logging prompts {self.log_prompts}")
        # print(f"[Trainer] logging control dict {self.log_control_dict}")
        df = pd.read_csv("eval_prompts/mjhq.csv")
        self.mjhq_prompts = df["captions"].astype(str).tolist()[:12]
        print(f"[Trainer] MJHQ prompts {self.mjhq_prompts}")
        param = next(self.switti.parameters()).to(self.device)
        self.model_dtype = param.dtype

        # --- Auxiliary losses (optional, CLI-gated) ---
        self.use_perceptual_loss = getattr(args, "use_perceptual_loss", False)
        self.perceptual_loss_weight = getattr(args, "perceptual_loss_weight", 0.1)
        self.perceptual_loss_every_n_steps = getattr(args, "perceptual_loss_every_n_steps", 4)
        self.perceptual_loss_fn = (
            PerceptualLoss(
                lpips_net=getattr(args, "lpips_net", "alex"),
                target_size=getattr(args, "perceptual_loss_resolution", 256),
            )
            if self.use_perceptual_loss else None
        )

        self.use_control_gate = getattr(args, "use_control_gate", False)
        self.use_gate_reg = getattr(args, "use_gate_reg", False)
        self.gate_reg_weight = getattr(args, "gate_reg_weight", 0.1)
        self.gate_reg_target = getattr(args, "gate_reg_target", 0.6)

    # build small control dict for pipe visualization (use up to N images)
    def _build_ctrl_for_pipe(self, ctrl_dict, n):
        """Build small control dict for pipe visualization (use up to N images)"""
        if ctrl_dict is None:
            return None
        
        sub = {}
        for k, v in ctrl_dict.items():
            if v is None:
                sub[k] = None
            elif isinstance(v, torch.Tensor):
                # Already preprocessed tensor from training batch or validation set
                sub[k] = v[:n].cpu()  # Just slice and move to CPU
            else:
                raise TypeError(f"Expected preprocessed tensor for {k}, got {type(v)}")
        
        return sub

    def _prepare_vis_image(self, item):
        if item is None:
            return None

        # PIL → tensor [0,1]
        if isinstance(item, PILImage):
            t = TF.to_tensor(item).float()
            return t

        # Tensor
        if torch.is_tensor(item):
            t = item.detach().cpu().float()

            # Remove batch dim if present
            if t.ndim == 4 and t.size(0) == 1:
                t = t.squeeze(0)

            # Must be CHW now
            if t.ndim != 3:
                raise ValueError(f"Expected CHW tensor, got {t.shape}")

            # [-1,1] → [0,1]
            if t.min() < 0:
                t = (t + 1) * 0.5

            return t.clamp(0, 1)

        raise ValueError(f"Unsupported type in _prepare_vis_image: {type(item)}")

    def _resize_and_crop_for_vis(self, t):
        # t: (3,H,W) in [0,1]
        mid_reso = round(self.args.data_load_reso * self.args.mid_reso)   
        t = TF.resize(t, mid_reso, antialias=True)
        t = TF.center_crop(t, (self.args.data_load_reso, self.args.data_load_reso))
        return t.clamp(0, 1)

    def _combine_side_by_side(self, left, right):
        if left is None:
            return right
        if right is None:
            return left

        # left, right: (3,H,W)
        H = max(left.shape[1], right.shape[1])

        if left.shape[1] != H:
            left = F.pad(left, (0,0,0, H - left.shape[1]))
        if right.shape[1] != H:
            right = F.pad(right, (0,0,0, H - right.shape[1]))

        return torch.cat([left, right], dim=2)

    def _control_grids_for_tb(self, control_dict, n_show=None):
        """
        Converts control images (tensor batch or PIL list) into
        visualization grids that match the size & layout of generated images.
        
        Returns dict[type] → grid tensor (3,H,W).
        """
        if control_dict is None:
            return None

        out = {}

        for ctrl_type, value in control_dict.items():
            if value is None:
                continue

            # Normalize input into list
            if torch.is_tensor(value):
                items = [value[i] for i in range(min(value.shape[0], n_show or value.shape[0]))]
            elif isinstance(value, list):
                items = value[:n_show] if n_show is not None else value
            else:
                print(f"[Warning] unexpected control type for {ctrl_type}: {type(value)}")
                continue

            # Convert each element to tensor (3,H,W) in [0,1]
            tensors = []
            for itm in items:
                if itm is None:
                    continue
                t = self._prepare_vis_image(itm)
                t = self._resize_and_crop_for_vis(t)
                tensors.append(t)

            if len(tensors) == 0:
                continue

            imgs = torch.stack(tensors, dim=0)   # (B,3,H,W)
            grid = make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))
            out[ctrl_type] = grid

        return out

    def _log_pipe_outputs(self, tb_lg, tag_prefix, imgs, control_dict, g_it, n_show=None):
        """
        Logs ONE PANEL per control type:
            | CONTROL GRID | GENERATED GRID |
        Both grids have equal size and are perfectly aligned.
        """
        # Generated grid already (3,H,W) — but make sure it's float32 [0,1]
        gen_grid = imgs.detach().cpu().float()
        gen_grid = gen_grid.clamp(0,1)

        # Build control grids
        control_grids = self._control_grids_for_tb(control_dict, n_show)

        # No control → only log generated
        if not control_grids:
            tb_lg.log_image(f"{tag_prefix}_generated", gen_grid, step=g_it)
            return

        # For each control type, build side-by-side panel
        for ctrl_type, ctrl_grid in control_grids.items():
            ctrl_grid = ctrl_grid.float().clamp(0,1)
            combined = self._combine_side_by_side(ctrl_grid, gen_grid)

            tb_lg.log_image(
                f"{tag_prefix}_{ctrl_type}_control_and_generated",
                combined,
                step=g_it,
            )

    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger,
        control_strength: float = 1.0,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        train_control_only = getattr(self.args, "freeze_switti_backbone", False)

        self.switti.train()

        if train_control_only:
            # Use the unwrapped module for attribute checks and direct submodule .train()
            if hasattr(self.switti_wo_ddp, "control_encoder") and self.switti_wo_ddp.control_encoder is not None:
                # put the unwrapped control encoder into train mode (important when wrapped)
                self.switti_wo_ddp.control_encoder.train()


        for accum_iter in range(self.grad_accum):
            batch = next(self.dataloader)
            if len(batch) == 3:
                image, prompt, orig_size = batch
                control_dict = None
            else:
                image, prompt, control_dict, orig_size = batch

            batch_height = [h for (w, h) in orig_size]
            batch_width = [w for (w, h) in orig_size]

            if control_dict is not None:
                processed = {}
                for k, v in control_dict.items():
                    if v is None: 
                        processed[k] = None
                    else:
                        v_device = v.to(self.device, non_blocking=True).to(self.model_dtype)
                        # SCALE BY CONTROL STRENGTH (element-wise multiplication)
                        processed[k] = v_device * control_strength
                control_dict = processed

            inp_B3HW = image.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW, size=(self.resos[-1], self.resos[-1]), mode="bicubic",
            )

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(
                inp_B3HW, noise_std=self.embed_noise_std
            )
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_switti_input(gt_idx_Bl)
            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(
                    torch.full((B, ), self.args.uncond_proba)
                )
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        prompt[i_] = ""
            (prompt_embeds,
             pooled_prompt_embeds,
             prompt_attn_bias,
             ) = self.pipe.encode_prompt(prompt, encode_null=False)

            with self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_height,
                    batch_width=batch_width,
                    control_dict=control_dict
                )
                loss = self.train_loss(logits_BLV.view(-1, V),
                                       gt_BL.view(-1),
                                       ).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()  / self.grad_accum 

                # --- Gate regularisation (optional) ---
                if self.use_control_gate:
                    gate_penalty_val = 0.0          # scalar for logging
                    if self.use_gate_reg:
                        gate_penalty = compute_gate_regularization(
                            self.switti_wo_ddp,
                            gate_target=self.gate_reg_target,
                        )
                        loss = loss + self.gate_reg_weight * gate_penalty
                        gate_penalty_val = gate_penalty.item()

                # --- Perceptual loss (optional) ---
                perceptual_val = 0.0
                if self.use_perceptual_loss and (g_it % self.perceptual_loss_every_n_steps == 0):
                    perceptual_loss = self.perceptual_loss_fn(
                        logits_BLV, gt_idx_Bl, self.vae_local
                    )
                    loss = loss + self.perceptual_loss_weight * perceptual_loss
                    perceptual_val = perceptual_loss.item()

            # backward
            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss,
                is_stepping=is_stepping,
                )

        # log to tensorboard
        if g_it > 0 and g_it % self.args.log_iters == 0:
            # recalculate logits in .eval() mode to log acc
            self.switti.eval()
            if self.args.use_gradient_checkpointing:
                self.switti.disable_gradient_checkpointing()
            with torch.no_grad(), self.optimizer.amp_ctx:
                logits_BLV = self.switti(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_height=batch_height,
                    batch_width=batch_width,
                    control_dict=control_dict
                )

            # Compute cluster usage
            pred_BL = logits_BLV.data.argmax(dim=-1)
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float().cuda()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (
                prob_per_class_is_chosen > 0.001 / V
            ).float().mean().item() * 100

            logits_lg = dict()
            kw = dict(z_voc_usage=cluster_usage, acc_total=0.0, L_total=0.0)
            for si, (bg, ed) in enumerate(self.begin_ends):
                pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                tar = gt_BL[:, bg:ed].reshape(-1)
                top5 = torch.topk(pred, 5, dim=-1)[1]

                acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                acc_top5 = torch.eq(tar[:, None], top5).any(dim=1).float().mean().item() * 100
                ce = self.val_loss(pred, tar).item()
                std = pred.std(dim=-1).mean().item()
                norm = pred.norm(dim=-1).mean().item()

                stats = torch.tensor([acc, acc_top5, ce, std, norm], device=dist.get_device())
                dist.allreduce(stats)
                stats /= dist.get_world_size()
                acc, acc_top5, ce, std, norm = stats.tolist()

                logits_lg[f"logits_std_{self.resos[si]}"] = std
                logits_lg[f"logits_norm_{self.resos[si]}"] = norm
                kw[f"acc_{self.resos[si]}"] = acc
                kw[f"acc_top5_{self.resos[si]}"] = acc_top5
                kw[f"L_{self.resos[si]}"] = ce
                kw[f"acc_total"] += acc / len(self.begin_ends)
                kw[f"L_total"] += ce / len(self.begin_ends)

            if g_it % self.args.log_images_iters == 0:
                with FSDP.summon_full_params(self.switti, writeback=False), torch.no_grad(), self.optimizer.amp_ctx:
                    torch.cuda.empty_cache()
                    for cfg in [6]: # SAAD: add o
                        subprompt = prompt[:16]
                        if control_dict and any(v is not None for v in control_dict.values()):
                            first_valid = next(v for v in control_dict.values() if v is not None)
                            n_show = min(len(subprompt), first_valid.shape[0])
                        else:
                            n_show = len(subprompt)
                        ctrl_for_pipe = self._build_ctrl_for_pipe(control_dict, n_show)
                        imgs = self.pipe(subprompt,
                                         cfg=cfg,
                                         top_k=self.args.top_k,
                                         top_p=self.args.top_p,
                                         return_pil=False,
                                         control_dict=ctrl_for_pipe,
                                         )
                        imgs = make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))
                        self._log_pipe_outputs(
                            tb_lg,
                            tag_prefix=f"train_topk={self.args.top_k}_topp={self.args.top_p}_cfg={cfg}",
                            imgs=imgs,
                            control_dict=ctrl_for_pipe,
                            g_it=g_it,
                            n_show=n_show,
                        )

                        imgs = self.pipe(
                            prompt=self.log_prompts,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                            control_dict=self.log_control_dict,
                        )
                        imgs = make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))
                        self._log_pipe_outputs(
                            tb_lg,
                            tag_prefix=f"eval_topk={self.args.top_k}_topp={self.args.top_p}_cfg={cfg}",
                            imgs=imgs,
                            control_dict=self.log_control_dict,
                            g_it=g_it,
                        )

                        # imgs = self.pipe(
                        #     prompt=self.log_prompts,
                        #     top_k=1,
                        #     cfg=cfg,
                        #     return_pil=False,
                        #     control_dict=self.log_control_dict,
                        # )
                        # imgs = make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))
                        # self._log_pipe_outputs(
                        #     tb_lg,
                        #     tag_prefix=f"eval_topk1_cfg={cfg}",
                        #     imgs=imgs,
                        #     control_dict=self.log_control_dict,
                        #     g_it=g_it,
                        # )

                        # NEW: Log MJHQ T2I samples
                        imgs = self.pipe(
                            prompt=self.mjhq_prompts,
                            cfg=cfg,
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            return_pil=False,
                            control_dict=None,  # T2I, no control
                        )
                        imgs = make_grid(imgs, nrow=math.floor(math.sqrt(len(imgs))))
                        self._log_pipe_outputs(
                            tb_lg,
                            tag_prefix=f"mjhq_t2i_samples_cfg_topk={self.args.top_k}_topp={self.args.top_p}_cfg={cfg}",
                            imgs=imgs,
                            control_dict=None,
                            g_it=g_it,
                        )
                        # Log MJHQ prompts
                        prompt_text = "\n".join([f"{i}: {p}" for i, p in enumerate(self.mjhq_prompts)])
                        tb_lg.log_text(
                            f"mjhq_t2i_prompts_cfg={cfg}",
                            prompt_text,
                            step=g_it
                        )
                        # Log COCO eval prompts
                        prompt_text = "\n".join([f"{i}: {p}" for i, p in enumerate(self.log_prompts)])
                        tb_lg.log_text(
                            f"coco_eval_prompts_cfg={cfg}",
                            prompt_text,
                            step=g_it
                        )

                        del imgs

            if dist.is_master():
                tb_lg.update(head="Logits_stats", **logits_lg, step=g_it)
                tb_lg.update(head="AR_iter_loss", **kw, step=g_it)

                # Log control gate values
                if hasattr(self.switti_wo_ddp, 'blocks'):
                    gate_values = []
                    for i, block in enumerate(self.switti_wo_ddp.blocks):
                        if hasattr(block, 'control_gate') and block.control_gate is not None:
                            gate = torch.sigmoid(block.control_gate).item()
                            gate_values.append(gate)
                    
                    if gate_values:
                        tb_lg.update(
                            head="Control_gates",
                            mean=sum(gate_values) / len(gate_values),
                            min=min(gate_values),
                            max=max(gate_values),
                            step=g_it
                        )

                # Log auxiliary losses
                if self.use_control_gate and self.use_gate_reg:
                    tb_lg.update(
                        head="Auxiliary_losses",
                        gate_penalty=gate_penalty_val,
                        step=g_it,
                    )
                if self.use_perceptual_loss:
                    tb_lg.update(
                        head="Auxiliary_losses",
                        perceptual_lpips=perceptual_val,
                        step=g_it,
                    )
            print(f"LOGGING {g_it} FINISHED")
            if self.args.use_gradient_checkpointing:
                self.switti.enable_gradient_checkpointing()
            self.switti.train()
            dist.barrier()

        return grad_norm.item(), scale_log2

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }
