"""
ControlAR wrapper for benchmark_speed.py — text-conditional (t2i) variant.

ControlAR is a text-to-image model built on LlamaGen-XL that adds spatial
control via conditional decoding. It uses a T5 (flan-t5-xl) text encoder,
NOT class labels.

Cluster paths (already downloaded):
  controlar_repo : /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR
  vq_ckpt        : checkpoints/vq/vq_ds16_t2i.pt
  gpt_ckpt       : checkpoints/llamagen/t2i_XL_stage2_512.pt
  t5_path        : checkpoints/t5-ckpt
  control_ckpt   : checkpoints/t2i/canny/canny_MR.safetensors
                   checkpoints/t2i/hed/hed.safetensors
                   checkpoints/t2i/depth/depth_MR.safetensors
                   checkpoints/t2i/seg/seg_cocostuff.safetensors

Config entry example:
  - name: ControlAR (canny, t2i, 512px)
    type: custom
    wrapper: scripts/wrappers/controlar_wrapper.py
    modality: canny
    controlar_repo: /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR
    vq_ckpt:      /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR/checkpoints/vq/vq_ds16_t2i.pt
    gpt_ckpt:     /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR/checkpoints/llamagen/t2i_XL_stage2_512.pt
    control_ckpt: /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR/checkpoints/t2i/canny/canny_MR.safetensors
    t5_path:      /home/hpc/iwnt/iwnt134h/thesis/repos/ControlAR/checkpoints/t5-ckpt/checkpoints/t5-ckpt
    t5_model_type: flan-t5-xl
    image_size: 512
    cfg_scale: 7.5
    top_k: 2000
    top_p: 1.0
    temperature: 1.0
    control_strength: 1.0
    cls_token_num: 120
"""

import os
import sys

import numpy as np
import torch
from PIL import Image


def load_pipe(run_cfg: dict):
    repo = run_cfg["controlar_repo"]
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # dinov2_adapter.py uses a relative path for from_pretrained — must run from repo root
    _orig_cwd = os.getcwd()
    os.chdir(repo)

    from tokenizer.tokenizer_image.vq_model import VQ_models
    from autoregressive.models.gpt_t2i import GPT_models
    from language.t5 import T5Embedder

    device = torch.device("cuda")
    precision = torch.bfloat16
    image_size = run_cfg.get("image_size", 512)
    downsample_size = 16  # VQ-16
    latent_size = image_size // downsample_size

    # VQ tokenizer
    vq_model = VQ_models["VQ-16"](
        codebook_size=16384,
        codebook_embed_dim=8,
    ).to(device).eval()
    ckpt = torch.load(run_cfg["vq_ckpt"], map_location="cpu")
    vq_model.load_state_dict(ckpt["model"])
    del ckpt

    # GPT (t2i)
    cls_token_num = run_cfg.get("cls_token_num", 120)
    gpt_model = GPT_models["GPT-XL"](
        block_size=latent_size ** 2,
        cls_token_num=cls_token_num,
        model_type="t2i",
        condition_type=run_cfg.get("modality", "canny"),
    ).to(device=device, dtype=precision).eval()

    # Load base t2i weights
    base_ckpt = torch.load(run_cfg["gpt_ckpt"], map_location="cpu")
    base_sd = base_ckpt.get("model", base_ckpt.get("module", base_ckpt))
    gpt_model.load_state_dict(base_sd, strict=False)
    del base_ckpt

    # Load control adapter weights (safetensors)
    control_ckpt = run_cfg.get("control_ckpt")
    if control_ckpt and os.path.exists(control_ckpt):
        from safetensors.torch import load_file
        control_sd = load_file(control_ckpt)
        gpt_model.load_state_dict(control_sd, strict=False)

    # T5 text encoder
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=run_cfg["t5_path"],
        dir_or_name=run_cfg.get("t5_model_type", "flan-t5-xl"),
        torch_dtype=precision,
        model_max_length=cls_token_num,
    )

    os.chdir(_orig_cwd)

    return {
        "vq": vq_model,
        "gpt": gpt_model,
        "t5": t5_model,
        "device": device,
        "precision": precision,
        "image_size": image_size,
        "downsample_size": downsample_size,
        "cls_token_num": cls_token_num,
    }


def generate(pipe: dict, prompt: str, ctrl_pil: Image.Image, run_cfg: dict):
    from autoregressive.models.generate import generate as ar_generate

    vq_model = pipe["vq"]
    gpt_model = pipe["gpt"]
    t5_model = pipe["t5"]
    device = pipe["device"]
    precision = pipe["precision"]
    image_size = pipe["image_size"]
    downsample_size = pipe["downsample_size"]
    cls_token_num = pipe["cls_token_num"]

    cfg_scale = run_cfg.get("cfg_scale", 7.5)
    temperature = run_cfg.get("temperature", 1.0)
    top_k = run_cfg.get("top_k", 2000)
    top_p = run_cfg.get("top_p", 1.0)
    control_strength = run_cfg.get("control_strength", 1.0)

    # Control image: resize, convert to [-1, 1], duplicate for CFG
    ctrl = ctrl_pil.resize((image_size, image_size)).convert("RGB")
    ctrl_t = torch.from_numpy(np.array(ctrl, dtype=np.float32) / 255.0)
    ctrl_t = ctrl_t.permute(2, 0, 1).unsqueeze(0)          # (1, 3, H, W)
    ctrl_t = 2.0 * ctrl_t - 1.0                             # [-1, 1]
    ctrl_t = ctrl_t.repeat(2, 1, 1, 1).to(device, dtype=precision)  # (2, 3, H, W)

    # Text embeddings (duplicate for CFG: [cond, uncond])
    prompts = [prompt, ""]
    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    # Left-padding (matches official sample_t2i.py)
    new_caption_embs = []
    for caption_emb, emb_mask in zip(caption_embs, emb_masks):
        valid_num = int(emb_mask.sum().item())
        new_caption_embs.append(
            torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
        )
    new_caption_embs = torch.stack(new_caption_embs)
    new_emb_masks = torch.flip(emb_masks, dims=[-1])

    c_indices = new_caption_embs * new_emb_masks[:, :, None]
    c_emb_masks = new_emb_masks

    latent_size = image_size // downsample_size
    qzshape = [2, 8, latent_size, latent_size]

    index_sample = ar_generate(
        gpt_model,
        c_indices,
        latent_size ** 2,
        c_emb_masks,
        condition=ctrl_t,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
        control_strength=control_strength,
    )
    vq_model.decode_code(index_sample, qzshape)
