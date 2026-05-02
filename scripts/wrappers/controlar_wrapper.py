"""
ControlAR wrapper for benchmark_speed.py.

Before using this wrapper:
  1. Clone the ControlAR repo:
       git clone https://github.com/hustvl/ControlAR.git /path/to/ControlAR

  2. Download the ControlAR checkpoint and tokenizer from their HuggingFace
     release (https://huggingface.co/hustvl/ControlAR) and note the paths.

  3. Add ControlAR to PYTHONPATH:
       export PYTHONPATH="/path/to/ControlAR:$PYTHONPATH"

  4. Point the config at this file:
       - name: ControlAR (canny, 256-token LlamaGen-XL)
         type: custom
         wrapper: scripts/wrappers/controlar_wrapper.py
         modality: canny
         # ControlAR-specific fields (read by load_pipe below):
         controlar_repo: /path/to/ControlAR
         vq_ckpt: /path/to/vq_ds16_c2i.pt
         gpt_ckpt: /path/to/controlar_canny_xl.pt
         gpt_model: GPT-XL            # GPT-B / GPT-L / GPT-XL
         image_size: 256              # 256 or 512
         num_classes: 1000            # ImageNet classes (not used for T2I, set to 1000)
         cfg_scale: 4.0
         top_k: 2000
         top_p: 1.0
         temperature: 1.0
         num_sampling_steps: 256      # == image_size^2 / vq_stride^2
         class_label: 207             # golden retriever; replace with target class

This wrapper assumes the standard ControlAR model (class-conditional,
ImageNet), which is the publicly released checkpoint. If you have a
text-conditional variant, adjust load_pipe / generate accordingly.
"""

import os
import sys

import torch
import numpy as np
from PIL import Image


def load_pipe(run_cfg: dict):
    """Build and return (vq_model, gpt_model, device) tuple."""
    repo = run_cfg.get("controlar_repo", "")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)

    # ControlAR imports (available after PYTHONPATH is set)
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from autoregressive.models.gpt import GPT_models

    device = torch.device("cuda")
    image_size = run_cfg.get("image_size", 256)
    vq_model_name = run_cfg.get("vq_model", "VQ-16")

    # Build and load VQ tokenizer
    vq_model = VQ_models[vq_model_name](
        codebook_size=16384,
        codebook_embed_dim=8,
    ).to(device)
    vq_model.eval()
    vq_ckpt = run_cfg.get("vq_ckpt")
    if vq_ckpt:
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])

    # Build and load GPT (ControlAR)
    gpt_model_name = run_cfg.get("gpt_model", "GPT-XL")
    num_classes = run_cfg.get("num_classes", 1000)
    gpt_model = GPT_models[gpt_model_name](
        vocab_size=16384,
        block_size=256,          # 256 tokens for 256x256; 1024 for 512x512
        num_classes=num_classes,
        cls_token_num=1,
        model_type="c2i",
        condition_type=run_cfg.get("modality", "canny"),
    ).to(device)
    gpt_model.eval()
    gpt_ckpt = run_cfg.get("gpt_ckpt")
    if gpt_ckpt:
        checkpoint = torch.load(gpt_ckpt, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        gpt_model.load_state_dict(state_dict, strict=False)

    return {"vq": vq_model, "gpt": gpt_model, "device": device}


def generate(pipe: dict, prompt: str, ctrl_pil: Image.Image, run_cfg: dict):
    """
    One forward pass.  `prompt` is ignored (model is class-conditional);
    class_label from run_cfg is used instead.
    """
    from autoregressive.models.gpt import GPT_models

    vq_model = pipe["vq"]
    gpt_model = pipe["gpt"]
    device = pipe["device"]

    image_size = run_cfg.get("image_size", 256)
    class_label = run_cfg.get("class_label", 207)
    cfg_scale = run_cfg.get("cfg_scale", 4.0)
    top_k = run_cfg.get("top_k", 2000)
    top_p = run_cfg.get("top_p", 1.0)
    temperature = run_cfg.get("temperature", 1.0)
    num_sampling_steps = run_cfg.get("num_sampling_steps", image_size * image_size // 256)

    # Preprocess control image
    ctrl = ctrl_pil.resize((image_size, image_size)).convert("RGB")
    ctrl_t = torch.from_numpy(np.array(ctrl, dtype=np.float32) / 255.0)
    ctrl_t = ctrl_t.permute(2, 0, 1).unsqueeze(0).to(device)

    c_indices = torch.tensor([class_label], device=device)
    # Unconditional token for CFG
    uc_indices = torch.tensor([1000], device=device)  # null class

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Standard ControlAR sampling loop: the model generates tokens
        # conditioned on (class, control_image) via its generate() method.
        index_sample = gpt_model.generate(
            cond=c_indices,
            cond_null=uc_indices,
            condition_image=ctrl_t,
            max_new_tokens=num_sampling_steps,
            emb_masks=None,
            cfg_scale=cfg_scale,
            cfg_interval=[-1, num_sampling_steps],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
        )
        # Decode tokens → image
        vq_model.decode_code(index_sample, shape=(1, 8, image_size // 16, image_size // 16))
