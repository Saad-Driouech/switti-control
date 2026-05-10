"""
generate_targeted.py — Generate images for specific sample indices only.
No metric computation. Fast alternative to generate.py for small figure sets.

Modes
-----
  cfg_sweep     : V2 approach at multiple CFG values for selected samples.
  null_control  : Any approach with null/zero control for selected samples.
                  Used for T2I quality comparison figure.

Works on both branches:
  spatial-control-v2.0  →  SwittiControlPipeline
  spatial-control        →  build_models (V1)

Output layout
-------------
  cfg_sweep:
    <out_dir>/cfg_sweep/<run_name>_cfg<cfg>/images/<idx:05d>.jpg
    <out_dir>/cfg_sweep/<run_name>_cfg<cfg>/images/<idx:05d>_ctrl.png
    (ctrl.png is copied from cfg6 run if it exists, else regenerated)

  null_control:
    <out_dir>/null_control/<run_name>/images/<idx:05d>.jpg

Usage
-----
  # CFG sweep — run from spatial-control-v2.0 branch:
  python scripts/generate_targeted.py \\
      --config scripts/eval_config_v2.yaml \\
      --mode cfg_sweep \\
      --run_names approach2_canny approach2_seg \\
      --cfg_values 3.0 4.5 7.5 9.0 \\
      --samples_per_run '{"approach2_canny": [1], "approach2_seg": [28]}' \\
      --out_dir /home/woody/iwnt/iwnt134h/switti/eval_targeted/

  # Null control — run from the appropriate branch for each approach:
  python scripts/generate_targeted.py \\
      --config scripts/eval_config_v2.yaml \\
      --mode null_control \\
      --run_names approach2_canny \\
      --samples 47 88 116 28 83 111 \\
      --out_dir /home/woody/iwnt/iwnt134h/switti/eval_targeted/
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from calculate_metrics import to_PIL_image

# ---------------------------------------------------------------------------
# Branch detection — import the right pipeline
# ---------------------------------------------------------------------------

try:
    from models.control_pipeline import SwittiControlPipeline
    _BRANCH = "v2"
except ImportError:
    _BRANCH = "v1"

print(f"[branch] detected: {_BRANCH}")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _strip_prefix(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_fully_sharded_module.", "")
        k = re.sub(r"^module\.", "", k)
        cleaned[k] = v
    return cleaned


def _build_pipe_v2(run: dict, cfg: dict):
    reso = cfg.get("reso", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = SwittiControlPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.get("pretrained_switti", "yresearch/Switti"),
        control_ckpt=run.get("ckpt"),
        torch_dtype=torch.float32,
        device=device,
        reso=reso,
        num_modalities=run.get("num_modalities", 6),
    )
    pipe.control_net.eval()
    return pipe, device


def _build_pipe_v1(run: dict, cfg: dict):
    from models import build_models, VQVAEHF
    from models.switti import SwittiHF
    from utils.arg_util import RESOLUTION_PATCH_NUMS_MAPPING

    reso = cfg.get("reso", 512)
    patch_nums = tuple(int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[reso].split("_"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_local, switti, pipe = build_models(
        device=device,
        patch_nums=patch_nums,
        depth=cfg.get("depth", 30),
        use_swiglu_ffn=True,
        use_crop_cond=True,
        control_encoder_type=run.get("control_encoder_type"),
        control_context_dim=run.get("control_context_dim", 384),
        control_fusion=run.get("control_fusion", "cross"),
        control_pretrained=run.get("control_pretrained", True),
        control_encoder_ckpt=run.get("control_encoder_ckpt"),
    )

    base = SwittiHF.from_pretrained(cfg.get("pretrained_switti", "yresearch/Switti"))
    switti.load_state_dict(base.state_dict(), strict=False)
    del base

    vae_local = VQVAEHF.from_pretrained(
        cfg.get("vae_ckpt", "yresearch/VQVAE-Switti"), reso=reso
    ).to(device)
    pipe.vae = vae_local

    ckpt_path = run.get("ckpt")
    if ckpt_path:
        sd = _strip_prefix(torch.load(ckpt_path, map_location="cpu"))
        switti.load_state_dict(sd, strict=False)
        print(f"  [ckpt] loaded {ckpt_path}")

    switti.eval()
    vae_local.eval()
    return pipe, device


def _build_pipe(run: dict, cfg: dict):
    if _BRANCH == "v2":
        return _build_pipe_v2(run, cfg)
    else:
        return _build_pipe_v1(run, cfg)


# ---------------------------------------------------------------------------
# Control image loading
# ---------------------------------------------------------------------------

def _load_ctrl_v2(fname: str, modality: str, control_path: str, reso: int) -> torch.Tensor:
    mid_reso = round(1.125 * reso)
    transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(reso),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    fname_png = fname.replace(".jpg", ".png")
    ctrl_fp = os.path.join(control_path, modality, fname_png)
    if os.path.exists(ctrl_fp):
        return transform(Image.open(ctrl_fp).convert("RGB"))
    return torch.zeros(3, reso, reso)


def _load_ctrl_v1(fname: str, modality: str, control_path: str, reso: int) -> torch.Tensor:
    from utils.data import JointTransform
    transform = JointTransform(final_reso=reso, mid_reso=1.125, hflip_prob=0.0)
    fname_png = fname.replace(".jpg", ".png")
    ctrl_fp = os.path.join(control_path, modality, fname_png)
    if os.path.exists(ctrl_fp):
        img = Image.open(ctrl_fp).convert("RGB")
        _, processed = transform(img, {modality: img})
        return processed[modality]
    return torch.zeros(3, reso, reso)


def _load_ctrl(fname, modality, control_path, reso):
    if _BRANCH == "v2":
        return _load_ctrl_v2(fname, modality, control_path, reso)
    return _load_ctrl_v1(fname, modality, control_path, reso)


# ---------------------------------------------------------------------------
# Inference call
# ---------------------------------------------------------------------------

def _infer(pipe, texts: list, ctrl_tensor, modality: str, cfg_val: float,
           seed: int, top_k: int, top_p: float, device: str,
           control_end_si: int = 8, null_control: bool = False):
    with torch.no_grad():
        if _BRANCH == "v2":
            ctrl_input = None if null_control else ctrl_tensor
            return pipe(
                prompt=texts,
                ctrl_image=ctrl_input,
                modality=modality,
                seed=seed,
                cfg=cfg_val,
                top_k=top_k,
                top_p=top_p,
                return_pil=False,
            )
        else:
            # V1
            ctrl_dict = None if null_control else {modality: ctrl_tensor}
            kwargs = dict(
                prompt=texts,
                seed=seed,
                cfg=cfg_val,
                top_k=top_k,
                top_p=top_p,
                more_smooth=False,
                return_pil=False,
            )
            if ctrl_dict is not None:
                kwargs["control_dict"] = ctrl_dict
                kwargs["control_end_si"] = control_end_si
            return pipe(**kwargs)


# ---------------------------------------------------------------------------
# CFG sweep
# ---------------------------------------------------------------------------

def run_cfg_sweep(cfg: dict, run: dict, cfg_values: list, sample_indices: list,
                  out_dir: Path, existing_cfg6_dir: Path | None):
    reso = cfg.get("reso", 512)
    modality = run["modality"]
    control_path = cfg.get("control_path")
    seed = cfg.get("seed", 42)
    top_k = cfg.get("top_k", 400)
    top_p = cfg.get("top_p", 0.95)

    df = pd.read_csv(cfg["coco_csv"])
    # Only keep the requested indices
    rows = df.iloc[sample_indices]

    pipe, device = _build_pipe(run, cfg)

    for cfg_val in cfg_values:
        run_dir = out_dir / f"{run['name']}_cfg{cfg_val:g}" / "images"
        run_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in zip(sample_indices, rows.itertuples()):
            out_jpg = run_dir / f"{idx:05d}.jpg"
            out_ctrl = run_dir / f"{idx:05d}_ctrl.png"
            if out_jpg.exists():
                print(f"  [skip] {out_jpg}")
                continue

            fname = str(row.file_name)
            caption = str(row.captions)

            # Control image
            ctrl_t = _load_ctrl(fname, modality, control_path, reso)
            ctrl_batch = ctrl_t.unsqueeze(0).to(device)

            img_tensors = _infer(pipe, [caption], ctrl_batch, modality,
                                 cfg_val, seed, top_k, top_p, device)
            to_PIL_image(img_tensors[0]).save(out_jpg, quality=95)

            # Save ctrl.png — copy from cfg6 if exists, else save from tensor
            if existing_cfg6_dir and not out_ctrl.exists():
                src_ctrl = existing_cfg6_dir / "images" / f"{idx:05d}_ctrl.png"
                if src_ctrl.exists():
                    shutil.copy(src_ctrl, out_ctrl)
            if not out_ctrl.exists():
                ctrl_pil = Image.fromarray(
                    (((ctrl_t + 1) / 2).clamp(0, 1)
                     .permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                ctrl_pil.save(out_ctrl)

            print(f"  [cfg={cfg_val:g}] sample {idx} -> {out_jpg}")

    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Null control
# ---------------------------------------------------------------------------

def run_null_control(cfg: dict, run: dict, sample_indices: list, out_dir: Path):
    reso = cfg.get("reso", 512)
    modality = run.get("modality", "canny")
    seed = cfg.get("seed", 42)
    top_k = cfg.get("top_k", 400)
    top_p = cfg.get("top_p", 0.95)
    control_end_si = cfg.get("control_end_si", 8)

    df = pd.read_csv(cfg["coco_csv"])
    rows = df.iloc[sample_indices]

    pipe, device = _build_pipe(run, cfg)

    run_dir = out_dir / run["name"] / "images"
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in zip(sample_indices, rows.itertuples()):
        out_jpg = run_dir / f"{idx:05d}.jpg"
        if out_jpg.exists():
            print(f"  [skip] {out_jpg}")
            continue

        caption = str(row.captions)
        img_tensors = _infer(pipe, [caption], None, modality,
                             cfg.get("guidance", 6.0), seed, top_k, top_p, device,
                             control_end_si=control_end_si, null_control=True)
        to_PIL_image(img_tensors[0]).save(out_jpg, quality=95)
        print(f"  [null] sample {idx} ({caption[:60]}) -> {out_jpg}")

    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["cfg_sweep", "null_control"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_names", nargs="+", default=None,
                    help="Run name(s) to process (default: all runs in config)")
    # CFG sweep args
    ap.add_argument("--cfg_values", nargs="+", type=float,
                    default=[3.0, 4.5, 7.5, 9.0],
                    help="CFG values to sweep (cfg6 already exists, skip it)")
    ap.add_argument("--samples_per_run", type=str, default=None,
                    help='JSON dict mapping run_name -> [sample_indices], '
                         'e.g. \'{"approach2_canny": [1], "approach2_seg": [28]}\'')
    # Null control args
    ap.add_argument("--samples", nargs="+", type=int,
                    default=[47, 88, 116, 28, 83, 111],
                    help="Sample indices for null control generation")
    # Optional: path to existing cfg6 eval dir for copying ctrl.png
    ap.add_argument("--existing_eval_dir", default=None,
                    help="Path to existing eval_v100 out_dir to copy ctrl.png from")
    ap.add_argument("--subset_csv", default=None,
                    help="Path to subset.csv from the eval run. If provided, sample indices "
                         "refer to rows in this file instead of the full coco_csv.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.subset_csv:
        cfg["coco_csv"] = args.subset_csv

    out_dir = Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = cfg["runs"]
    if args.run_names:
        runs = [r for r in runs if r["name"] in args.run_names]
    if not runs:
        raise ValueError(f"No matching runs found. Available: {[r['name'] for r in cfg['runs']]}")

    if args.mode == "cfg_sweep":
        samples_per_run = json.loads(args.samples_per_run) if args.samples_per_run else {}
        for run in runs:
            idxs = samples_per_run.get(run["name"], args.samples)
            existing = None
            if args.existing_eval_dir:
                existing = Path(args.existing_eval_dir) / f"{run['name']}_cfg6"
            print(f"\n=== cfg_sweep: {run['name']} | samples={idxs} | cfgs={args.cfg_values}")
            run_cfg_sweep(cfg, run, args.cfg_values, idxs, out_dir, existing)

    elif args.mode == "null_control":
        for run in runs:
            print(f"\n=== null_control: {run['name']} | samples={args.samples}")
            run_null_control(cfg, run, args.samples, out_dir)

    print(f"\n[done] Output: {out_dir}")


if __name__ == "__main__":
    main()
