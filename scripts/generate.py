"""
Inference-only: generate images for one (run, cfg) combination and save to disk.
Decouple inference from metric computation so each step can run independently.

Output layout:
    <out_dir>/subset.csv                         — shared, created once
    <out_dir>/<run_name>_cfg<cfg>/images/        — generated JPEGs (named by global index)

Usage (single GPU):
    python scripts/generate.py \\
        --config scripts/eval_config.yaml \\
        --run_name v2_depth --cfg 6.0 \\
        --out_dir /path/to/eval_out/

Usage (multi-GPU DDP):
    torchrun --standalone --nproc_per_node=4 scripts/generate.py \\
        --config scripts/eval_config.yaml \\
        --run_name v2_depth --cfg 6.0 \\
        --out_dir /path/to/eval_out/
"""

import argparse
import os
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

import dist
from calculate_metrics import prepare_prompts, to_PIL_image
from scripts.evaluate import _build_pipe, _build_subset_csv, _make_args


def _safe_barrier():
    if dist.initialized():
        torch.distributed.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True,
                    help="Name matching a run entry in config['runs']")
    ap.add_argument("--cfg", type=float, required=True,
                    help="CFG guidance value")
    ap.add_argument("--out_dir", required=True)
    args_cli = ap.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.initialize()

    out_dir = Path(args_cli.out_dir)
    if dist.is_master():
        out_dir.mkdir(parents=True, exist_ok=True)
    _safe_barrier()

    run = next((r for r in cfg["runs"] if r["name"] == args_cli.run_name), None)
    if run is None:
        raise ValueError(
            f"Run '{args_cli.run_name}' not found. "
            f"Available: {[r['name'] for r in cfg['runs']]}"
        )

    # Build subset CSV (idempotent — skip if already exists)
    subset_csv = out_dir / "subset.csv"
    if dist.is_master() and not subset_csv.exists():
        n_final = _build_subset_csv(
            full_csv=cfg["coco_csv"],
            n=cfg["num_samples"],
            seed=cfg.get("seed", 42),
            eval_batch_size=cfg.get("eval_batch_size", 4),
            out_path=str(subset_csv),
        )
        print(f"[subset] wrote {n_final} rows -> {subset_csv}")
    _safe_barrier()
    n_final = len(pd.read_csv(subset_csv))

    result_name = f"{args_cli.run_name}_cfg{args_cli.cfg:g}"
    images_dir = out_dir / result_name / "images"
    if dist.is_master():
        images_dir.mkdir(parents=True, exist_ok=True)
    _safe_barrier()

    args = _make_args(cfg, n_final, guidance=args_cli.cfg)
    pipe = _build_pipe(run, cfg)
    pipe.control_net.eval()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = args.eval_batch_size

    rank_caption_batches, rank_filename_batches = prepare_prompts(
        str(subset_csv), batch_size, n_final
    )

    control_path = cfg.get("control_path")
    modality = run.get("modality")

    mid_reso = round(1.125 * args.data_load_reso)
    ctrl_transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(args.data_load_reso),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]) if control_path is not None else None

    print(f"[rank {rank}] generating {sum(len(b) for b in rank_caption_batches)} images...")

    for lb, (captions_batch, filenames_batch) in enumerate(
        tqdm(zip(rank_caption_batches, rank_filename_batches),
             total=len(rank_caption_batches), disable=(rank != 0))
    ):
        global_start = (lb * world_size + rank) * batch_size
        texts = [str(c) for c in captions_batch]

        ctrl_tensor_batch = None
        if ctrl_transform is not None and modality is not None:
            raw_ctrl = []
            for fname in filenames_batch:
                fname = str(fname)
                fname_png = fname.replace(".jpg", ".png")
                if fname in ("None", "nan", ""):
                    t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                else:
                    ctrl_fp = os.path.join(control_path, modality, fname_png)
                    if os.path.exists(ctrl_fp):
                        try:
                            t = ctrl_transform(Image.open(ctrl_fp).convert("RGB"))
                        except Exception as e:
                            print(f"[warn] {ctrl_fp}: {e}")
                            t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                    else:
                        t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                raw_ctrl.append(t)
            ctrl_tensor_batch = torch.stack(raw_ctrl)

        with torch.no_grad():
            if ctrl_tensor_batch is not None:
                image_tensors = pipe(
                    prompt=texts,
                    ctrl_image=ctrl_tensor_batch,
                    modality=modality,
                    seed=args.seed,
                    cfg=args_cli.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    return_pil=False,
                )
            else:
                image_tensors = pipe(
                    prompt=texts,
                    seed=args.seed,
                    cfg=args_cli.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    return_pil=False,
                )

        for j, t in enumerate(image_tensors):
            global_idx = global_start + j
            to_PIL_image(t).save(images_dir / f"{global_idx:05d}.jpg", quality=95)
            if ctrl_tensor_batch is not None:
                ctrl_t = ctrl_tensor_batch[j]
                ctrl_pil = Image.fromarray(
                    (((ctrl_t + 1) / 2).clamp(0, 1)
                     .permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                ctrl_pil.save(images_dir / f"{global_idx:05d}_ctrl.png")

    _safe_barrier()
    if dist.is_master():
        n_saved = len(list(images_dir.glob("*.jpg")))
        print(f"[done] saved {n_saved} images -> {images_dir}")

    if dist.initialized():
        dist.finalize()


if __name__ == "__main__":
    main()
