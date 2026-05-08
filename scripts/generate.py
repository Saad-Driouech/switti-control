"""
Inference-only: generate images for one (run, cfg) combination and save to disk.
Decouple inference from metric computation so each step can run independently.

Output layout:
    <out_dir>/subset.csv                         — shared, created once
    <out_dir>/<run_name>_cfg<cfg>/images/        — generated JPEGs (named by global index)
                                                   + matching _ctrl.png for each

Usage (single GPU):
    python scripts/generate.py \\
        --config scripts/eval_config.yaml \\
        --run_name approach1_canny --cfg 6.0 \\
        --out_dir /path/to/eval_out/

Usage (multi-GPU DDP):
    torchrun --standalone --nproc_per_node=4 scripts/generate.py \\
        --config scripts/eval_config.yaml \\
        --run_name approach1_canny --cfg 6.0 \\
        --out_dir /path/to/eval_out/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
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

    modality = run.get("modality")
    args = _make_args(cfg, n_final, modality, guidance=args_cli.cfg)
    pipe = _build_pipe(run, cfg)
    pipe.switti.eval()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = args.eval_batch_size

    rank_caption_batches, rank_filename_batches = prepare_prompts(
        str(subset_csv), batch_size, n_final
    )

    control_path = cfg.get("control_path")

    # JointTransform matches training: Resize(mid_reso, LANCZOS) + CenterCrop
    ctrl_transform = None
    if control_path and modality:
        from utils.data import JointTransform
        ctrl_transform = JointTransform(
            final_reso=args.data_load_reso,
            mid_reso=args.mid_reso,
            hflip_prob=0.0,
        )

    print(f"[rank {rank}] generating {sum(len(b) for b in rank_caption_batches)} images...")

    for lb, (captions_batch, filenames_batch) in enumerate(
        tqdm(zip(rank_caption_batches, rank_filename_batches),
             total=len(rank_caption_batches), disable=(rank != 0))
    ):
        global_start = (lb * world_size + rank) * batch_size
        texts = [str(c) for c in captions_batch]

        ctrl_dict_batch = None
        if ctrl_transform is not None:
            ctrl_tensors = []
            for fname in filenames_batch:
                fname = str(fname)
                fname_png = fname.replace(".jpg", ".png")
                if fname in ("None", "nan", ""):
                    t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                else:
                    ctrl_fp = os.path.join(control_path, modality, fname_png)
                    if os.path.exists(ctrl_fp):
                        try:
                            img = Image.open(ctrl_fp).convert("RGB")
                            _, processed = ctrl_transform(img, {modality: img})
                            t = processed[modality]
                        except Exception as e:
                            print(f"[warn] {ctrl_fp}: {e}")
                            t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                    else:
                        t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                ctrl_tensors.append(t)
            ctrl_dict_batch = {modality: torch.stack(ctrl_tensors)}

        with torch.no_grad():
            image_tensors = pipe(
                prompt=texts,
                seed=args.seed,
                cfg=args_cli.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=False,
                return_pil=False,
                control_dict=ctrl_dict_batch,
                control_end_si=args.control_end_si,
            )

        for j, t in enumerate(image_tensors):
            global_idx = global_start + j
            to_PIL_image(t).save(images_dir / f"{global_idx:05d}.jpg", quality=95)
            if ctrl_dict_batch is not None:
                ctrl_t = ctrl_dict_batch[modality][j]
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
