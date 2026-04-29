"""
Universal evaluation pipeline for thesis reporting.

Runs every modality/checkpoint listed in a YAML config against the SAME
deterministic subset of COCO val2014 prompts, computes:
  - FID, CLIP score, PickScore, ImageReward
  - control-modality-specific metrics (SSIM, edge IoU, depth/normal/HED/pose/seg)
and writes one row per run to results.jsonl + results.csv.

Usage (single GPU):
    python scripts/evaluate.py --config scripts/eval_config_example.yaml \\
        --out_dir /path/to/eval_run/

Usage (multi-GPU DDP):
    torchrun --standalone --nproc_per_node=4 scripts/evaluate.py \\
        --config scripts/eval_config_example.yaml --out_dir /path/to/eval_run/
"""

import argparse
import gc
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import dist
from calculate_metrics import distributed_metrics_with_csv, to_PIL_image
from models import build_models, VQVAEHF
from models.switti import SwittiHF
from utils.arg_util import RESOLUTION_PATCH_NUMS_MAPPING
from utils.fid_score_in_memory import calculate_fid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fsdp_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_fully_sharded_module.", "")
        k = re.sub(r"^module\.", "", k)
        cleaned[k] = v
    return cleaned


def _safe_barrier():
    if dist.initialized():
        torch.distributed.barrier()


def _build_subset_csv(full_csv: str, n: int, seed: int, eval_batch_size: int,
                      out_path: str) -> int:
    """Sample `n` rows deterministically; truncate to a multiple of
    eval_batch_size * world_size so the existing dataloader assertion passes.
    Returns the final row count."""
    df = pd.read_csv(full_csv)
    assert "file_name" in df.columns and "captions" in df.columns, \
        f"Expected 'file_name' and 'captions' columns in {full_csv}"

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    block = eval_batch_size * dist.get_world_size()
    final_n = (len(df) // block) * block
    if final_n == 0:
        raise ValueError(
            f"Subset size {len(df)} smaller than batch block {block}. "
            f"Reduce eval_batch_size or increase num_samples.")
    df = df.iloc[:final_n].reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return final_n


def _make_args(cfg: dict, num_samples: int, modality: str | None) -> SimpleNamespace:
    """Build the args object that `distributed_metrics_with_csv` reads."""
    reso = cfg.get("reso", 512)
    patch_nums = tuple(int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[reso].split("_"))
    return SimpleNamespace(
        metrics_max_count=num_samples,
        eval_batch_size=cfg.get("eval_batch_size", 4),
        control_types=[modality] if modality else None,
        data_load_reso=reso,
        mid_reso=cfg.get("mid_reso", 1.125),
        num_images_for_metrics=1,
        seed=cfg.get("seed", 42),
        guidance=cfg.get("guidance", 6.0),
        top_k=cfg.get("top_k", 400),
        top_p=cfg.get("top_p", 0.95),
        control_end_si=cfg.get("control_end_si", 8),
        clip_model_name_or_path=cfg.get(
            "clip_model_name_or_path",
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ),
        pickscore_model_name_or_path=cfg.get(
            "pickscore_model_name_or_path", "yuvalkirstain/PickScore_v1"
        ),
        image_reward_path=cfg.get("image_reward_path", "ImageReward-v1.0"),
        coco_ref_stats_path=cfg.get(
            "coco_ref_stats_path", "stats/fid_stats_mscoco256_val.npz"
        ),
        inception_path=cfg.get(
            "inception_path", "stats/pt_inception-2015-12-05-6726825d.pth"
        ),
        patch_nums=patch_nums,
    )


def _build_pipe(run: dict, cfg: dict):
    """Build a fresh pipe, load base pretrained Switti + VAE, then overlay
    the run-specific fine-tuned checkpoint."""
    reso = cfg.get("reso", 512)
    patch_nums = tuple(int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[reso].split("_"))
    device = dist.get_device()

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

    base_id = cfg.get("pretrained_switti", "yresearch/Switti")
    base = SwittiHF.from_pretrained(base_id)
    missing, _ = switti.load_state_dict(base.state_dict(), strict=False)
    del base
    if dist.is_master():
        print(f"  [base] missing={len(missing)}")

    vae_id = cfg.get("vae_ckpt", "yresearch/VQVAE-Switti")
    vae_local = VQVAEHF.from_pretrained(vae_id, reso=reso).to(device)
    pipe.vae = vae_local

    ckpt_path = run.get("ckpt")
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _strip_fsdp_prefix(sd)
        miss, unexp = switti.load_state_dict(sd, strict=False)
        if dist.is_master():
            print(f"  [ckpt] {ckpt_path} missing={len(miss)} unexpected={len(unexp)}")

    switti.eval()
    vae_local.eval()
    pipe.switti = switti
    return pipe


def _evaluate_one(run: dict, cfg: dict, subset_csv: str, num_samples: int,
                  control_path: str | None) -> dict:
    """Run distributed eval for a single (modality, ckpt) entry."""
    args = _make_args(cfg, num_samples, run.get("modality"))
    pipe = _build_pipe(run, cfg)

    local_images, l_pick, l_clip, l_ir, l_ctrl = distributed_metrics_with_csv(
        pipe, subset_csv, control_path, args
    )

    ws = dist.get_world_size()
    if ws > 1:
        dist.allreduce(l_pick)
        dist.allreduce(l_clip)
        dist.allreduce(l_ir)
        for t in l_ctrl.values():
            dist.allreduce(t)

    pick = l_pick.item() / ws
    clip = l_clip.item() / ws
    ir = l_ir.item() / ws
    ctrl = {k: t.item() / ws for k, t in l_ctrl.items()}

    gathered = dist.allgather(local_images) if ws > 1 else local_images
    fid = None
    if dist.is_master():
        pil_images = [to_PIL_image(im) for im in gathered]
        fid = float(calculate_fid(
            pil_images, args.coco_ref_stats_path,
            inception_path=args.inception_path,
        ))

    result = {
        "name": run["name"],
        "modality": run.get("modality"),
        "ckpt": run.get("ckpt"),
        "control_encoder_type": run.get("control_encoder_type"),
        "control_fusion": run.get("control_fusion"),
        "num_samples": num_samples,
        "fid": fid,
        "clip_score": clip,
        "pick_score": pick,
        "image_reward": ir,
    }
    result.update(ctrl)

    del pipe, local_images, gathered
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--out_dir", required=True)
    args_cli = ap.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.initialize()

    if dist.is_master():
        os.makedirs(args_cli.out_dir, exist_ok=True)
        with open(os.path.join(args_cli.out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
    _safe_barrier()

    subset_csv = os.path.join(args_cli.out_dir, "subset.csv")
    if dist.is_master():
        n_final = _build_subset_csv(
            full_csv=cfg["coco_csv"],
            n=cfg["num_samples"],
            seed=cfg.get("seed", 42),
            eval_batch_size=cfg.get("eval_batch_size", 4),
            out_path=subset_csv,
        )
        print(f"[subset] wrote {n_final} rows -> {subset_csv}")
    _safe_barrier()
    n_final = len(pd.read_csv(subset_csv))

    results_path = os.path.join(args_cli.out_dir, "results.jsonl")
    if dist.is_master() and os.path.exists(results_path):
        print(f"[warn] {results_path} exists — appending")

    for run in cfg["runs"]:
        if dist.is_master():
            print(f"\n=========================================")
            print(f"=== EVAL: {run['name']}  (modality={run.get('modality')})")
            print(f"=========================================")

        result = _evaluate_one(
            run=run,
            cfg=cfg,
            subset_csv=subset_csv,
            num_samples=n_final,
            control_path=cfg.get("control_path"),
        )

        if dist.is_master():
            print(f"[result] {json.dumps(result, indent=2)}")
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
        _safe_barrier()

    if dist.is_master():
        rows = []
        with open(results_path) as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args_cli.out_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[done] wrote {csv_path} ({len(df)} rows)")

    if dist.initialized():
        dist.finalize()


if __name__ == "__main__":
    main()
