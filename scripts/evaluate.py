"""
Evaluation pipeline for Approach 2 (SwittiControlNet / ControlNet-style branch).

Runs every run listed in a YAML config against the SAME deterministic subset
of COCO val2014 prompts, computes:
  - FID, CLIP score, PickScore, ImageReward
  - control-modality-specific metrics (SSIM, edge IoU, depth, normal, HED, pose, seg)
and writes one row per run to results.jsonl + results.csv.

Usage (single GPU):
    python scripts/evaluate.py --config scripts/eval_config_example.yaml \
        --out_dir /path/to/eval_run/

Usage (multi-GPU DDP):
    torchrun --standalone --nproc_per_node=4 scripts/evaluate.py \
        --config scripts/eval_config_example.yaml --out_dir /path/to/eval_run/
"""

import argparse
import gc
import json
import os
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import dist
from calculate_metrics import distributed_metrics_with_csv, to_PIL_image
from models.control_pipeline import SwittiControlPipeline
from utils.control_metrics import free_control_metric_models
from utils.fid_score_in_memory import calculate_fid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_barrier():
    if dist.initialized():
        torch.distributed.barrier()


def _build_subset_csv(full_csv: str, n: int, seed: int, eval_batch_size: int,
                      out_path: str) -> int:
    """Sample n rows deterministically; truncate to a multiple of
    eval_batch_size * world_size. Returns the final row count."""
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


def _make_args(cfg: dict, num_samples: int,
               guidance: float | None = None) -> SimpleNamespace:
    """Build the args namespace that distributed_metrics_with_csv reads.

    `guidance` overrides cfg["guidance"] when provided (used for CFG sweeps).
    """
    reso = cfg.get("reso", 512)
    if guidance is None:
        guidance = cfg.get("guidance", 6.0)
    return SimpleNamespace(
        metrics_max_count=num_samples,
        eval_batch_size=cfg.get("eval_batch_size", 4),
        data_load_reso=reso,
        num_images_for_metrics=1,
        seed=cfg.get("seed", 42),
        guidance=guidance,
        top_k=cfg.get("top_k", 400),
        top_p=cfg.get("top_p", 0.95),
        clip_model_name_or_path=cfg.get(
            "clip_model_name_or_path", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        pickscore_model_name_or_path=cfg.get(
            "pickscore_model_name_or_path", "yuvalkirstain/PickScore_v1"),
        image_reward_path=cfg.get("image_reward_path", "ImageReward-v1.0"),
        coco_ref_stats_path=cfg.get(
            "coco_ref_stats_path", "stats/fid_stats_mscoco256_val.npz"),
        inception_path=cfg.get(
            "inception_path", "stats/pt_inception-2015-12-05-6726825d.pth"),
    )


def _build_pipe(run: dict, cfg: dict) -> SwittiControlPipeline:
    """Load SwittiControlPipeline with pretrained Switti + control checkpoint."""
    reso = cfg.get("reso", 512)
    device = dist.get_device()
    pipe = SwittiControlPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.get("pretrained_switti", "yresearch/Switti"),
        control_ckpt=run.get("ckpt"),
        torch_dtype=torch.float32,
        device=device,
        reso=reso,
        num_modalities=run.get("num_modalities", 7),
    )
    pipe.control_net.eval()
    return pipe


def _save_samples(run: dict, cfg: dict, subset_csv: str, pil_images: list,
                  out_dir: str, save_name: str | None = None) -> None:
    """Save generated images, control maps, and original images for qualitative analysis."""
    n = cfg.get("num_save_samples", 100)
    modality = run.get("modality")
    control_path = cfg.get("control_path")
    images_path = cfg.get("coco_images_path")  # optional: path to original COCO val images
    reso = cfg.get("reso", 512)

    df = pd.read_csv(subset_csv)
    n = min(n, len(df), len(pil_images))

    save_dir = os.path.join(out_dir, "samples", save_name or run["name"])
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        fname = str(df.iloc[i].get("file_name", "None"))
        sample_dir = os.path.join(save_dir, f"{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Generated image
        pil_images[i].save(os.path.join(sample_dir, "generated.jpg"), quality=95)

        # Control map — copy directly to preserve quality
        if modality and control_path and fname != "None":
            fname_png = fname.replace(".jpg", ".png")
            ctrl_fp = os.path.join(control_path, modality, fname_png)
            if os.path.exists(ctrl_fp):
                shutil.copy(ctrl_fp, os.path.join(sample_dir, f"control_{modality}.png"))

        # Original image — resize to match generated resolution
        if images_path and fname != "None":
            from PIL import Image as PILImage
            orig_fp = os.path.join(images_path, fname)
            if os.path.exists(orig_fp):
                orig = PILImage.open(orig_fp).convert("RGB").resize(
                    (reso, reso), PILImage.LANCZOS
                )
                orig.save(os.path.join(sample_dir, "original.jpg"), quality=95)

    print(f"[samples] saved {n} samples -> {save_dir}")


def _evaluate_with_pipe(pipe, run: dict, cfg: dict, subset_csv: str,
                        num_samples: int, control_path: str | None,
                        out_dir: str, guidance: float,
                        result_name: str) -> dict:
    """Run eval for one (run, guidance) combination using a pre-built pipe."""
    args = _make_args(cfg, num_samples, guidance=guidance)

    preview_dir = None
    local_images, l_pick, l_clip, l_ir, l_ctrl = distributed_metrics_with_csv(
        pipe,
        subset_csv,
        args,
        control_path=control_path,
        control_modality=run.get("modality"),
        preview_save_dir=preview_dir,
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
        _save_samples(run, cfg, subset_csv, pil_images, out_dir,
                      save_name=result_name)

    result = {
        "name": result_name,
        "run": run["name"],
        "modality": run.get("modality"),
        "cfg": guidance,
        "ckpt": run.get("ckpt"),
        "num_modalities": run.get("num_modalities", 7),
        "num_samples": num_samples,
        "fid": fid,
        "clip_score": clip,
        "pick_score": pick,
        "image_reward": ir,
    }
    result.update(ctrl)

    del local_images, gathered
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
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

    # cfg_sweep: list of guidance values to evaluate for every run.
    # Falls back to a single-element list using the global guidance value,
    # which preserves backwards-compatible behaviour.
    cfg_values = cfg.get("cfg_sweep") or [cfg.get("guidance", 6.0)]

    for run in cfg["runs"]:
        if dist.is_master():
            print(f"\n=========================================")
            print(f"=== RUN: {run['name']}  (modality={run.get('modality')})")
            print(f"=== CFG sweep: {cfg_values}")
            print(f"=========================================")

        pipe = _build_pipe(run, cfg)

        for guidance in cfg_values:
            if len(cfg_values) == 1:
                result_name = run["name"]
            else:
                result_name = f"{run['name']}_cfg{guidance:g}"

            if dist.is_master():
                print(f"\n--- guidance={guidance}  name={result_name}")

            result = _evaluate_with_pipe(
                pipe=pipe,
                run=run,
                cfg=cfg,
                subset_csv=subset_csv,
                num_samples=n_final,
                control_path=cfg.get("control_path"),
                out_dir=args_cli.out_dir,
                guidance=guidance,
                result_name=result_name,
            )
            free_control_metric_models()

            if dist.is_master():
                print(f"[result] {json.dumps(result, indent=2)}")
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
            _safe_barrier()

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

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
