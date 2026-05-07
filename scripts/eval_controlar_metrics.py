"""
Post-hoc metrics for ControlAR-generated images.

Loads images saved by evaluations/infer_coco_csv.py and computes the same
metrics as evaluate.py (FID, CLIP, PickScore, ImageReward, control metrics),
writing one result row to results.jsonl + results.csv in --out_dir.

Usage:
    python scripts/eval_controlar_metrics.py \
        --images_dir /path/to/eval_controlar/canny/visualization \
        --csv        /path/to/eval_out/subset.csv \
        --control_path /path/to/val_control \
        --modality   canny \
        --run_name   controlar_canny \
        --out_dir    /path/to/eval_out
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from calculate_metrics import calculate_scores
from utils.control_metrics import calculate_control_metrics
from utils.fid_score_in_memory import calculate_fid


def load_generated_images(images_dir: str, n: int) -> list:
    """Load PNG files 000000.png … (n-1).png as PIL images."""
    images = []
    for i in tqdm(range(n), desc="loading images"):
        fp = os.path.join(images_dir, f"{i:06d}.png")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Expected {fp} — did ControlAR finish generating all {n} images?")
        images.append(Image.open(fp).convert("RGB"))
    return images


def load_ctrl_tensors(csv_records: list, control_path: str, modality: str,
                      reso: int) -> list:
    """Load pre-extracted control maps as normalised tensors (same as eval pipeline)."""
    transform = transforms.Compose([
        transforms.Resize((reso, reso)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = []
    for row in csv_records:
        fname = str(row.get("file_name", "None"))
        fname_png = fname.replace(".jpg", ".png")
        ctrl_fp = os.path.join(control_path, modality, fname_png)
        if os.path.exists(ctrl_fp):
            try:
                t = transform(Image.open(ctrl_fp).convert("RGB"))
            except Exception:
                t = torch.zeros(3, reso, reso)
        else:
            t = torch.zeros(3, reso, reso)
        tensors.append(t)
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True,
                    help="Path to ControlAR visualization/ output dir")
    ap.add_argument("--csv", required=True,
                    help="subset.csv used for both ControlAR inference and your eval")
    ap.add_argument("--control_path", required=True,
                    help="Root of pre-extracted control maps")
    ap.add_argument("--modality", required=True)
    ap.add_argument("--run_name", required=True,
                    help="Row name in results.jsonl (e.g. controlar_canny)")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write/append results.jsonl and results.csv")
    ap.add_argument("--reso", type=int, default=512)
    ap.add_argument("--clip_model_name_or_path",
                    default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--pickscore_model_name_or_path",
                    default="yuvalkirstain/PickScore_v1")
    ap.add_argument("--image_reward_path", default="ImageReward-v1.0")
    ap.add_argument("--coco_ref_stats_path",
                    default="stats/fid_stats_mscoco256_val.npz")
    ap.add_argument("--inception_path",
                    default="stats/pt_inception-2015-12-05-6726825d.pth")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    n = len(df)
    prompts = df["captions"].astype(str).tolist()
    print(f"[info] {n} samples, modality={args.modality}, run={args.run_name}")

    # ---- Load generated images ----
    pil_images = load_generated_images(args.images_dir, n)

    # ---- CLIP / PickScore / ImageReward ----
    print("[metrics] computing CLIP / PickScore / ImageReward...")
    pick_score, clip_score, image_reward = calculate_scores(
        pil_images, prompts, device=device,
        clip_model_name_or_path=args.clip_model_name_or_path,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        image_reward_path=args.image_reward_path,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # ---- FID ----
    print("[metrics] computing FID...")
    fid = float(calculate_fid(
        pil_images, args.coco_ref_stats_path,
        inception_path=args.inception_path,
    ))

    # ---- Control metrics ----
    print("[metrics] computing control metrics...")
    ctrl_tensors = load_ctrl_tensors(
        df.to_dict("records"), args.control_path, args.modality, args.reso)
    ctrl_metrics = calculate_control_metrics(
        pil_images, ctrl_tensors, args.modality, device=device)

    # ---- Write result row ----
    result = {
        "name":        args.run_name,
        "run":         args.run_name,
        "modality":    args.modality,
        "num_samples": n,
        "fid":         fid,
        "clip_score":  float(clip_score),
        "pick_score":  float(pick_score),
        "image_reward": float(image_reward),
    }
    result.update(ctrl_metrics)
    print(f"[result] {json.dumps(result, indent=2)}")

    results_path = os.path.join(args.out_dir, "results.jsonl")
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    rows = []
    with open(results_path) as f:
        for line in f:
            rows.append(json.loads(line))
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "results.csv"), index=False)

    print(f"[done] appended to {results_path}")


if __name__ == "__main__":
    main()
