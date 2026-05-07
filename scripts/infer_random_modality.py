"""
Random-modality inference for the all-modalities SwittiControlNet checkpoint.

Assigns a random control modality to each sample, loads the matching control
map, and generates an image. No metrics are computed — output is visual only.

Usage:
    python scripts/infer_random_modality.py \
        --ckpt /path/to/model_state_dict.pt \
        --csv  eval_prompts/coco.csv \
        --control_path /path/to/val_control \
        --out_dir /path/to/output \
        --num_samples 100 \
        --batch_size 4 \
        --cfg 6.0
"""

import argparse
import json
import os
import random
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.control_pipeline import SwittiControlPipeline
from models.control_switti import MODALITY_IDS

MODALITIES = list(MODALITY_IDS.keys())  # canny, depth, seg, normals, hed, gray, openpose

_ctrl_transform_cache = {}


def _ctrl_transform(reso: int):
    if reso not in _ctrl_transform_cache:
        _ctrl_transform_cache[reso] = transforms.Compose([
            transforms.Resize((reso, reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return _ctrl_transform_cache[reso]


def load_ctrl(control_path: str, modality: str, fname: str, reso: int):
    """Load a control map; returns (tensor, PIL) or (None, None) if missing."""
    fname_png = str(fname).replace(".jpg", ".png")
    ctrl_fp = os.path.join(control_path, modality, fname_png)
    if not os.path.exists(ctrl_fp):
        return None, None
    pil = Image.open(ctrl_fp).convert("RGB")
    tensor = _ctrl_transform(reso)(pil)
    return tensor, pil.resize((reso, reso), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model_state_dict.pt")
    ap.add_argument("--csv", required=True, help="CSV with 'captions' and 'file_name' columns")
    ap.add_argument("--control_path", required=True, help="Root dir: <control_path>/<modality>/<fname>.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reso", type=int, default=512)
    ap.add_argument("--num_modalities", type=int, default=7)
    ap.add_argument(
        "--modalities", nargs="+", default=MODALITIES,
        help="Restrict which modalities to sample from (default: all 7)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("Loading SwittiControlPipeline...")
    pipe = SwittiControlPipeline.from_pretrained(
        pretrained_model_name_or_path="yresearch/Switti",
        control_ckpt=args.ckpt,
        torch_dtype=torch.float32,
        device="cuda",
        reso=args.reso,
        num_modalities=args.num_modalities,
    )
    pipe.control_net.eval()
    print("Model loaded.")

    # -------------------------------------------------------------------------
    # Sample rows, assign random modality, filter to rows with a control map
    # -------------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    df = df.sample(n=min(args.num_samples, len(df)), random_state=args.seed).reset_index(drop=True)
    df["modality"] = [rng.choice(args.modalities) for _ in range(len(df))]

    valid = []
    for _, row in df.iterrows():
        fname_png = str(row.get("file_name", "None")).replace(".jpg", ".png")
        ctrl_fp = os.path.join(args.control_path, row["modality"], fname_png)
        if os.path.exists(ctrl_fp):
            valid.append(row.to_dict())
    df = pd.DataFrame(valid).reset_index(drop=True)

    if df.empty:
        print("No valid samples found — check --control_path and --csv.")
        return

    print(f"{len(df)} valid samples | modality distribution: {df['modality'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # Generate — process each modality as a batch to keep the pipe happy
    # -------------------------------------------------------------------------
    records = []  # metadata for summary.json

    for modality in args.modalities:
        subset = df[df["modality"] == modality].reset_index(drop=True)
        if subset.empty:
            continue

        print(f"\n[{modality}] {len(subset)} samples")

        for start in tqdm(range(0, len(subset), args.batch_size), desc=modality):
            batch = subset.iloc[start: start + args.batch_size]
            prompts = [str(r["captions"]) for _, r in batch.iterrows()]

            ctrl_tensors, ctrl_pils = [], []
            for _, row in batch.iterrows():
                t, pil = load_ctrl(args.control_path, modality, row["file_name"], args.reso)
                ctrl_tensors.append(t)
                ctrl_pils.append(pil)

            ctrl_batch = torch.stack(ctrl_tensors)  # (B, 3, H, W)

            gen_images = pipe(
                prompt=prompts,
                ctrl_image=ctrl_batch,
                modality=modality,
                seed=args.seed,
                cfg=args.cfg,
                top_k=400,
                top_p=0.95,
                return_pil=True,
            )

            for j, (gen, ctrl_pil) in enumerate(zip(gen_images, ctrl_pils)):
                row = batch.iloc[j]
                global_idx = start + j
                tag = f"{modality}_{global_idx:04d}"
                sample_dir = os.path.join(args.out_dir, tag)
                os.makedirs(sample_dir, exist_ok=True)

                gen.save(os.path.join(sample_dir, "generated.jpg"), quality=95)
                if ctrl_pil is not None:
                    ctrl_pil.save(os.path.join(sample_dir, f"control_{modality}.png"))

                # Side-by-side comparison: control | generated
                side = Image.new("RGB", (args.reso * 2, args.reso))
                if ctrl_pil is not None:
                    side.paste(ctrl_pil, (0, 0))
                side.paste(gen, (args.reso, 0))
                side.save(os.path.join(sample_dir, "comparison.jpg"), quality=95)

                with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                    f.write(str(row["captions"]))

                records.append({
                    "tag": tag,
                    "modality": modality,
                    "file_name": str(row["file_name"]),
                    "prompt": str(row["captions"]),
                })

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone. {len(records)} images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
