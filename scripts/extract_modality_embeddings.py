"""
Run on the cluster to extract modality embedding weights from checkpoints.
Saves one tiny .npy file per checkpoint (~3.5 KB each).

Usage:
    python scripts/extract_modality_embeddings.py \
        --ckpt_dir /path/to/v2_uni_checkpoints \
        --outdir   /tmp/modality_embs \
        --glob     "model_*.pt"

Then rsync the outdir to your local machine and run analyze_modality_embeddings.py.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def extract_step(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--glob",     default="model_*.pt")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(ckpt_dir.glob(args.glob), key=extract_step)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir} matching {args.glob}")

    for ckpt in ckpts:
        step = extract_step(ckpt)
        state = torch.load(ckpt, map_location="cpu", mmap=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

        try:
            key = next(k for k in state if "modality_embed.weight" in k)
        except StopIteration:
            print(f"[skip] {ckpt.name}: modality_embed.weight not found")
            continue

        w = state[key].float().numpy()[:6]   # (6, 128), drop null token
        out = outdir / f"step_{step:06d}.npy"
        np.save(out, w)
        print(f"  {ckpt.name} → {out.name}  shape={w.shape}")

    print(f"\nDone. {len(list(outdir.glob('*.npy')))} files in {outdir}")
    print(f"rsync command:\n  rsync -av tinyx:{outdir}/ ./results/modality_embs/")


if __name__ == "__main__":
    main()
