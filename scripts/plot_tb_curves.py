"""
Plot training loss and eval metrics from TensorBoard event files.

Usage:
    python scripts/plot_tb_curves.py --logdir tb_logs --outdir results/figures/training_curves

Directory structure expected:
    tb_logs/
        <run_name>/          # e.g. v1_add_canny, v1_cross_canny, v2_canny, ...
            events.out.tfevents.*
        OR
        <group>/
            <run_name>/
                events.out.tfevents.*

Output:
    loss_curves.pdf        — L_total per run, all runs overlaid
    fid_curves.pdf         — FID per run
    clip_curves.pdf        — CLIP score per run
    grad_norm_curves.pdf   — gradient norm per run
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── tag names ────────────────────────────────────────────────────────────────
LOSS_TAG   = "Control_iter_loss/L_total"
FID_TAG    = "coco_t2i_metrics_top_k=400_top_p=0.95_cfg=6/FID"
CLIP_TAG   = "coco_t2i_metrics_top_k=400_top_p=0.95_cfg=6/CLIP"
GRAD_TAG   = "Control_opt_grad/grad/grad_norm"

# ── colour palette (add more if needed) ──────────────────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def find_run_dirs(logdir: Path) -> dict[str, Path]:
    """Return {run_name: dir_path} for every leaf dir containing TB events."""
    runs = {}
    for root, dirs, files in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in files):
            name = Path(root).relative_to(logdir).as_posix().replace("/", "_")
            runs[name] = Path(root)
    return dict(sorted(runs.items()))


def load_scalar(event_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for one tag from one run dir."""
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average (TensorBoard-style smoothing)."""
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def plot_tag(
    runs: dict[str, Path],
    tag: str,
    ylabel: str,
    title: str,
    outpath: Path,
    smoothing: float = 0.9,
    normalize_x: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    any_plotted = False

    for i, (name, d) in enumerate(runs.items()):
        steps, values = load_scalar(d, tag)
        if len(steps) == 0:
            continue
        color = PALETTE[i % len(PALETTE)]
        x = steps / steps[-1] * 100 if normalize_x else steps
        ax.plot(x, values, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(x, smooth(values, smoothing), color=color, linewidth=1.6, label=name)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"  [skip] no data for tag: {tag}")
        return

    ax.set_xlabel("Training steps (%)" if normalize_x else "Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.xaxis.set_major_formatter(
        mticker.PercentFormatter() if normalize_x
        else mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved → {outpath}")


def plot_loss_separate(
    runs: dict[str, Path],
    outpath: Path,
    smoothing: float = 0.9,
):
    """Two-panel figure: V1 runs (left) and V2 runs (right), separate x-axes."""
    v1 = {k: v for k, v in runs.items() if "v1" in k.lower() or "add" in k.lower() or "cross" in k.lower()}
    v2 = {k: v for k, v in runs.items() if "v2" in k.lower()}
    other = {k: v for k, v in runs.items() if k not in v1 and k not in v2}

    groups = [(v1 or other, "Approach 1"), (v2, "Approach 2")]
    groups = [(g, lbl) for g, lbl in groups if g]

    if not groups:
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
    if len(groups) == 1:
        axes = [axes]

    for ax, (group, label) in zip(axes, groups):
        for i, (name, d) in enumerate(group.items()):
            steps, values = load_scalar(d, LOSS_TAG)
            if len(steps) == 0:
                continue
            color = PALETTE[i % len(PALETTE)]
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.8)
            ax.plot(steps, smooth(values, smoothing), color=color,
                    linewidth=1.6, label=name)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved → {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",  default="tb_logs",
                        help="Root directory containing TB event files")
    parser.add_argument("--outdir",  default="results/figures/training_curves",
                        help="Output directory for plots")
    parser.add_argument("--smooth",  type=float, default=0.9,
                        help="EMA smoothing factor (0 = off, 0.99 = heavy)")
    parser.add_argument("--normalize-x", action="store_true",
                        help="Normalize x-axis to 0–100%% of total steps")
    args = parser.parse_args()

    logdir  = Path(args.logdir)
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = find_run_dirs(logdir)
    if not runs:
        print(f"No TensorBoard event files found under {logdir}")
        return
    print(f"Found {len(runs)} run(s): {list(runs)}")

    # ── loss (two-panel) ─────────────────────────────────────────────────────
    plot_loss_separate(runs, outdir / "loss_curves.pdf", smoothing=args.smooth)

    # ── loss (all runs overlaid, optional) ───────────────────────────────────
    plot_tag(runs, LOSS_TAG, "Cross-entropy loss", "Training loss",
             outdir / "loss_all.pdf", smoothing=args.smooth,
             normalize_x=args.normalize_x)

    # ── FID ──────────────────────────────────────────────────────────────────
    plot_tag(runs, FID_TAG, "FID ↓", "FID over training",
             outdir / "fid_curves.pdf", smoothing=0.0,
             normalize_x=args.normalize_x)

    # ── CLIP ─────────────────────────────────────────────────────────────────
    plot_tag(runs, CLIP_TAG, "CLIP score ↑", "CLIP score over training",
             outdir / "clip_curves.pdf", smoothing=0.0,
             normalize_x=args.normalize_x)

    # ── gradient norm ────────────────────────────────────────────────────────
    plot_tag(runs, GRAD_TAG, "Gradient norm", "Gradient norm over training",
             outdir / "grad_norm_curves.pdf", smoothing=args.smooth,
             normalize_x=args.normalize_x)


if __name__ == "__main__":
    main()
