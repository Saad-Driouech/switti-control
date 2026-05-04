"""
Read results.jsonl produced by evaluate.py and emit:
  - a markdown table (stdout)
  - a LaTeX `tabular` (stdout, after the markdown)

Usage:
    python scripts/report_eval.py /path/to/eval_run/results.jsonl
"""

import argparse
import json
import sys

import pandas as pd

# Generic columns common to every run. Modality-specific columns are
# discovered from the data and appended automatically.
GENERIC_COLS = ["fid", "clip_score", "pick_score", "image_reward"]

# Pretty headers
HEADERS = {
    "name": "Run",
    "modality": "Modality",
    "fid": "FID ↓",
    "clip_score": "CLIP ↑",
    "pick_score": "PickScore ↑",
    "image_reward": "ImageReward ↑",
    "ssim": "SSIM ↑",
    "edge_iou": "EdgeIoU ↑",
    "edge_f1": "EdgeF1 ↑",
    "edge_precision": "EdgeP ↑",
    "edge_recall": "EdgeR ↑",
    "depth_abs_rel": "AbsRel ↓",
    "depth_rmse": "RMSE ↓",
    "depth_delta1": "δ<1.25 ↑",
    "normal_mae_deg": "Normal° ↓",
    "normal_cosine_sim": "NormalCos ↑",
    "hed_ssim": "HED-SSIM ↑",
    "hed_f1": "HED-F1 ↑",
    "pose_ssim": "Pose-SSIM ↑",
    "pose_skeleton_f1": "Pose-F1 ↑",
    "seg_miou": "mIoU ↑",
    "seg_pixel_acc": "PixAcc ↑",
}


def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="results.jsonl from evaluate.py")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Column order: name, modality, generic, then everything else
    extra_cols = [c for c in df.columns
                  if c not in {"name", "modality", "ckpt",
                               "control_encoder_type", "control_fusion",
                               "num_samples"} | set(GENERIC_COLS)]
    cols = ["name", "modality"] + GENERIC_COLS + extra_cols
    cols = [c for c in cols if c in df.columns]

    # ---------- Markdown ----------
    print("## Universal evaluation results")
    print(f"\n_{len(df)} runs · {df['num_samples'].iloc[0] if 'num_samples' in df else '?'} samples each_\n")
    header = "| " + " | ".join(HEADERS.get(c, c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")

    # ---------- LaTeX ----------
    print("\n\n% ---------- LaTeX ----------")
    print("\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}")
    print("\\toprule")
    print(" & ".join(HEADERS.get(c, c).replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$") for c in cols) + " \\\\")
    print("\\midrule")
    for _, row in df.iterrows():
        print(" & ".join(fmt(row.get(c)) for c in cols) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
