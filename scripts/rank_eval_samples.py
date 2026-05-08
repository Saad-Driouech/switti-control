"""
Rank generated evaluation samples by composite quality + control-adherence score.

Works for both v1 and v2 eval output folders. Reads the samples/ subfolder
produced by evaluate.py and ranks each (modality, cfg) run independently.

Usage:
    python scripts/rank_eval_samples.py \\
        --eval_dir /path/to/eval_folder \\
        --out_dir  /path/to/ranking_output \\
        --top_n    50 \\
        --quality_weight 0.5 \\
        --control_weight 0.5

Output per (modality, cfg) folder:
    <out_dir>/<run>/metrics.csv       — all samples ranked, with per-sample scores
    <out_dir>/<run>/rank<NNN>_<idx>.jpg — top-N generated images

Top-level:
    <out_dir>/summary.md              — top-5 across all runs
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

def _detect_modality(run_dir: Path) -> str | None:
    """Infer modality from the control map filename in the first sample."""
    for sample_dir in sorted(run_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        for f in sample_dir.iterdir():
            if f.name.startswith("control_") and f.suffix == ".png":
                return f.name[len("control_"):-len(".png")]
    return None


# ---------------------------------------------------------------------------
# Per-sample CLIP score
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None


def _load_clip(device, model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import AutoModel, AutoProcessor
        _clip_processor = AutoProcessor.from_pretrained(model_name)
        _clip_model = AutoModel.from_pretrained(model_name).eval().to(device)


def _free_clip():
    global _clip_model, _clip_processor
    _clip_model = None
    _clip_processor = None
    import gc; gc.collect(); torch.cuda.empty_cache()


def compute_clip_scores(images: list, prompts: list, device: str,
                        model_name: str, batch_size: int = 32) -> np.ndarray:
    _load_clip(device, model_name)
    scores = []
    for i in range(0, len(images), batch_size):
        imgs_b = images[i:i + batch_size]
        txts_b = prompts[i:i + batch_size]
        img_inputs = _clip_processor(images=imgs_b, return_tensors="pt",
                                     padding=True).to(device)
        txt_inputs = _clip_processor(text=txts_b, return_tensors="pt",
                                     padding=True, truncation=True).to(device)
        with torch.no_grad():
            img_emb = _clip_model.get_image_features(**img_inputs)
            txt_emb = _clip_model.get_text_features(**txt_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            s = (img_emb * txt_emb).sum(dim=-1).cpu().float().numpy()
        scores.extend(s.tolist())
    return np.array(scores)


# ---------------------------------------------------------------------------
# Per-sample control metrics
# ---------------------------------------------------------------------------

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32) / 255.0


def compute_edge_f1(gen: Image.Image, ctrl: Image.Image,
                    threshold: float = 0.1) -> float:
    """Binary F1 between detected canny edges on generated image and reference."""
    import cv2
    gen_gray = np.array(gen.convert("L"))
    median = float(np.median(gen_gray))
    lo = int(max(0, 0.66 * median))
    hi = int(min(255, 1.33 * median))
    gen_edges = cv2.Canny(gen_gray, lo, hi).astype(bool)

    ctrl_gray = _pil_to_gray_np(ctrl)
    ctrl_edges = ctrl_gray > threshold

    tp = (gen_edges & ctrl_edges).sum()
    fp = (gen_edges & ~ctrl_edges).sum()
    fn = (~gen_edges & ctrl_edges).sum()
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_ssim(gen: Image.Image, ctrl: Image.Image) -> float:
    from skimage.metrics import structural_similarity as ssim
    g = _pil_to_gray_np(gen.resize(ctrl.size, Image.LANCZOS))
    c = _pil_to_gray_np(ctrl)
    return float(ssim(g, c, data_range=1.0))


def compute_depth_rmse(gen: Image.Image, ctrl: Image.Image, device: str) -> float:
    from utils.control_metrics import _get_depth_estimator
    model = _get_depth_estimator(device)
    gen_t = torch.from_numpy(
        np.array(gen.convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.infer(gen_t).squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    ctrl_np = _pil_to_gray_np(ctrl.resize(gen.size, Image.NEAREST))
    ctrl_np = (ctrl_np - ctrl_np.min()) / (ctrl_np.max() - ctrl_np.min() + 1e-8)

    valid = ctrl_np > 0.05
    if valid.sum() == 0:
        return 1.0
    pred_r = pred[valid]
    ctrl_r = ctrl_np[valid]
    # scale-invariant: fit linear scale
    scale = np.dot(pred_r, ctrl_r) / (np.dot(pred_r, pred_r) + 1e-8)
    return float(np.sqrt(np.mean((scale * pred_r - ctrl_r) ** 2)))


def compute_normal_rmse(gen: Image.Image, ctrl: Image.Image) -> float:
    from utils.control_metrics import _get_normal_estimator
    detector = _get_normal_estimator()
    gen_normal = detector(gen)
    g = _pil_to_np(gen_normal.resize(ctrl.size, Image.NEAREST))
    c = _pil_to_np(ctrl)
    g = g * 2 - 1
    c = c * 2 - 1
    g_norm = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
    c_norm = c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-8)
    cos = np.clip((g_norm * c_norm).sum(axis=-1), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def compute_seg_iou(gen: Image.Image, ctrl: Image.Image, device: str) -> float:
    from utils.control_metrics import _get_seg_model
    import torchvision.transforms.functional as tvf
    model = _get_seg_model(device)
    gen_t = tvf.to_tensor(gen.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(gen_t)[0]
    gen_mask = torch.zeros(gen_t.shape[-2:], dtype=torch.bool)
    for mask in out.get("masks", []):
        if mask.squeeze().float().mean() > 0.5:
            gen_mask |= mask.squeeze().bool().cpu()

    ctrl_gray = _pil_to_gray_np(ctrl.resize(gen.size, Image.NEAREST))
    ctrl_mask = ctrl_gray > 0.1

    inter = (gen_mask.numpy() & ctrl_mask).sum()
    union = (gen_mask.numpy() | ctrl_mask).sum()
    return float(inter / (union + 1e-8))


CONTROL_METRIC_FN = {
    "canny":   ("edge_f1",      lambda g, c, dev: compute_edge_f1(g, c)),
    "hed":     ("ssim",         lambda g, c, dev: compute_ssim(g, c)),
    "depth":   ("depth_rmse",   lambda g, c, dev: compute_depth_rmse(g, c, dev)),
    "normals": ("normal_rmse",  lambda g, c, dev: compute_normal_rmse(g, c)),
    "seg":     ("seg_iou",      lambda g, c, dev: compute_seg_iou(g, c, dev)),
    "gray":    ("ssim",         lambda g, c, dev: compute_ssim(g, c)),
}

# For these metrics, lower = better → invert during normalisation
LOWER_IS_BETTER = {"depth_rmse", "normal_rmse"}


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------

def rank_run(run_dir: Path, subset_df: pd.DataFrame, modality: str,
             device: str, clip_model: str,
             quality_w: float, control_w: float) -> pd.DataFrame:
    sample_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not sample_dirs:
        return pd.DataFrame()

    images, prompts, ctrl_maps, indices = [], [], [], []

    for sd in sample_dirs:
        idx = int(sd.name)
        gen_fp = sd / "generated.jpg"
        if not gen_fp.exists():
            continue

        # Prompt: prefer prompt.txt, fall back to subset.csv by index
        prompt_fp = sd / "prompt.txt"
        if prompt_fp.exists():
            prompt = prompt_fp.read_text().strip()
        elif idx < len(subset_df):
            prompt = str(subset_df.iloc[idx].get("captions", ""))
        else:
            prompt = ""

        # Control map
        ctrl_fp = None
        for f in sd.iterdir():
            if f.name.startswith("control_") and f.suffix == ".png":
                ctrl_fp = f
                break

        images.append(Image.open(gen_fp).convert("RGB"))
        prompts.append(prompt)
        ctrl_maps.append(Image.open(ctrl_fp).convert("RGB") if ctrl_fp else None)
        indices.append(idx)

    if not images:
        return pd.DataFrame()

    # CLIP scores
    print(f"  Computing CLIP scores ({len(images)} samples)...")
    clip_scores = compute_clip_scores(images, prompts, device, clip_model)
    _free_clip()

    # Control scores
    ctrl_scores = np.zeros(len(images))
    metric_name = "none"
    if modality in CONTROL_METRIC_FN and any(c is not None for c in ctrl_maps):
        metric_name, metric_fn = CONTROL_METRIC_FN[modality]
        print(f"  Computing {metric_name} ({len(images)} samples)...")
        for i, (gen, ctrl) in enumerate(tqdm(zip(images, ctrl_maps),
                                              total=len(images), leave=False)):
            if ctrl is not None:
                try:
                    ctrl_scores[i] = metric_fn(gen, ctrl, device)
                except Exception as e:
                    print(f"  [warn] sample {indices[i]}: {e}")
                    ctrl_scores[i] = np.nan

        # Free any loaded control metric models
        from utils.control_metrics import free_control_metric_models
        free_control_metric_models()

    # Normalise to [0, 1] (ignore NaN)
    def _norm(arr, invert=False):
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return np.where(valid, 0.5, np.nan)
        mn, mx = arr[valid].min(), arr[valid].max()
        if mx == mn:
            return np.where(valid, 0.5, np.nan)
        n = (arr - mn) / (mx - mn)
        return (1 - n) if invert else n

    clip_norm = _norm(clip_scores)
    ctrl_norm = _norm(ctrl_scores, invert=(metric_name in LOWER_IS_BETTER))

    # Composite score
    if modality in CONTROL_METRIC_FN:
        composite = quality_w * np.nan_to_num(clip_norm) + \
                    control_w * np.nan_to_num(ctrl_norm)
    else:
        composite = np.nan_to_num(clip_norm)

    df = pd.DataFrame({
        "sample_idx":    indices,
        "prompt":        prompts,
        "clip_score":    clip_scores,
        "clip_norm":     clip_norm,
        metric_name:     ctrl_scores,
        "control_norm":  ctrl_norm,
        "composite":     composite,
    })
    df["rank"] = df["composite"].rank(ascending=False, method="min").astype(int)
    df.sort_values("rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True,
                    help="Path to eval output folder (contains samples/ and subset.csv)")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write ranking results")
    ap.add_argument("--top_n", type=int, default=50,
                    help="Number of top images to copy per run (default 50)")
    ap.add_argument("--quality_weight", type=float, default=0.5)
    ap.add_argument("--control_weight", type=float, default=0.5)
    ap.add_argument("--clip_model",
                    default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    samples_dir = eval_dir / "samples"
    subset_csv = eval_dir / "subset.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not samples_dir.exists():
        raise FileNotFoundError(f"No samples/ folder found in {eval_dir}")

    subset_df = pd.read_csv(subset_csv) if subset_csv.exists() else pd.DataFrame()

    run_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run subfolders found in {samples_dir}")

    all_top = []  # for summary

    for run_dir in run_dirs:
        run_name = run_dir.name
        modality = _detect_modality(run_dir)
        print(f"\n=== {run_name}  (modality={modality}) ===")

        df = rank_run(
            run_dir=run_dir,
            subset_df=subset_df,
            modality=modality or "",
            device=args.device,
            clip_model=args.clip_model,
            quality_w=args.quality_weight,
            control_w=args.control_weight,
        )
        if df.empty:
            print("  No samples found, skipping.")
            continue

        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        # Save metrics CSV
        df.to_csv(run_out / "metrics.csv", index=False)

        # Copy top-N images
        top_n = min(args.top_n, len(df))
        for _, row in df.head(top_n).iterrows():
            idx = int(row["sample_idx"])
            rank = int(row["rank"])
            src = run_dir / f"{idx:04d}" / "generated.jpg"
            dst = run_out / f"rank{rank:03d}_sample{idx:04d}.jpg"
            if src.exists():
                shutil.copy(src, dst)

        # Collect top-5 for summary
        for _, row in df.head(5).iterrows():
            all_top.append({
                "run":       run_name,
                "modality":  modality,
                "rank":      int(row["rank"]),
                "sample_idx": int(row["sample_idx"]),
                "composite": round(float(row["composite"]), 4),
                "clip_score": round(float(row["clip_score"]), 4),
                "prompt":    row["prompt"][:120],
            })

        print(f"  Done. Top sample: idx={df.iloc[0]['sample_idx']}, "
              f"composite={df.iloc[0]['composite']:.3f}, "
              f"clip={df.iloc[0]['clip_score']:.3f}")

    # Write summary
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Ranking Summary\n\n")
        f.write(f"Weights: quality={args.quality_weight}, "
                f"control={args.control_weight}\n\n")
        for run_name in sorted({r["run"] for r in all_top}):
            rows = [r for r in all_top if r["run"] == run_name]
            f.write(f"## {run_name}\n\n")
            f.write("| Rank | Sample | Composite | CLIP | Prompt |\n")
            f.write("|------|--------|-----------|------|--------|\n")
            for r in rows:
                f.write(f"| {r['rank']} | {r['sample_idx']:04d} | "
                        f"{r['composite']} | {r['clip_score']} | "
                        f"{r['prompt']} |\n")
            f.write("\n")

    print(f"\nDone. Results -> {out_dir}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
