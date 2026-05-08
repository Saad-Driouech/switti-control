"""
Metrics-only: compute all metrics from pre-generated images (no generative model needed).
Run after generate.py has saved images to disk.

Reads:
    <out_dir>/subset.csv
    <out_dir>/<run_name>_cfg<cfg>/images/

Writes:
    <out_dir>/<run_name>_cfg<cfg>/result.json
    <out_dir>/results.jsonl   (appended)
    <out_dir>/results.csv     (rewritten from full jsonl)

Usage:
    python scripts/compute_metrics.py \\
        --config scripts/eval_config.yaml \\
        --run_name approach1_canny --cfg 6.0 \\
        --out_dir /path/to/eval_out/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from calculate_metrics import calculate_scores
from utils.control_metrics import calculate_control_metrics, free_control_metric_models
from utils.fid_score_in_memory import calculate_fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--cfg", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = ap.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    run = next((r for r in cfg["runs"] if r["name"] == args_cli.run_name), None)
    if run is None:
        raise ValueError(
            f"Run '{args_cli.run_name}' not found. "
            f"Available: {[r['name'] for r in cfg['runs']]}"
        )

    out_dir = Path(args_cli.out_dir)
    result_name = f"{args_cli.run_name}_cfg{args_cli.cfg:g}"
    run_dir = out_dir / result_name
    images_dir = run_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"images/ not found at {images_dir}. Run generate.py first."
        )

    subset_csv = out_dir / "subset.csv"
    df = pd.read_csv(subset_csv)
    captions = df["captions"].astype(str).tolist()
    filenames = df["file_name"].astype(str).tolist() \
        if "file_name" in df.columns else [""] * len(df)

    # Load generated images sorted by global index
    image_paths = sorted(images_dir.glob("*.jpg"), key=lambda p: p.stem)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Loading {len(image_paths)} generated images...")
    pil_images = [Image.open(p).convert("RGB") for p in tqdm(image_paths)]
    prompts = [captions[int(p.stem)] for p in image_paths]

    # Quality metrics
    reso = cfg.get("reso", 512)
    print("Computing quality metrics (CLIP / PickScore / ImageReward)...")
    pick_score, clip_score, image_reward = calculate_scores(
        pil_images,
        prompts,
        device=args_cli.device,
        clip_model_name_or_path=cfg.get(
            "clip_model_name_or_path", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        pickscore_model_name_or_path=cfg.get(
            "pickscore_model_name_or_path", "yuvalkirstain/PickScore_v1"),
        image_reward_path=cfg.get("image_reward_path", "ImageReward-v1.0"),
    )

    # FID
    print("Computing FID...")
    fid = float(calculate_fid(
        pil_images,
        cfg.get("coco_ref_stats_path", "stats/fid_stats_mscoco256_val.npz"),
        inception_path=cfg.get(
            "inception_path", "stats/pt_inception-2015-12-05-6726825d.pth"),
    ))

    # Control metrics
    ctrl_metrics = {}
    modality = run.get("modality")
    control_path = cfg.get("control_path")

    if modality and control_path:
        print(f"Computing control metrics ({modality})...")
        from utils.data import JointTransform
        ctrl_transform = JointTransform(
            final_reso=reso,
            mid_reso=cfg.get("mid_reso", 1.125),
            hflip_prob=0.0,
        )

        ctrl_tensors = []
        for p in image_paths:
            global_idx = int(p.stem)
            fname = filenames[global_idx]
            fname_png = fname.replace(".jpg", ".png")
            if fname in ("None", "nan", ""):
                ctrl_tensors.append(None)
            else:
                ctrl_fp = os.path.join(control_path, modality, fname_png)
                if os.path.exists(ctrl_fp):
                    try:
                        img = Image.open(ctrl_fp).convert("RGB")
                        _, processed = ctrl_transform(img, {modality: img})
                        ctrl_tensors.append(processed[modality])
                    except Exception as e:
                        print(f"[warn] {ctrl_fp}: {e}")
                        ctrl_tensors.append(None)
                else:
                    ctrl_tensors.append(None)

        # v1 calculate_control_metrics expects {modality: list_of_tensors}
        ctrl_dict = {modality: ctrl_tensors}
        raw = calculate_control_metrics(
            pil_images, ctrl_dict, modality, device=args_cli.device
        )
        # prefix keys with modality to match v1 convention
        ctrl_metrics = {f"{modality}_{k}": v for k, v in raw.items()}
        free_control_metric_models()
        print(f"  {ctrl_metrics}")

    result = {
        "name":                 result_name,
        "run":                  run["name"],
        "modality":             modality,
        "cfg":                  args_cli.cfg,
        "ckpt":                 run.get("ckpt"),
        "control_encoder_type": run.get("control_encoder_type"),
        "control_fusion":       run.get("control_fusion"),
        "num_samples":          len(pil_images),
        "fid":                  fid,
        "clip_score":           float(clip_score),
        "pick_score":           float(pick_score),
        "image_reward":         float(image_reward),
    }
    result.update(ctrl_metrics)

    result_json = run_dir / "result.json"
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] {json.dumps(result, indent=2)}")

    results_jsonl = out_dir / "results.jsonl"
    with open(results_jsonl, "a") as f:
        f.write(json.dumps(result) + "\n")

    rows = []
    with open(results_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    print(f"\n[done] {result_name}")
    print(f"  FID={fid:.2f}  CLIP={float(clip_score):.4f}  "
          f"Pick={float(pick_score):.4f}  IR={float(image_reward):.4f}")


if __name__ == "__main__":
    main()
