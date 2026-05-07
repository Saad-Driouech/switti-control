"""
Prepare COCOStuff segmentation maps for switti-control training.

Uses the official COCO panoptic annotations (same format ControlAR uses):
RGB-encoded PNG where pixel color = panoptic segment ID
(id = R + G*256 + B*256²).

Output layout (matches existing modalities in switti-control):
    <out_ctrl_dir>/seg_cocostuff/<image_filename_stem>.png   -- seg control maps
    <out_csv>                                                  -- file_name, captions CSV

Downloads needed (from https://cocodataset.org/#download):
    - train2017.zip                  (images, ~18 GB)
    - panoptic_train2017.zip         (panoptic PNGs, ~821 MB)
    - annotations_trainval2017.zip   (JSONs: captions + panoptic, ~241 MB)

Expected directory layout before running:
    <data_root>/
        train2017/               ← COCO train images
        panoptic_train2017/      ← extracted from panoptic_train2017.zip
        annotations/
            captions_train2017.json
            panoptic_train2017.json

Usage:
    python scripts/prepare_cocostuff_seg.py \\
        --data_root /path/to/coco2017 \\
        --out_ctrl_dir /path/to/train_control \\
        --out_csv /path/to/train2017_cocostuff.csv \\
        --split train \\
        --num_workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Worker (runs in child process — no shared state)
# ---------------------------------------------------------------------------

def _process_one(task):
    """Resize panoptic PNG and save. Returns (out_path, ok)."""
    pan_src, out_path, size = task
    try:
        img = Image.open(pan_src).convert("RGB")
        img = img.resize((size, size), Image.NEAREST)  # NEAREST preserves segment IDs
        img.save(out_path)
        return out_path, True
    except Exception as e:
        return out_path, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Root dir with train2017/, panoptic_train2017/, annotations/")
    ap.add_argument("--out_ctrl_dir", required=True,
                    help="Output root for control maps (seg/ subdir will be created)")
    ap.add_argument("--out_csv", required=True,
                    help="Path to write the output CSV (file_name, captions)")
    ap.add_argument("--seg_name", default="seg_cocostuff",
                    help="Sub-directory name under out_ctrl_dir (default: seg_cocostuff)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--size", type=int, default=512,
                    help="Resize panoptic PNGs to this square size")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pan_dir   = data_root / f"panoptic_{args.split}2017"
    ann_dir   = data_root / "annotations"
    pan_json  = ann_dir / f"panoptic_{args.split}2017.json"
    cap_json  = ann_dir / f"captions_{args.split}2017.json"

    for p in [pan_dir, pan_json, cap_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    out_seg_dir = Path(args.out_ctrl_dir) / args.seg_name
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build image_id → {file_name, panoptic_file_name} ----
    print("Loading panoptic annotations...")
    with open(pan_json) as f:
        pan_data = json.load(f)

    id_to_imgfile = {img["id"]: img["file_name"] for img in pan_data["images"]}
    id_to_panfile = {ann["image_id"]: ann["file_name"] for ann in pan_data["annotations"]}

    # ---- Build image_id → first caption ----
    print("Loading captions...")
    with open(cap_json) as f:
        cap_data = json.load(f)

    id_to_caption = {}
    for ann in cap_data["annotations"]:
        iid = ann["image_id"]
        if iid not in id_to_caption:            # keep first caption only
            id_to_caption[iid] = ann["caption"]

    # ---- Build task list ----
    tasks = []
    rows  = []
    missing_pan = 0

    for img_info in pan_data["images"]:
        iid      = img_info["id"]
        img_file = img_info["file_name"]                  # e.g. 000000012345.jpg
        stem     = Path(img_file).stem                    # 000000012345

        pan_file = id_to_panfile.get(iid)
        if pan_file is None:
            missing_pan += 1
            continue

        pan_src  = pan_dir / pan_file
        out_path = out_seg_dir / f"{stem}.png"

        caption  = id_to_caption.get(iid, "")
        rows.append({"file_name": img_file, "captions": caption})

        if not out_path.exists():
            tasks.append((str(pan_src), str(out_path), args.size))

    if missing_pan:
        print(f"[warn] {missing_pan} images had no panoptic annotation — skipped")

    print(f"{len(rows)} images total | {len(tasks)} control maps to generate")

    # ---- Parallel processing ----
    if tasks:
        failed = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
            futs = {exe.submit(_process_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futs), total=len(tasks),
                            desc="generating seg maps"):
                _, ok = fut.result()
                if not ok:
                    failed += 1
        if failed:
            print(f"[warn] {failed} images failed to process")
    else:
        print("All control maps already exist — skipping generation")

    # ---- CSV ----
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"CSV written: {args.out_csv}  ({len(df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
