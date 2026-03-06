"""
Offline extraction of spatial control maps from images.

Supports:
    canny  -- cv2.Canny with adaptive thresholds (0.1×median, 0.4×median pixel value)
    depth  -- MiDaS DPT-Hybrid

Usage::

    python scripts/precompute_control_maps.py \\
        --data_path /data/coco \\
        --output_dir /data/ctrl_maps \\
        --modalities canny depth \\
        --subset train2014 \\
        --device cuda

Output layout::

    <output_dir>/<modality>/<subset>/<image_id>.png
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff", ".webp"}


def extract_canny(pil_img: Image.Image) -> np.ndarray:
    """Adaptive Canny edge detection."""
    gray = np.array(pil_img.convert("L"))
    median = float(np.median(gray))
    low_thresh = int(max(0, 0.1 * median))
    high_thresh = int(min(255, 0.4 * median))
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return np.stack([edges] * 3, axis=-1)


def build_midas(device: str):
    """Load MiDaS DPT-Hybrid model and transform."""
    try:
        import torch
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas.eval().to(device)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        return midas, transform
    except Exception as e:
        raise RuntimeError(f"Failed to load MiDaS: {e}") from e


def extract_depth(pil_img: Image.Image, midas, transform, device: str) -> np.ndarray:
    """Estimate depth with MiDaS and return as RGB uint8."""
    img_np = np.array(pil_img.convert("RGB"))
    input_batch = transform(img_np).to(device)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=pil_img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_np = depth.cpu().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    depth_u8 = (depth_np * 255).astype(np.uint8)
    return np.stack([depth_u8] * 3, axis=-1)


def process_dataset(
    data_path: str,
    output_dir: str,
    modalities: list,
    subset: str,
    device: str,
    batch_size: int = 1,
):
    img_dir = Path(data_path) / subset
    all_imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in EXTENSIONS],
        key=lambda p: p.stem,
    )
    print(f"Found {len(all_imgs)} images in {img_dir}")

    midas, midas_transform = None, None
    if "depth" in modalities:
        print("Loading MiDaS DPT-Hybrid...")
        midas, midas_transform = build_midas(device)

    for modality in modalities:
        out_dir = Path(output_dir) / modality / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{modality}] Writing to {out_dir}")

        for img_path in tqdm(all_imgs, desc=modality):
            out_path = out_dir / f"{img_path.stem}.png"
            if out_path.exists():
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            if modality == "canny":
                arr = extract_canny(pil_img)
            elif modality == "depth":
                arr = extract_depth(pil_img, midas, midas_transform, device)
            else:
                print(f"  Skipping unsupported modality '{modality}'")
                break

            Image.fromarray(arr.astype(np.uint8)).save(out_path)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Precompute spatial control maps")
    parser.add_argument("--data_path", required=True, help="Root COCO directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for ctrl maps")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["canny"],
        choices=["canny", "depth"],
        help="Which control maps to extract",
    )
    parser.add_argument("--subset", default="train2014", help="Dataset subset name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    process_dataset(
        data_path=args.data_path,
        output_dir=args.output_dir,
        modalities=args.modalities,
        subset=args.subset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
