import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from controlnet_aux import HEDdetector, OpenposeDetector


# Optional MiDaS depth model
USE_DEPTH = True

if USE_DEPTH:
    import sys
    sys.path.append("/home/hpc/iwnt/iwnt134h/ZoeDepth")

    import torch
    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model
    from zoedepth.utils.misc import pil_to_batched_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config("zoedepth_nk", "infer")

    # Disable automatic resizing
    config.do_resize = False

    zoe = build_model(config).to(device).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_rgb(np_img, path):
    img = Image.fromarray(np_img.astype(np.uint8))
    img.save(path, "PNG")


def generate_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.stack([edges]*3, axis=-1)


def generate_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255)
    mag = mag.astype(np.uint8)
    return np.stack([mag]*3, axis=-1)


def generate_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return np.stack([lap]*3, axis=-1)


def generate_normals(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    normals = np.zeros((*gray.shape, 3), dtype=np.float32)
    normals[..., 0] = sobelx
    normals[..., 1] = sobely
    normals[..., 2] = 1.0
    normals = cv2.normalize(normals, None, 0, 255, cv2.NORM_MINMAX)
    return normals.astype(np.uint8)


def generate_depth(img):
    """
    img: numpy array uint8 [H, W, 3] in RGB
    Returns: depth map as RGB numpy [H, W, 3] uint8 - same size as input
    """
    h, w = img.shape[:2]  # Store original dimensions

    # Convert numpy → PIL → tensor
    pil_img = Image.fromarray(img)
    t = pil_to_batched_tensor(pil_img).to(device)

    with torch.no_grad():
        output = zoe(t)

    depth = output['metric_depth'].squeeze().cpu().numpy()  # [H', W']

    # Resize back to original dimensions if needed
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255]
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # Create RGB for consistency with your dataset
    depth_rgb = np.stack([depth_norm]*3, axis=-1)

    return depth_rgb


def generate_hed(img):
    """img: numpy uint8 [H, W, 3] RGB. Returns HED edge map as RGB numpy [H, W, 3] uint8."""
    h, w = img.shape[:2]
    pil_img = Image.fromarray(img)
    hed_map = hed_detector(pil_img, detect_resolution=min(h, w), image_resolution=min(h, w))
    return np.array(hed_map.convert("RGB"))


def generate_openpose(img):
    """img: numpy uint8 [H, W, 3] RGB. Returns pose map as RGB numpy [H, W, 3] uint8."""
    h, w = img.shape[:2]
    pil_img = Image.fromarray(img)
    pose_map = openpose_detector(pil_img, detect_resolution=min(h, w), image_resolution=min(h, w))
    return np.array(pose_map.convert("RGB"))


def main(root_dir, subset="train2014"):
    input_dir = os.path.join(root_dir, subset)
    control_root = os.path.join(root_dir, "val_control" if "val" in subset else "train_control")
    print(f"Generating control images for {subset} and saving them to {control_root}")

    input_files = [
        f for f in os.listdir(input_dir)
        if f.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    ]

    # Prepare control folders
    folders = [
        "canny", "sobel", "laplacian", "normals", "gray", "hed", "openpose"
    ] + (["depth"] if USE_DEPTH else [])

    for f in folders:
        ensure_dir(os.path.join(control_root, f))

    for fname in tqdm(input_files, desc="Generating controls"):
        file_id = os.path.splitext(fname)[0]
        img_path = os.path.join(input_dir, fname)

        # Determine which controls still need to be generated
        tasks = {
            "gray":      lambda img: np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)]*3, axis=-1),
            "canny":     generate_canny,
            "sobel":     generate_sobel,
            "laplacian": generate_laplacian,
            "normals":   generate_normals,
            "hed":       generate_hed,
            "openpose":  generate_openpose,
        }
        if USE_DEPTH:
            tasks["depth"] = generate_depth

        pending = {
            ctrl: fn for ctrl, fn in tasks.items()
            if not os.path.exists(os.path.join(control_root, ctrl, file_id + ".png"))
        }

        if not pending:
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        for ctrl, fn in pending.items():
            save_rgb(fn(img), os.path.join(control_root, ctrl, file_id + ".png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default="train2014")
    args = parser.parse_args()
    main(args.root_dir, args.subset)
