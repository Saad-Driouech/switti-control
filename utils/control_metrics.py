"""
Control-specific evaluation metrics for spatially-conditioned generation.

Dispatches to the appropriate metric set based on control type:
  - canny:  Edge IoU, Edge F1/precision/recall
  - depth:  Pearson correlation, MAE, RMSE
  - gray:   SSIM
"""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────────────────────────────────────
# Conversion utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (C, H, W) float32 tensor in [-1, 1] to a uint8 RGB PIL Image.
    """
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected (C, H, W) tensor, got {tensor.shape}")
    tensor = ((tensor + 1.0) / 2.0).clamp(0, 1).detach().cpu()
    img_np = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    if img_np.shape[2] == 1:
        return Image.fromarray(img_np.squeeze(2), mode="L").convert("RGB")
    return Image.fromarray(img_np, mode="RGB")


def _ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        return _tensor_to_pil(img)
    raise TypeError(f"Expected PIL Image or Tensor, got {type(img)}")


# ─────────────────────────────────────────────────────────────────────────────
# SSIM (grayscale)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ssim(generated_images, control_images) -> float:
    scores = []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_gray = np.array(_ensure_pil(gen).convert("L")).astype(np.float64)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L")).astype(np.float64)
        scores.append(ssim(gen_gray, ctrl_gray, data_range=255.0,
                           gaussian_weights=True, use_sample_covariance=False))
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Canny edge metrics
# ─────────────────────────────────────────────────────────────────────────────

def _extract_edges(pil_img: Image.Image, t1: int = 100, t2: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, t1, t2)


def calculate_edge_similarity(generated_images, control_images,
                               threshold1: int = 100, threshold2: int = 200) -> float:
    """Edge IoU between generated-image edges and control edge map."""
    scores = []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_edges = (_extract_edges(_ensure_pil(gen), threshold1, threshold2) > 0)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L"))
        ctrl_edges = (ctrl_gray > 127)
        inter = np.logical_and(gen_edges, ctrl_edges).sum()
        union = np.logical_or(gen_edges, ctrl_edges).sum()
        scores.append(float(inter) / float(union) if union > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def calculate_edge_f1(generated_images, control_images,
                      threshold1: int = 100, threshold2: int = 200) -> dict:
    """Edge F1, precision, recall."""
    f1s, precs, recs = [], [], []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_edges = (_extract_edges(_ensure_pil(gen), threshold1, threshold2) > 0)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L"))
        ctrl_edges = (ctrl_gray > 127)
        tp = np.logical_and(gen_edges, ctrl_edges).sum()
        fp = np.logical_and(gen_edges, ~ctrl_edges).sum()
        fn = np.logical_and(~gen_edges, ctrl_edges).sum()
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    if not f1s:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Depth metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_depth_correlation(generated_images, control_images) -> float:
    correlations = []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_gray = np.array(_ensure_pil(gen).convert("L")).astype(np.float64)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L")).astype(np.float64)
        gen_norm = (gen_gray - gen_gray.min()) / (gen_gray.max() - gen_gray.min() + 1e-8)
        ctrl_norm = (ctrl_gray - ctrl_gray.min()) / (ctrl_gray.max() - ctrl_gray.min() + 1e-8)
        corr = np.corrcoef(gen_norm.flatten(), ctrl_norm.flatten())[0, 1]
        correlations.append(max(0.0, corr))
    return float(np.mean(correlations)) if correlations else 0.0


def calculate_depth_mae(generated_images, control_images) -> float:
    scores = []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_gray = np.array(_ensure_pil(gen).convert("L")).astype(np.float64)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L")).astype(np.float64)
        scores.append(np.mean(np.abs(gen_gray - ctrl_gray)))
    return float(np.mean(scores) / 255.0) if scores else 0.0


def calculate_depth_rmse(generated_images, control_images) -> float:
    scores = []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_gray = np.array(_ensure_pil(gen).convert("L")).astype(np.float64)
        ctrl_gray = np.array(_ensure_pil(ctrl).convert("L")).astype(np.float64)
        scores.append(np.sqrt(np.mean((gen_gray - ctrl_gray) ** 2)))
    return float(np.mean(scores) / 255.0) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def calculate_control_metrics(generated_images, control_tensors: list,
                               control_type: str, device: str = "cuda") -> dict:
    """
    Args:
        generated_images: list of PIL Images (RGB uint8)
        control_tensors:  list of (3, H, W) float32 tensors in [-1, 1]
        control_type:     'canny' | 'depth' | 'gray'
    Returns:
        dict of {metric_name: float}
    """
    valid = [(g, c) for g, c in zip(generated_images, control_tensors) if c is not None]
    if not valid:
        return {}
    gen_list, ctrl_list = zip(*valid)
    gen_list, ctrl_list = list(gen_list), list(ctrl_list)

    metrics = {}
    try:
        if control_type == "gray":
            metrics["ssim"] = calculate_ssim(gen_list, ctrl_list)
        elif control_type == "canny":
            metrics["edge_iou"] = calculate_edge_similarity(gen_list, ctrl_list)
            f1_res = calculate_edge_f1(gen_list, ctrl_list)
            metrics["edge_f1"] = f1_res["f1"]
            metrics["edge_precision"] = f1_res["precision"]
            metrics["edge_recall"] = f1_res["recall"]
        elif control_type == "depth":
            metrics["depth_corr"] = calculate_depth_correlation(gen_list, ctrl_list)
            metrics["depth_mae"] = calculate_depth_mae(gen_list, ctrl_list)
            metrics["depth_rmse"] = calculate_depth_rmse(gen_list, ctrl_list)
    except Exception as e:
        import traceback
        print(f"[ERROR] Control metric failed for {control_type}: {e}")
        traceback.print_exc()
    return metrics
