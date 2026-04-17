"""
Control-specific evaluation metrics for spatially-conditioned generation.

Dispatches to the appropriate metric set based on control type:
  - canny:    Edge IoU, Edge F1/precision/recall
  - depth:    AbsRel, RMSE, δ<1.25 (ZoeDepth estimated, Eigen et al. 2014)
  - gray:     SSIM
  - normals:  Mean angular error (degrees), cosine similarity
  - hed:      SSIM on soft edge maps, edge F1
  - openpose: Skeleton SSIM, skeleton pixel F1
  - seg:      mIoU, pixel accuracy (via COCO Mask R-CNN)
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
# Lazy model loaders
# ─────────────────────────────────────────────────────────────────────────────

_hed_detector = None
_depth_estimator = None
_depth_estimator_device = None
_normal_estimator = None
_openpose_detector = None
_seg_model = None
_seg_model_device = None


def _get_hed_detector():
    global _hed_detector
    if _hed_detector is None:
        from controlnet_aux import HEDdetector
        _hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return _hed_detector


def _get_depth_estimator(device):
    global _depth_estimator, _depth_estimator_device
    if _depth_estimator is None or _depth_estimator_device != str(device):
        from zoedepth.utils.config import get_config
        from zoedepth.models.builder import build_model
        config = get_config("zoedepth_nk", "infer")
        config.do_resize = False
        _depth_estimator = build_model(config).to(device).eval()
        _depth_estimator_device = str(device)
    return _depth_estimator


def _get_normal_estimator():
    global _normal_estimator
    if _normal_estimator is None:
        from controlnet_aux import NormalBaeDetector
        _normal_estimator = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    return _normal_estimator


def _get_openpose_detector():
    global _openpose_detector
    if _openpose_detector is None:
        from controlnet_aux import OpenposeDetector
        _openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _openpose_detector


# ─────────────────────────────────────────────────────────────────────────────
# Depth metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_depth_metrics(generated_images, control_images, device="cuda") -> dict:
    """
    Estimate depth from generated images using ZoeDepth and compare to control depth maps.

    Mirrors the preprocessing pipeline: the same ZoeDepth (zoedepth_nk) model used to
    produce the control depth maps is run on each generated image.

    Follows the scale-invariant evaluation protocol from Eigen et al. 2014, adapted for
    normalized depth (both maps normalized to [0,1] independently).

    Returns:
        {'depth_abs_rel': float (lower is better),
         'depth_rmse':    float (lower is better),
         'depth_delta1':  float (higher is better; δ < 1.25 threshold accuracy)}
    """
    import cv2
    from zoedepth.utils.misc import pil_to_batched_tensor

    zoe = _get_depth_estimator(device)
    abs_rel_scores, rmse_scores, delta1_scores = [], [], []

    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_pil  = _ensure_pil(gen)
        ctrl_pil = _ensure_pil(ctrl)
        h, w = gen_pil.height, gen_pil.width

        # Run ZoeDepth on generated image — same model used in preprocessing
        t = pil_to_batched_tensor(gen_pil).to(device)
        with torch.no_grad():
            output = zoe(t)
        gen_depth = output["metric_depth"].squeeze().cpu().numpy().astype(np.float64)
        if gen_depth.shape != (h, w):
            gen_depth = cv2.resize(gen_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        ctrl_depth = np.array(ctrl_pil.convert("L")).astype(np.float64)

        # Normalize both to [0,1] independently (scale-invariant comparison)
        gen_norm  = (gen_depth  - gen_depth.min())  / (gen_depth.max()  - gen_depth.min()  + 1e-8)
        ctrl_norm = (ctrl_depth - ctrl_depth.min()) / (ctrl_depth.max() - ctrl_depth.min() + 1e-8)

        eps = 1e-8
        valid = ctrl_norm > eps
        if not valid.any():
            continue

        abs_rel_scores.append(float(np.mean(np.abs(gen_norm[valid] - ctrl_norm[valid]) / ctrl_norm[valid])))
        rmse_scores.append(float(np.sqrt(np.mean((gen_norm[valid] - ctrl_norm[valid]) ** 2))))
        ratio = np.maximum(gen_norm[valid] / (ctrl_norm[valid] + eps),
                           ctrl_norm[valid] / (gen_norm[valid]  + eps))
        delta1_scores.append(float(np.mean(ratio < 1.25)))

    return {
        "depth_abs_rel": float(np.mean(abs_rel_scores)) if abs_rel_scores else 0.0,
        "depth_rmse":    float(np.mean(rmse_scores))    if rmse_scores    else 0.0,
        "depth_delta1":  float(np.mean(delta1_scores))  if delta1_scores  else 0.0,
    }


def _get_seg_model(device):
    global _seg_model, _seg_model_device
    if _seg_model is None or _seg_model_device != str(device):
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn,
            MaskRCNN_ResNet50_FPN_Weights,
        )
        _seg_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        _seg_model.eval()
        _seg_model = _seg_model.to(device)
        _seg_model_device = str(device)
    return _seg_model


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation color palette (must match control_preprocessor.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _build_coco_palette():
    """Build COCO category_id ↔ RGB mappings using the same seeded RNG as the preprocessor."""
    id_to_color, color_to_id = {}, {}
    for cat_id in range(1, 91):
        np.random.seed(cat_id)
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        id_to_color[cat_id] = color
        color_to_id[color] = cat_id
    return id_to_color, color_to_id


_COCO_ID_TO_COLOR, _COCO_COLOR_TO_ID = _build_coco_palette()


def _decode_seg_control(ctrl_np: np.ndarray) -> np.ndarray:
    """Decode colorized segmentation control map → per-pixel COCO category ID (0 = background)."""
    encoded = (
        ctrl_np[:, :, 0].astype(np.int32) * 65536
        + ctrl_np[:, :, 1].astype(np.int32) * 256
        + ctrl_np[:, :, 2].astype(np.int32)
    )
    cat_map = np.zeros(ctrl_np.shape[:2], dtype=np.int64)
    for (r, g, b), cat_id in _COCO_COLOR_TO_ID.items():
        cat_map[encoded == r * 65536 + g * 256 + b] = cat_id
    return cat_map


# ─────────────────────────────────────────────────────────────────────────────
# Surface normal metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_normal_metrics(generated_images, control_images) -> dict:
    """
    Estimate surface normals from generated images using NormalBae and compare
    to control normal maps.

    Mirrors the preprocessing pipeline: the same NormalBaeDetector used to produce the
    control normal maps is run on each generated image.

    Standard metrics from Eigen & Fergus 2015 / Wang et al. 2015.

    Returns:
        {'normal_mae_deg':    float (lower is better),
         'normal_cosine_sim': float (higher is better)}
    """
    detector = _get_normal_estimator()
    mae_scores, cos_scores = [], []

    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_pil  = _ensure_pil(gen)
        ctrl_pil = _ensure_pil(ctrl)
        h, w = gen_pil.height, gen_pil.width

        # Run NormalBae on generated image — same detector used in preprocessing
        gen_normal_pil = detector(gen_pil, detect_resolution=min(h, w), image_resolution=min(h, w))
        gen_np  = np.array(gen_normal_pil.convert("RGB")).astype(np.float32)
        ctrl_np = np.array(ctrl_pil).astype(np.float32)

        # Decode RGB → normal vector in [-1, 1]  (NormalBae encoding: n = pixel/127.5 - 1)
        gen_n  = gen_np  / 127.5 - 1.0
        ctrl_n = ctrl_np / 127.5 - 1.0

        gen_unit  = gen_n  / np.linalg.norm(gen_n,  axis=-1, keepdims=True).clip(min=1e-8)
        ctrl_unit = ctrl_n / np.linalg.norm(ctrl_n, axis=-1, keepdims=True).clip(min=1e-8)
        cos = np.clip(np.sum(gen_unit * ctrl_unit, axis=-1), -1.0, 1.0)
        cos_scores.append(float(np.mean(cos)))
        mae_scores.append(float(np.mean(np.degrees(np.arccos(cos)))))

    return {
        "normal_mae_deg":    float(np.mean(mae_scores)) if mae_scores else 0.0,
        "normal_cosine_sim": float(np.mean(cos_scores)) if cos_scores else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HED edge metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_hed_metrics(generated_images, control_images) -> dict:
    """
    Re-detect HED edges on generated images and compare to control HED maps.

    Mirrors BSDS500 F-measure evaluation style.

    Returns:
        {'hed_ssim': float (higher is better),
         'hed_f1':   float (higher is better)}
    """
    detector = _get_hed_detector()
    ssim_scores, f1_scores = [], []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        gen_pil  = _ensure_pil(gen)
        ctrl_pil = _ensure_pil(ctrl)
        h, w = gen_pil.height, gen_pil.width
        gen_hed  = np.array(detector(gen_pil, detect_resolution=min(h, w),
                                     image_resolution=min(h, w)).convert("L")).astype(np.float64)
        ctrl_gray = np.array(ctrl_pil.convert("L")).astype(np.float64)
        ssim_scores.append(ssim(gen_hed, ctrl_gray, data_range=255.0,
                                gaussian_weights=True, use_sample_covariance=False))
        gen_bin  = (gen_hed   > 127.0)
        ctrl_bin = (ctrl_gray > 127.0)
        tp = np.logical_and(gen_bin, ctrl_bin).sum()
        fp = np.logical_and(gen_bin, ~ctrl_bin).sum()
        fn = np.logical_and(~gen_bin, ctrl_bin).sum()
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {
        "hed_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "hed_f1":   float(np.mean(f1_scores))   if f1_scores   else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenPose / skeleton metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pose_metrics(generated_images, control_images) -> dict:
    """
    Re-detect body skeleton on generated images and compare to control pose maps.

    Frames with entirely black control maps (no person detected) are skipped.
    Standard proxy metrics used in ControlNet and T2I-Adapter papers.

    Returns:
        {'pose_ssim':        float (higher is better),
         'pose_skeleton_f1': float (higher is better)}
    """
    detector = _get_openpose_detector()
    ssim_scores, f1_scores = [], []
    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        ctrl_pil  = _ensure_pil(ctrl)
        ctrl_gray = np.array(ctrl_pil.convert("L"))
        if ctrl_gray.max() == 0:          # no person in control → skip
            continue
        gen_pil = _ensure_pil(gen)
        h, w = gen_pil.height, gen_pil.width
        gen_pose = np.array(detector(gen_pil, detect_resolution=min(h, w),
                                     image_resolution=min(h, w)).convert("L")).astype(np.float64)
        ctrl_f = ctrl_gray.astype(np.float64)
        ssim_scores.append(ssim(gen_pose, ctrl_f, data_range=255.0,
                                gaussian_weights=True, use_sample_covariance=False))
        # Low threshold: skeleton lines are thin and bright
        gen_bin  = (gen_pose > 10.0)
        ctrl_bin = (ctrl_f   > 10.0)
        tp = np.logical_and(gen_bin, ctrl_bin).sum()
        fp = np.logical_and(gen_bin, ~ctrl_bin).sum()
        fn = np.logical_and(~gen_bin, ctrl_bin).sum()
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {
        "pose_ssim":        float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "pose_skeleton_f1": float(np.mean(f1_scores))   if f1_scores   else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_seg_metrics(generated_images, control_images, device: str = "cuda") -> dict:
    """
    Segmentation adherence via COCO-pretrained Mask R-CNN (torchvision).

    Pipeline:
      1. Decode colorized control map → per-pixel COCO category IDs.
      2. Run MaskRCNN-ResNet50-FPN on generated image.
      3. Compute mIoU over categories present in control, and pixel accuracy
         over annotated (non-background) pixels.

    Frames with entirely black control maps are skipped.

    Returns:
        {'seg_miou':      float (higher is better),
         'seg_pixel_acc': float (higher is better)}
    """
    from torchvision.transforms.functional import to_tensor as tvf_to_tensor

    model = _get_seg_model(device)
    miou_scores, pixel_acc_scores = [], []

    for gen, ctrl in zip(generated_images, control_images):
        if ctrl is None:
            continue
        ctrl_np = np.array(_ensure_pil(ctrl))
        if ctrl_np.max() == 0:
            continue
        ctrl_cat = _decode_seg_control(ctrl_np)   # [H, W] int64
        present_cats = set(np.unique(ctrl_cat)) - {0}
        if not present_cats:
            continue

        gen_tensor = tvf_to_tensor(_ensure_pil(gen)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(gen_tensor)[0]

        h, w = ctrl_cat.shape
        pred_cat = np.zeros((h, w), dtype=np.int64)
        masks  = output["masks"].squeeze(1).cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        for mask, label in zip(masks[scores > 0.5], labels[scores > 0.5]):
            pred_cat[mask > 0.5] = int(label)

        ious = []
        for cat_id in present_cats:
            gt_m   = (ctrl_cat == cat_id)
            pred_m = (pred_cat == cat_id)
            inter  = np.logical_and(gt_m, pred_m).sum()
            union  = np.logical_or(gt_m, pred_m).sum()
            if union > 0:
                ious.append(float(inter) / float(union))
        if ious:
            miou_scores.append(float(np.mean(ious)))

        annotated = ctrl_cat > 0
        if annotated.sum() > 0:
            correct = np.sum((ctrl_cat == pred_cat) & annotated)
            pixel_acc_scores.append(float(correct) / float(annotated.sum()))

    return {
        "seg_miou":      float(np.mean(miou_scores))      if miou_scores      else 0.0,
        "seg_pixel_acc": float(np.mean(pixel_acc_scores)) if pixel_acc_scores else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def calculate_control_metrics(generated_images, control_tensors: list,
                               control_type: str, device: str = "cuda") -> dict:
    """
    Args:
        generated_images: list of PIL Images (RGB uint8)
        control_tensors:  list of (3, H, W) float32 tensors in [-1, 1]
        control_type:     'canny' | 'depth' | 'gray' | 'normals' | 'hed' | 'openpose' | 'seg'
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
            metrics.update(calculate_depth_metrics(gen_list, ctrl_list, device=device))
        elif control_type == "normals":
            metrics.update(calculate_normal_metrics(gen_list, ctrl_list))
        elif control_type == "hed":
            metrics.update(calculate_hed_metrics(gen_list, ctrl_list))
        elif control_type == "openpose":
            metrics.update(calculate_pose_metrics(gen_list, ctrl_list))
        elif control_type == "seg":
            metrics.update(calculate_seg_metrics(gen_list, ctrl_list, device=device))
    except Exception as e:
        import traceback
        print(f"[ERROR] Control metric failed for {control_type}: {e}")
        traceback.print_exc()
    return metrics
