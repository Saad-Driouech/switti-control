# utils/control_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2


# ============================================================
# NORMALIZATION & CONVERSION UTILITIES
# ============================================================

def _tensor_to_pil(tensor):
    """
    Convert tensor in [-1, 1] to PIL Image in RGB [0, 255] uint8.

    NO HEURISTICS. Assumes tensor is ALWAYS in [-1, 1] (from training pipeline).

    Args:
        tensor: Tensor (C, H, W) or (1, C, H, W) in float32 [-1, 1]

    Returns:
        PIL Image in RGB mode, uint8 [0, 255]
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        else:
            raise ValueError(f"Cannot convert batched tensor: {tensor.shape}")

    if tensor.ndim != 3:
        raise ValueError(f"Expected (C, H, W) tensor, got {tensor.shape}")

    # [-1, 1] → [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)

    # Convert to numpy HWC [0, 255] uint8
    tensor = tensor.detach().cpu()
    img_np = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # Handle grayscale (C=1)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)  # (H, W, 1) -> (H, W)
        return Image.fromarray(img_np, mode='L').convert('RGB')
    else:
        return Image.fromarray(img_np, mode='RGB')


def _ensure_pil(img):
    """
    Ensure input is PIL Image.

    For generated images: Already PIL, just return.
    For control images: Convert from tensor.

    Args:
        img: PIL Image or Tensor

    Returns:
        PIL Image in RGB mode
    """
    if isinstance(img, Image.Image):
        return img.convert('RGB')
    elif isinstance(img, torch.Tensor):
        return _tensor_to_pil(img)
    else:
        raise TypeError(f"Expected PIL Image or Tensor, got {type(img)}")


# ============================================================
# LAZY MODEL LOADERS
# ============================================================

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


# ============================================================
# SEGMENTATION COLOR PALETTE (matches control_preprocessor.py)
# ============================================================

def _build_coco_palette():
    """Build COCO category_id → RGB and RGB → category_id mappings.
    Must match _coco_category_color() in control_preprocessor.py exactly."""
    id_to_color = {}
    color_to_id = {}
    for cat_id in range(1, 91):  # COCO IDs 1-90
        np.random.seed(cat_id)
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        id_to_color[cat_id] = color
        color_to_id[color] = cat_id
    return id_to_color, color_to_id


_COCO_ID_TO_COLOR, _COCO_COLOR_TO_ID = _build_coco_palette()


def _decode_seg_control(ctrl_np: np.ndarray) -> np.ndarray:
    """
    Decode a colorized segmentation control map to a per-pixel COCO category ID map.

    Uses the same deterministic color scheme as _coco_category_color() in
    control_preprocessor.py. Background / unannotated pixels map to 0.

    Args:
        ctrl_np: uint8 [H, W, 3] RGB numpy array

    Returns:
        int64 [H, W] array of COCO category IDs (0 = background)
    """
    h, w = ctrl_np.shape[:2]
    # Encode each pixel as a 24-bit integer for fast vectorised lookup
    encoded = (
        ctrl_np[:, :, 0].astype(np.int32) * 65536
        + ctrl_np[:, :, 1].astype(np.int32) * 256
        + ctrl_np[:, :, 2].astype(np.int32)
    )
    cat_map = np.zeros((h, w), dtype=np.int64)
    for (r, g, b), cat_id in _COCO_COLOR_TO_ID.items():
        color_enc = r * 65536 + g * 256 + b
        cat_map[encoded == color_enc] = cat_id
    return cat_map


# ============================================================
# GRAYSCALE CONTROL METRICS
# ============================================================

def calculate_ssim(generated_images, control_images):
    """
    Calculate SSIM between generated and control images in grayscale domain.

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])

    Returns:
        float: Mean SSIM score [0, 1], higher is better (1.0 = perfect match)
    """
    ssim_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)

        # Convert to grayscale numpy arrays
        gen_gray = np.array(gen_img.convert('L')).astype(np.float64)  # [0, 255]
        ctrl_gray = np.array(ctrl_img.convert('L')).astype(np.float64)  # [0, 255]

        # Calculate SSIM
        score = ssim(
            gen_gray,
            ctrl_gray,
            data_range=255.0,
            gaussian_weights=True,
            use_sample_covariance=False
        )
        ssim_scores.append(score)

    return float(np.mean(ssim_scores)) if ssim_scores else 0.0


# ============================================================
# CANNY EDGE CONTROL METRICS
# ============================================================

def calculate_edge_similarity(generated_images, control_images, threshold1=100, threshold2=200):
    """
    Calculate edge IoU for canny edge control.

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])
        threshold1, threshold2: Canny edge detection thresholds

    Returns:
        float: Edge IoU score [0, 1], higher is better
    """
    edge_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)

        # Extract edges from generated image
        gen_np = np.array(gen_img.convert('RGB'))
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        gen_edges = cv2.Canny(gen_gray, threshold1, threshold2)

        # Convert control to grayscale edge map
        ctrl_gray = np.array(ctrl_img.convert('L'))

        # Binarize both (non-zero = edge)
        gen_binary = (gen_edges > 0).astype(bool)
        ctrl_binary = (ctrl_gray > 127).astype(bool)  # Threshold at mid-gray

        # Calculate IoU
        intersection = np.logical_and(gen_binary, ctrl_binary).sum()
        union = np.logical_or(gen_binary, ctrl_binary).sum()

        iou = float(intersection) / float(union) if union > 0 else 0.0
        edge_scores.append(iou)

    return float(np.mean(edge_scores)) if edge_scores else 0.0


def calculate_edge_f1(generated_images, control_images, threshold1=100, threshold2=200):
    """
    Calculate edge F1 score (harmonic mean of precision & recall).

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])
        threshold1, threshold2: Canny thresholds

    Returns:
        dict: {'f1': float, 'precision': float, 'recall': float}
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)

        # Extract edges from generated
        gen_np = np.array(gen_img.convert('RGB'))
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        gen_edges = cv2.Canny(gen_gray, threshold1, threshold2)

        # Convert control to grayscale edge map
        ctrl_gray = np.array(ctrl_img.convert('L'))

        # Binarize both
        gen_binary = (gen_edges > 0).astype(bool)
        ctrl_binary = (ctrl_gray > 127).astype(bool)

        # True positives, false positives, false negatives
        tp = np.logical_and(gen_binary, ctrl_binary).sum()
        fp = np.logical_and(gen_binary, ~ctrl_binary).sum()
        fn = np.logical_and(~gen_binary, ctrl_binary).sum()

        # Precision and Recall
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    if not f1_scores:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    return {
        'f1': float(np.mean(f1_scores)),
        'precision': float(np.mean(precision_scores)),
        'recall': float(np.mean(recall_scores)),
    }


# ============================================================
# DEPTH CONTROL METRICS
# ============================================================

def calculate_depth_metrics(generated_images, control_images, device='cuda'):
    """
    Estimate depth from generated images using ZoeDepth and compare to control depth maps.

    Mirrors the preprocessing pipeline exactly: the same ZoeDepth (zoedepth_nk) model
    used to produce the control depth maps is run on the generated image.

    Follows the scale-invariant depth evaluation protocol from Eigen et al. 2014,
    adapted for normalized depth (both maps normalized to [0,1] before comparison
    since no absolute metric scale is available at evaluation time).

    Metrics:
        'depth_abs_rel': Mean absolute relative error (lower is better)
        'depth_rmse':    RMSE on normalized depth (lower is better)
        'depth_delta1':  Fraction of pixels where max(pred/gt, gt/pred) < 1.25
                         (higher is better; standard δ₁ accuracy)

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images:   List of Tensors (3, 512, 512, float32 [-1,1])
        device:           torch device for ZoeDepth inference

    Returns:
        dict: {'depth_abs_rel': float, 'depth_rmse': float, 'depth_delta1': float}
    """
    from zoedepth.utils.misc import pil_to_batched_tensor

    zoe = _get_depth_estimator(device)

    abs_rel_scores = []
    rmse_scores = []
    delta1_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        gen_pil = _ensure_pil(gen_img)
        ctrl_pil = _ensure_pil(ctrl_img)
        h, w = gen_pil.height, gen_pil.width

        # Run ZoeDepth on generated image — same model used in preprocessing
        t = pil_to_batched_tensor(gen_pil).to(device)
        with torch.no_grad():
            output = zoe(t)
        gen_depth = output['metric_depth'].squeeze().cpu().numpy()  # [H', W'] float
        if gen_depth.shape != (h, w):
            import cv2
            gen_depth = cv2.resize(gen_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Control depth: grayscale [0,255] (stored as 3-channel RGB, take one channel)
        ctrl_depth = np.array(ctrl_pil.convert('L')).astype(np.float64)  # [H, W]

        # Normalize both to [0,1] independently (scale-invariant comparison)
        gen_norm = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
        ctrl_norm = (ctrl_depth - ctrl_depth.min()) / (ctrl_depth.max() - ctrl_depth.min() + 1e-8)

        # Only evaluate over pixels where control depth is non-trivial
        eps = 1e-8
        valid = ctrl_norm > eps
        if not valid.any():
            continue

        # AbsRel: mean(|pred - gt| / gt) — standard Eigen et al. metric
        abs_rel = float(np.mean(np.abs(gen_norm[valid] - ctrl_norm[valid]) / ctrl_norm[valid]))
        abs_rel_scores.append(abs_rel)

        # RMSE on normalized depth
        rmse = float(np.sqrt(np.mean((gen_norm - ctrl_norm) ** 2)))
        rmse_scores.append(rmse)

        # δ < 1.25 threshold accuracy
        ratio = np.maximum(
            gen_norm[valid] / (ctrl_norm[valid] + eps),
            ctrl_norm[valid] / (gen_norm[valid] + eps),
        )
        delta1_scores.append(float(np.mean(ratio < 1.25)))

    return {
        'depth_abs_rel': float(np.mean(abs_rel_scores)) if abs_rel_scores else 0.0,
        'depth_rmse':    float(np.mean(rmse_scores))    if rmse_scores    else 0.0,
        'depth_delta1':  float(np.mean(delta1_scores))  if delta1_scores  else 0.0,
    }


# ============================================================
# SURFACE NORMAL METRICS
# ============================================================

def calculate_normal_metrics(generated_images, control_images):
    """
    Estimate surface normals from generated images using NormalBae and compare
    to control normal maps.

    Mirrors the preprocessing pipeline exactly: the same NormalBaeDetector used to
    produce the control normal maps is run on the generated image, giving estimated
    normals that can be meaningfully compared to the control normals.

    Standard metrics from surface normal estimation literature
    (Eigen & Fergus 2015, Wang et al. 2015).

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images:   List of Tensors (3, 512, 512, float32 [-1,1])

    Returns:
        dict:
            'normal_mae_deg':    Mean angular error in degrees (lower is better)
            'normal_cosine_sim': Mean cosine similarity in [-1, 1] (higher is better)
    """
    detector = _get_normal_estimator()

    mae_scores = []
    cos_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        gen_pil = _ensure_pil(gen_img)
        ctrl_pil = _ensure_pil(ctrl_img)
        h, w = gen_pil.height, gen_pil.width

        # Run NormalBae on generated image — same detector used in preprocessing
        gen_normal_pil = detector(gen_pil, detect_resolution=min(h, w), image_resolution=min(h, w))
        gen_np  = np.array(gen_normal_pil.convert('RGB')).astype(np.float32)   # [H, W, 3] [0,255]
        ctrl_np = np.array(ctrl_pil).astype(np.float32)                         # [H, W, 3] [0,255]

        # Decode RGB → normal vector in [-1, 1]  (NormalBae encoding: n = pixel/127.5 - 1)
        gen_normals  = gen_np  / 127.5 - 1.0  # [H, W, 3]
        ctrl_normals = ctrl_np / 127.5 - 1.0  # [H, W, 3]

        # L2-normalise (handle zero vectors)
        gen_norms  = np.linalg.norm(gen_normals,  axis=-1, keepdims=True).clip(min=1e-8)
        ctrl_norms = np.linalg.norm(ctrl_normals, axis=-1, keepdims=True).clip(min=1e-8)
        gen_unit  = gen_normals  / gen_norms
        ctrl_unit = ctrl_normals / ctrl_norms

        # Cosine similarity per pixel, then mean
        cos_sim = np.sum(gen_unit * ctrl_unit, axis=-1).clip(-1.0, 1.0)  # [H, W]
        cos_scores.append(float(np.mean(cos_sim)))

        # Angular error in degrees
        mae_scores.append(float(np.mean(np.degrees(np.arccos(cos_sim)))))

    return {
        'normal_mae_deg':    float(np.mean(mae_scores)) if mae_scores else 0.0,
        'normal_cosine_sim': float(np.mean(cos_scores)) if cos_scores else 0.0,
    }


# ============================================================
# HED EDGE METRICS
# ============================================================

def calculate_hed_metrics(generated_images, control_images):
    """
    Re-detect HED edges on generated images and compare to control HED maps.

    Standard evaluation mirrors the BSDS500 F-measure benchmark:
    edges are extracted from the generated image using the same HED detector
    used during preprocessing, then compared pixel-wise to the control map.

    Metrics:
        'hed_ssim': SSIM between extracted and control soft edge maps (higher is better)
        'hed_f1':   Edge F1 on binarized maps (higher is better)

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images:   List of Tensors (3, 512, 512, float32 [-1,1])

    Returns:
        dict: {'hed_ssim': float, 'hed_f1': float}
    """
    detector = _get_hed_detector()

    ssim_scores = []
    f1_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        gen_pil = _ensure_pil(gen_img)
        ctrl_pil = _ensure_pil(ctrl_img)
        h, w = gen_pil.height, gen_pil.width

        # Extract HED from generated image
        gen_hed_pil = detector(gen_pil, detect_resolution=min(h, w), image_resolution=min(h, w))
        gen_gray = np.array(gen_hed_pil.convert('L')).astype(np.float64)  # [H, W]

        ctrl_gray = np.array(ctrl_pil.convert('L')).astype(np.float64)    # [H, W]

        # SSIM on soft edge maps
        score = ssim(gen_gray, ctrl_gray, data_range=255.0,
                     gaussian_weights=True, use_sample_covariance=False)
        ssim_scores.append(score)

        # Edge F1 on binarized maps (threshold at mid-gray)
        gen_bin = (gen_gray > 127.0).astype(bool)
        ctrl_bin = (ctrl_gray > 127.0).astype(bool)
        tp = np.logical_and(gen_bin, ctrl_bin).sum()
        fp = np.logical_and(gen_bin, ~ctrl_bin).sum()
        fn = np.logical_and(~gen_bin, ctrl_bin).sum()
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return {
        'hed_ssim': float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        'hed_f1':   float(np.mean(f1_scores))   if f1_scores   else 0.0,
    }


# ============================================================
# OPENPOSE / SKELETON METRICS
# ============================================================

def calculate_pose_metrics(generated_images, control_images):
    """
    Re-detect body skeleton on generated images and compare to control pose maps.

    Images where the control map is entirely black (no person detected) are
    skipped so they do not drag the mean down.

    Metrics:
        'pose_ssim':        SSIM on rendered skeleton maps (higher is better).
                            Standard proxy metric used in ControlNet and T2I-Adapter papers.
        'pose_skeleton_f1': Skeleton-pixel F1 on binarized maps (higher is better).

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images:   List of Tensors (3, 512, 512, float32 [-1,1])

    Returns:
        dict: {'pose_ssim': float, 'pose_skeleton_f1': float}
    """
    detector = _get_openpose_detector()

    ssim_scores = []
    f1_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        ctrl_pil = _ensure_pil(ctrl_img)
        ctrl_gray = np.array(ctrl_pil.convert('L'))

        # Skip frames with no person in the control map (all black)
        if ctrl_gray.max() == 0:
            continue

        gen_pil = _ensure_pil(gen_img)
        h, w = gen_pil.height, gen_pil.width

        # Re-detect pose on generated image
        gen_pose_pil = detector(gen_pil, detect_resolution=min(h, w), image_resolution=min(h, w))
        gen_gray = np.array(gen_pose_pil.convert('L')).astype(np.float64)
        ctrl_gray_f = ctrl_gray.astype(np.float64)

        # SSIM on rendered skeleton maps
        score = ssim(gen_gray, ctrl_gray_f, data_range=255.0,
                     gaussian_weights=True, use_sample_covariance=False)
        ssim_scores.append(score)

        # Skeleton pixel F1 — low threshold because skeleton lines are thin and bright
        gen_bin  = (gen_gray  > 10.0).astype(bool)
        ctrl_bin = (ctrl_gray_f > 10.0).astype(bool)
        tp = np.logical_and(gen_bin, ctrl_bin).sum()
        fp = np.logical_and(gen_bin, ~ctrl_bin).sum()
        fn = np.logical_and(~gen_bin, ctrl_bin).sum()
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return {
        'pose_ssim':        float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        'pose_skeleton_f1': float(np.mean(f1_scores))   if f1_scores   else 0.0,
    }


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def calculate_seg_metrics(generated_images, control_images, device='cuda'):
    """
    Compute segmentation adherence using a COCO-pretrained Mask R-CNN.

    Pipeline:
        1. Decode the colorized control map back to per-pixel COCO category IDs
           using the deterministic palette from control_preprocessor.py.
        2. Run MaskRCNN-ResNet50-FPN (COCO-pretrained, torchvision) on the
           generated image to obtain predicted instance masks + category labels.
        3. Merge predicted masks into a per-pixel predicted category map.
        4. Compute mIoU over categories present in the control map, and
           pixel accuracy over annotated (non-background) pixels.

    Images where the control map is entirely black (no annotations) are skipped.

    Metrics:
        'seg_miou':      Mean IoU over present COCO categories (higher is better).
        'seg_pixel_acc': Pixel accuracy over annotated pixels (higher is better).

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images:   List of Tensors (3, 512, 512, float32 [-1,1])
        device:           torch device for Mask R-CNN inference

    Returns:
        dict: {'seg_miou': float, 'seg_pixel_acc': float}
    """
    from torchvision.transforms.functional import to_tensor as tvf_to_tensor

    model = _get_seg_model(device)

    miou_scores = []
    pixel_acc_scores = []

    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue

        ctrl_np = np.array(_ensure_pil(ctrl_img))  # [H, W, 3] uint8

        # Skip unannotated frames
        if ctrl_np.max() == 0:
            continue

        # Decode control colors → COCO category ID map
        ctrl_cat_map = _decode_seg_control(ctrl_np)  # [H, W] int64

        present_cats = set(np.unique(ctrl_cat_map)) - {0}
        if not present_cats:
            continue

        # Run Mask R-CNN on generated image
        gen_pil = _ensure_pil(gen_img)
        gen_tensor = tvf_to_tensor(gen_pil).unsqueeze(0).to(device)  # [1, 3, H, W] float [0,1]

        with torch.no_grad():
            output = model(gen_tensor)[0]

        # Build predicted category map (higher-confidence instances win)
        h, w = ctrl_cat_map.shape
        pred_cat_map = np.zeros((h, w), dtype=np.int64)

        masks  = output['masks'].squeeze(1).cpu().numpy()   # [N, H, W] float [0,1]
        labels = output['labels'].cpu().numpy()             # [N] int
        scores = output['scores'].cpu().numpy()             # [N] float

        # Filter low-confidence detections
        keep = scores > 0.5
        for mask, label in zip(masks[keep], labels[keep]):
            pred_cat_map[(mask > 0.5)] = int(label)

        # mIoU over categories present in control
        ious = []
        for cat_id in present_cats:
            gt_mask   = (ctrl_cat_map == cat_id)
            pred_mask = (pred_cat_map == cat_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union        = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                ious.append(float(intersection) / float(union))

        if ious:
            miou_scores.append(float(np.mean(ious)))

        # Pixel accuracy over annotated pixels only
        annotated = ctrl_cat_map > 0
        if annotated.sum() > 0:
            correct = np.sum((ctrl_cat_map == pred_cat_map) & annotated)
            pixel_acc_scores.append(float(correct) / float(annotated.sum()))

    return {
        'seg_miou':      float(np.mean(miou_scores))      if miou_scores      else 0.0,
        'seg_pixel_acc': float(np.mean(pixel_acc_scores)) if pixel_acc_scores else 0.0,
    }


# ============================================================
# DISPATCHER FUNCTION
# ============================================================

def calculate_control_metrics(generated_images, control_dict, control_type, device='cuda'):
    """
    Dispatcher function - calls appropriate metrics based on control type.

    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_dict: Dict {control_type: List[Tensor(3,512,512, float32 [-1,1])]}
        control_type: str, one of ['gray', 'canny', 'depth', 'normals', 'hed', 'openpose', 'seg']
        device: torch device (used for segmentation model)

    Returns:
        dict: {metric_name: score}
    """
    if control_type not in control_dict:
        return {}

    control_images = control_dict[control_type]

    # Filter out None values
    valid_pairs = [(g, c) for g, c in zip(generated_images, control_images) if c is not None]

    if not valid_pairs:
        return {}

    valid_gen, valid_ctrl = zip(*valid_pairs)
    valid_gen = list(valid_gen)
    valid_ctrl = list(valid_ctrl)

    metrics = {}

    try:
        if control_type == 'gray':
            # Structural similarity in grayscale domain
            metrics['ssim'] = calculate_ssim(valid_gen, valid_ctrl)

        elif control_type == 'canny':
            # Edge IoU (primary metric)
            metrics['edge_iou'] = calculate_edge_similarity(valid_gen, valid_ctrl)
            # Edge F1, precision, recall (diagnostic metrics)
            edge_f1_results = calculate_edge_f1(valid_gen, valid_ctrl)
            metrics['edge_f1']        = edge_f1_results['f1']
            metrics['edge_precision'] = edge_f1_results['precision']
            metrics['edge_recall']    = edge_f1_results['recall']

        elif control_type == 'depth':
            # AbsRel, RMSE, δ<1.25 on ZoeDepth-estimated depth (Eigen et al. 2014 protocol)
            metrics.update(calculate_depth_metrics(valid_gen, valid_ctrl, device=device))

        elif control_type == 'normals':
            # Mean angular error + cosine similarity (standard in surface normal estimation)
            metrics.update(calculate_normal_metrics(valid_gen, valid_ctrl))

        elif control_type == 'hed':
            # SSIM + edge F1 on HED maps re-extracted from generated images
            metrics.update(calculate_hed_metrics(valid_gen, valid_ctrl))

        elif control_type == 'openpose':
            # Skeleton SSIM + skeleton pixel F1
            metrics.update(calculate_pose_metrics(valid_gen, valid_ctrl))

        elif control_type == 'seg':
            # mIoU + pixel accuracy via COCO Mask R-CNN
            metrics.update(calculate_seg_metrics(valid_gen, valid_ctrl, device=device))

    except Exception as e:
        print(f"[ERROR] Control metric calculation failed for {control_type}: {e}")
        import traceback
        traceback.print_exc()
        return {}

    return metrics
