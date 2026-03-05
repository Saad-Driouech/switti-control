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

def calculate_depth_correlation(generated_images, control_images):
    """
    Calculate depth correlation (Pearson correlation).
    
    This is the ONLY mathematically sound depth metric when using grayscale as proxy.
    Correlation is scale-invariant, so independent normalization is fine.
    
    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])
    
    Returns:
        float: Pearson correlation [0, 1], higher is better
    """
    correlations = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
        
        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)
        
        # Convert to grayscale [0, 255] uint8
        gen_gray = np.array(gen_img.convert('L')).astype(np.float64)
        ctrl_gray = np.array(ctrl_img.convert('L')).astype(np.float64)
        
        # Normalize independently (OK for correlation, which is scale-invariant)
        gen_norm = (gen_gray - gen_gray.min()) / (gen_gray.max() - gen_gray.min() + 1e-8)
        ctrl_norm = (ctrl_gray - ctrl_gray.min()) / (ctrl_gray.max() - ctrl_gray.min() + 1e-8)
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(gen_norm.flatten(), ctrl_norm.flatten())[0, 1]
        
        # Clip to [0, 1] (negative correlation = bad)
        correlations.append(max(0.0, correlation))
    
    return float(np.mean(correlations)) if correlations else 0.0


def calculate_depth_mae(generated_images, control_images):
    """
    Calculate Mean Absolute Error in the SHARED [0, 255] space.
    
    This preserves scale information by NOT normalizing independently.
    Both images are converted to grayscale [0, 255], then MAE is computed directly.
    
    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])
    
    Returns:
        float: MAE in [0, 1] range, lower is better (0 = perfect match)
    """
    mae_scores = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
        
        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)
        
        # Convert to grayscale [0, 255] - SAME SCALE, NO INDEPENDENT NORMALIZATION
        gen_gray = np.array(gen_img.convert('L')).astype(np.float64)
        ctrl_gray = np.array(ctrl_img.convert('L')).astype(np.float64)
        
        # Calculate MAE in shared [0, 255] space
        mae = np.mean(np.abs(gen_gray - ctrl_gray))
        mae_scores.append(mae)
    
    return float(np.mean(mae_scores) / 255.0) if mae_scores else 0.0


def calculate_depth_rmse(generated_images, control_images):
    """
    Calculate RMSE in the SHARED [0, 255] space.
    
    This preserves scale information by NOT normalizing independently.
    
    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_images: List of Tensors (3, 512, 512, float32 [-1,1])
    
    Returns:
        float: RMSE in [0, 1] range, lower is better (0 = perfect match)
    """
    rmse_scores = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
        
        # Ensure both are PIL RGB [0, 255] uint8
        gen_img = _ensure_pil(gen_img)
        ctrl_img = _ensure_pil(ctrl_img)
        
        # Convert to grayscale [0, 255] - SAME SCALE
        gen_gray = np.array(gen_img.convert('L')).astype(np.float64)
        ctrl_gray = np.array(ctrl_img.convert('L')).astype(np.float64)
        
        # Calculate RMSE in shared [0, 255] space
        rmse = np.sqrt(np.mean((gen_gray - ctrl_gray) ** 2))
        rmse_scores.append(rmse)
    
    return float(np.mean(rmse_scores) / 255.0) if rmse_scores else 0.0


# ============================================================
# DISPATCHER FUNCTION
# ============================================================

def calculate_control_metrics(generated_images, control_dict, control_type, device='cuda'):
    """
    Dispatcher function - calls appropriate metrics based on control type.
    
    Args:
        generated_images: List of PIL Images (RGB, 512×512, uint8 [0,255])
        control_dict: Dict {control_type: List[Tensor(3,512,512, float32 [-1,1])]}
        control_type: str, one of ['gray', 'canny', 'depth']
        device: torch device (unused, for future LPIPS support)
    
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
            metrics['edge_f1'] = edge_f1_results['f1']
            metrics['edge_precision'] = edge_f1_results['precision']
            metrics['edge_recall'] = edge_f1_results['recall']
        
        elif control_type == 'depth':
            # Correlation (primary metric - scale-invariant, mathematically sound)
            metrics['depth_corr'] = calculate_depth_correlation(valid_gen, valid_ctrl)
            
            # MAE (secondary metric - preserves scale)
            metrics['depth_mae'] = calculate_depth_mae(valid_gen, valid_ctrl)
            
            # RMSE (secondary metric - preserves scale, penalizes large errors more)
            metrics['depth_rmse'] = calculate_depth_rmse(valid_gen, valid_ctrl)
    
    except Exception as e:
        print(f"[ERROR] Control metric calculation failed for {control_type}: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    return metrics