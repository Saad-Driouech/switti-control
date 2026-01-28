# utils/control_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
import cv2


def calculate_ssim(generated_images, control_images):
    """
    Calculate SSIM between generated and control images.
    For grayscale control.
    
    Args:
        generated_images: List of PIL Images (512×512)
        control_images: List of PIL Images (variable size)
    
    Returns:
        float: Mean SSIM score [0, 1], higher is better
    """
    ssim_scores = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
            
        # Convert to grayscale numpy arrays
        if isinstance(gen_img, Image.Image):
            gen_gray = np.array(gen_img.convert('L'))
        else:
            gen_gray = (rgb2gray(np.array(gen_img.convert('RGB'))) * 255).astype(np.uint8)
            
        if isinstance(ctrl_img, Image.Image):
            # Resize control to match generated size
            ctrl_resized = ctrl_img.resize(gen_img.size, Image.BILINEAR)
            ctrl_gray = np.array(ctrl_resized.convert('L'))
        else:
            ctrl_gray = (rgb2gray(np.array(ctrl_img.convert('RGB'))) * 255).astype(np.uint8)
        
        # Ensure same shape
        if gen_gray.shape != ctrl_gray.shape:
            # Fallback resize
            from skimage.transform import resize as sk_resize
            ctrl_gray = (sk_resize(ctrl_gray, gen_gray.shape, anti_aliasing=True) * 255).astype(np.uint8)
        
        # Calculate SSIM
        score = ssim(gen_gray, ctrl_gray, data_range=255)
        ssim_scores.append(score)
    
    if len(ssim_scores) == 0:
        return 0.0
    
    return float(np.mean(ssim_scores))


def calculate_edge_similarity(generated_images, control_images, threshold1=50, threshold2=150):
    """
    Calculate edge similarity for canny control.
    Measures how well generated edges match control edges.
    
    Args:
        generated_images: List of PIL Images (512×512)
        control_images: List of PIL Images (variable size, canny edge maps)
    
    Returns:
        float: Edge matching score [0, 1], higher is better
    """
    edge_scores = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
            
        # Convert generated to numpy
        gen_np = np.array(gen_img.convert('RGB'))
        
        # Resize control to match generated size
        ctrl_resized = ctrl_img.resize(gen_img.size, Image.BILINEAR)
        ctrl_np = np.array(ctrl_resized.convert('L'))
        
        # Extract edges from generated image
        gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
        gen_edges = cv2.Canny(gen_gray, threshold1, threshold2)
        
        # Control is already edge map (normalize to 0-255 if needed)
        ctrl_edges = ctrl_np
        if ctrl_edges.max() <= 1.0:
            ctrl_edges = (ctrl_edges * 255).astype(np.uint8)
        
        # Ensure same shape (should be guaranteed after resize)
        if gen_edges.shape != ctrl_edges.shape:
            from skimage.transform import resize as sk_resize
            ctrl_edges = (sk_resize(ctrl_edges, gen_edges.shape, anti_aliasing=True) * 255).astype(np.uint8)
        
        # Calculate IoU (Intersection over Union) of edges
        gen_binary = gen_edges > 0
        ctrl_binary = ctrl_edges > 0
        
        intersection = np.logical_and(gen_binary, ctrl_binary).sum()
        union = np.logical_or(gen_binary, ctrl_binary).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        edge_scores.append(iou)
    
    if len(edge_scores) == 0:
        return 0.0
    
    return float(np.mean(edge_scores))


def calculate_depth_correlation(generated_images, control_images):
    """
    Calculate depth correlation for depth control.
    Measures structural similarity in depth space.
    
    Note: This function requires MiDaS model to be loaded.
    For now, we'll implement a simpler version using grayscale correlation.
    
    Args:
        generated_images: List of PIL Images (512×512)
        control_images: List of PIL Images (variable size, depth maps)
    
    Returns:
        float: Depth correlation score [0, 1], higher is better
    """
    correlations = []
    
    for gen_img, ctrl_img in zip(generated_images, control_images):
        if ctrl_img is None:
            continue
        
        # Simple version: treat depth maps as grayscale and compute correlation
        # (Full version would use MiDaS to estimate depth from generated image)
        
        # Convert generated to grayscale (proxy for depth)
        gen_gray = np.array(gen_img.convert('L')).astype(np.float32) / 255.0
        
        # Resize control to match generated size
        ctrl_resized = ctrl_img.resize(gen_img.size, Image.BILINEAR)
        ctrl_gray = np.array(ctrl_resized.convert('L')).astype(np.float32) / 255.0
        
        # Ensure same shape
        if gen_gray.shape != ctrl_gray.shape:
            from skimage.transform import resize as sk_resize
            ctrl_gray = sk_resize(ctrl_gray, gen_gray.shape, anti_aliasing=True)
        
        # Normalize both to [0, 1]
        gen_norm = (gen_gray - gen_gray.min()) / (gen_gray.max() - gen_gray.min() + 1e-8)
        ctrl_norm = (ctrl_gray - ctrl_gray.min()) / (ctrl_gray.max() - ctrl_gray.min() + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(gen_norm.flatten(), ctrl_norm.flatten())[0, 1]
        correlations.append(max(0, correlation))  # Clip negative correlations to 0
    
    if len(correlations) == 0:
        return 0.0
    
    return float(np.mean(correlations))


def calculate_control_metrics(generated_images, control_dict, control_type):
    """
    Dispatcher function - calls appropriate metric based on control type.
    
    Args:
        generated_images: List of PIL Images (512×512)
        control_dict: Dict {control_type: [PIL Images]}
        control_type: str, one of ['gray', 'canny', 'depth']
    
    Returns:
        dict: {metric_name: score}
    """
    if control_type not in control_dict:
        return {}
    
    control_images = control_dict[control_type]
    
    # Filter out None values
    valid_pairs = [(g, c) for g, c in zip(generated_images, control_images) if c is not None]
    
    if len(valid_pairs) == 0:
        return {}
    
    valid_gen, valid_ctrl = zip(*valid_pairs)
    
    metrics = {}
    
    try:
        if control_type == 'gray':
            metrics['ssim'] = calculate_ssim(list(valid_gen), list(valid_ctrl))
        
        if control_type == 'canny':
            metrics['edge_similarity'] = calculate_edge_similarity(list(valid_gen), list(valid_ctrl))
        
        if control_type == 'depth':
            metrics['depth_correlation'] = calculate_depth_correlation(list(valid_gen), list(valid_ctrl))
    except Exception as e:
        print(f"[WARNING] Control metric calculation failed for {control_type}: {e}")
        # Return empty dict on error
        return {}
    
    return metrics