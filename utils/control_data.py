"""
Dataset and data utilities for spatially-controlled Switti training.

Supports pre-computed control maps (recommended for training speed) and
online extraction for canny edges.

Modality IDs:
    canny:  0
    depth:  1
    seg:    2
    normals: 3
    hed:    4
    gray:   5
    openpose: 6
    null:   num_modalities  (used for CFG dropout)
"""
import csv
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MODALITY_IDS = {"canny": 0, "depth": 1, "seg": 2, "normals": 3, "hed": 4, "gray": 5, "openpose": 6,
                "seg_cocostuff": 2}  # same embedding slot as seg


def _build_transform(final_reso: int, mid_reso_factor: float = 1.125):
    mid_reso = round(mid_reso_factor * final_reso)
    return transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        # normalise [0,1] → [-1,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _extract_canny(pil_img: Image.Image) -> Image.Image:
    """Extract Canny edges online using adaptive thresholds."""
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise ImportError("cv2 is required for online canny extraction: pip install opencv-python") from e

    gray = np.array(pil_img.convert("L"))
    median = float(np.median(gray))
    low_thresh = int(max(0, 0.1 * median))
    high_thresh = int(min(255, 0.2 * median))  # adjusted; see precompute script for better heuristic
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


class SpatialControlDataset(Dataset):
    """
    Dataset that pairs images with spatial control maps and captions.

    Args:
        root_dir: directory with images and a <subset_name>.csv caption file
        subset_name: sub-directory name (e.g. "train2014")
        modalities: list of modality names to randomly sample from per item
        ctrl_maps_dir: if set, loads pre-computed maps from
                       <ctrl_maps_dir>/<modality>/<subset_name>/<image_id>.<ext>
                       Otherwise extracts canny maps online.
        final_reso: output resolution
        mid_reso_factor: resize factor before centre crop
        max_cnt: cap on dataset size (None = all)
    """

    EXTENSIONS = {"jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"}

    def __init__(
        self,
        root_dir: str,
        subset_name: str = "train2014",
        modalities: Optional[list] = None,
        ctrl_maps_dir: Optional[str] = None,
        final_reso: int = 256,
        mid_reso_factor: float = 1.125,
        max_cnt: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.subset_name = subset_name
        self.modalities = modalities or ["canny"]
        self.ctrl_maps_dir = ctrl_maps_dir

        sample_dir = os.path.join(root_dir, subset_name)
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.rsplit(".", 1)[-1].lower() in self.EXTENSIONS
            ],
            key=lambda p: os.path.basename(p).rsplit(".", 1)[0],
        )
        if max_cnt is not None:
            self.samples = self.samples[:max_cnt]

        # Load captions
        self.captions = {}
        csv_path = os.path.join(root_dir, f"{subset_name}.csv")
        with open(csv_path, newline="\n") as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]

        self.img_transform = _build_transform(final_reso, mid_reso_factor)
        # Control map transform: resize + crop only (no colour normalisation)
        mid_reso = round(mid_reso_factor * final_reso)
        self.ctrl_transform = transforms.Compose([
            transforms.Resize(mid_reso, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_ctrl_map(self, img_path: str, modality: str) -> Image.Image:
        """Load or compute the control map for a given image and modality."""
        img_id = os.path.basename(img_path).rsplit(".", 1)[0]

        if self.ctrl_maps_dir is not None:
            ctrl_path = os.path.join(
                self.ctrl_maps_dir, modality, self.subset_name, f"{img_id}.png"
            )
            if os.path.exists(ctrl_path):
                return Image.open(ctrl_path).convert("RGB")

        # Fallback: online extraction (canny only supported out of the box)
        if modality == "canny":
            pil_img = Image.open(img_path).convert("RGB")
            return _extract_canny(pil_img)

        raise FileNotFoundError(
            f"Control map not found and online extraction not supported for modality '{modality}'. "
            f"Run scripts/precompute_control_maps.py first."
        )

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Sample a random modality for this item
        modality = self.modalities[torch.randint(len(self.modalities), (1,)).item()]
        ctrl_pil = self._load_ctrl_map(img_path, modality)

        return {
            "image": self.img_transform(image),
            "ctrl_image": self.ctrl_transform(ctrl_pil),
            "caption": self.captions.get(os.path.basename(img_path), ""),
            "modality_id": MODALITY_IDS[modality],
        }


def control_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch])
    ctrl_images = torch.stack([x["ctrl_image"] for x in batch])
    captions = [x["caption"] for x in batch]
    modality_ids = torch.tensor([x["modality_id"] for x in batch], dtype=torch.long)
    return images, ctrl_images, captions, modality_ids


def build_control_dataset(
    data_path: str,
    final_reso: int,
    modalities: Optional[list] = None,
    ctrl_maps_dir: Optional[str] = None,
    mid_reso_factor: float = 1.125,
    max_cnt: Optional[int] = None,
    subset_name: str = "train2014",
) -> SpatialControlDataset:
    ds = SpatialControlDataset(
        root_dir=data_path,
        subset_name=subset_name,
        modalities=modalities or ["canny"],
        ctrl_maps_dir=ctrl_maps_dir,
        final_reso=final_reso,
        mid_reso_factor=mid_reso_factor,
        max_cnt=max_cnt,
    )
    print(f"[ControlDataset] {len(ds)=}, modalities={ds.modalities}")
    return ds
