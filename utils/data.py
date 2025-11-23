import csv
import os

import PIL.Image as PImage
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from typing import Optional, Union, List, Dict


class COCODataset(Dataset):
    def __init__(self, root_dir, subset_name="subset", transform=None, max_cnt=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            "jpg",
            "jpeg",
            "png",
            "ppm",
            "bmp",
            "pgm",
            "tif",
            "tiff",
            "webp",
        )
        sample_dir = os.path.join(root_dir, subset_name)

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.split('.')[-1] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        # restrict num samples
        self.samples = self.samples if max_cnt is None else self.samples[:max_cnt]  

        # Collect captions
        self.captions = {}
        with open(
            os.path.join(root_dir, f"{subset_name}.csv"), newline="\n"
        ) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")
        orig_size = sample.size

        if self.transform:
            sample = self.transform(sample)

        return sample, self.captions[os.path.basename(sample_path)], orig_size


class JointTransform:
    """
    Synchronizes augmentation parameters across the image and a DICT of controls.
    """
    def __init__(self, final_reso: int, mid_reso=1.125, hflip_prob=0.5):
        self.final_reso = final_reso
        self.mid_reso = round(mid_reso * final_reso)
        self.hflip_prob = hflip_prob

    def __call__(self, img: Image.Image, controls: Dict[str, Image.Image]):
        # 1. Determine Random Parameters ONCE
        do_flip = random.random() < self.hflip_prob
        
        # 2. Transform Main Image
        img = self._transform_single(img, do_flip)
        
        # 3. Transform All Controls
        processed_controls = {}
        for k, v in controls.items():
            if v is None:
                # Handle missing files gracefully (zero tensor)
                processed_controls[k] = torch.zeros_like(img)
            else:
                processed_controls[k] = self._transform_single(v, do_flip)
                
        return img, processed_controls

    def _transform_single(self, pic, do_flip):
        # Resize
        pic = TF.resize(pic, self.mid_reso, interpolation=InterpolationMode.LANCZOS)
        
        # Center Crop
        pic = TF.center_crop(pic, (self.final_reso, self.final_reso))
        
        # Horizontal Flip
        if do_flip:
            pic = TF.hflip(pic)
            
        # To Tensor
        pic = TF.to_tensor(pic) # [0, 1]
        
        # Normalize
        # Maps [0, 1] -> [-1, 1]
        pic = pic.add(pic).add_(-1)
        
        return pic


class ControlDataset(Dataset):
    def __init__(
        self,
        base_dataset: COCODataset,
        control_types: Union[str, List[str]],
        joint_transform: JointTransform,
        control_root: Optional[str] = None,
        strict: bool = True,
    ):
        self.base = base_dataset
        self.joint_transform = joint_transform
        self.strict = strict
        
        if isinstance(control_types, str):
            self.control_types = [control_types]
        else:
            self.control_types = control_types

        # Pre-calculate roots
        self.control_roots = {}
        for ctype in self.control_types:
            if control_root:
                self.control_roots[ctype] = os.path.join(control_root, ctype)
            else:
                self.control_roots[ctype] = os.path.join(self.base.root_dir, "control", ctype)

    def __len__(self):
        return len(self.base)

    def _load_control(self, ctype, filename):
        root = self.control_roots[ctype]
        name_no_ext = os.path.splitext(filename)[0]
        
        # Try extensions
        for ext in ["png", "jpg", "jpeg", "bmp", "tif"]:
            path = os.path.join(root, f"{name_no_ext}.{ext}")
            if os.path.exists(path):
                img = Image.open(path)
                # Ensure correct mode
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
        return None

    def __getitem__(self, idx):
        # 1. Load Base (Raw PIL)
        image, caption, orig_size = self.base[idx]
        filename = os.path.basename(self.base.samples[idx])

        # 2. Load All Controls (Raw PIL)
        raw_controls = {}
        for ctype in self.control_types:
            ctrl_img = self._load_control(ctype, filename)
            
            if ctrl_img is None:
                if self.strict:
                    raise FileNotFoundError(f"Control {ctype} missing for {filename}")
                raw_controls[ctype] = None # Handled in JointTransform
            else:
                raw_controls[ctype] = ctrl_img

        # 3. Apply Joint Transform ONCE
        # This fixes the "Iterative Destruction" bug
        image_tensor, control_tensors = self.joint_transform(image, raw_controls)

        return image_tensor, caption, control_tensors, orig_size


def coco_collate_fn(batch):
    # Standard format: (image, caption, orig_size)
    if len(batch[0]) == 3:
        return torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch]

    # Control format: (image, caption, controls_dict, orig_size)
    if len(batch[0]) == 4:
        images = torch.stack([x[0] for x in batch])
        captions = [x[1] for x in batch]
        
        # Stack controls
        control_keys = batch[0][2].keys()
        controls = {
            k: torch.stack([x[2][k] for x in batch]) 
            for k in control_keys
        }
        
        orig_sizes = [x[3] for x in batch]

        return images, captions, controls, orig_sizes


def build_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
    control_types: Optional[Union[str, List[str]]] = None,
    strict: bool = True,
):
    # 1. If no controls, we need a standard transform for the base dataset
    if not control_types:
        # Define standard transform (No Joint logic needed)
        mid_reso = round(mid_reso * final_reso)
        train_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            lambda x: x.add(x).add_(-1), # Normalize
        ]
        if hflip:
            train_aug.insert(0, transforms.RandomHorizontalFlip())
            
        # Return base dataset with internal transform
        dataset = COCODataset(data_path, subset_name="train2014", transform=transforms.Compose(train_aug))

    # 2. If controls exist, use JointTransform
    else:
        joint_trans = JointTransform(
            final_reso=final_reso,
            mid_reso=mid_reso,
            hflip_prob=0.5 if hflip else 0.0
        )

        base = COCODataset(data_path, subset_name="train2014", transform=None) # Load raw PIL
        
        dataset = ControlDataset(
            base, 
            control_types=control_types, 
            joint_transform=joint_trans,
            strict=strict
        )
    
    print(f"[Dataset] Loaded {len(dataset)} samples. HFlip={hflip}")
    return dataset


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def print_aug(transform, label):
    print(f"Transform {label} = ")
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print("---------------------------\n")
