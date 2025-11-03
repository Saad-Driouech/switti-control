import csv
import os

import PIL.Image as PImage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import cv2
import numpy as np


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def normalize_255_into_pm1(x):
    return x.div_(127.5).sub(-1)


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

        if self.transform:
            sample = self.transform(sample)

        return sample, self.captions[os.path.basename(sample_path)]
    
class ControlDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset (e.g. COCODataset) to add control images.
    Returns (image, control_image, caption)
    """
    def __init__(self, base_dataset, control_type: str = "edges"):
        self.base = base_dataset
        self.control_type = control_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, caption = self.base[idx]
        control = self.make_control_image(image)
        return image, control, caption

    def make_control_image(self, image):
        # convert tensor [C,H,W] → numpy [H,W,C] in 0–255
        if isinstance(image, torch.Tensor):
            img_np = (image.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
        else:
            img_np = np.array(image)

        if self.control_type == "edges":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            control = np.stack([edges] * 3, axis=-1)
        elif self.control_type == "gray":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            control = np.stack([gray] * 3, axis=-1)
        elif self.control_type == "same":
            control = img_np
        else:
            raise ValueError(f"Unknown control_type: {self.control_type}")

        control_tensor = torch.from_numpy(control).permute(2, 0, 1).float() / 255.0
        control_tensor = normalize_01_into_pm1(control_tensor)
        return control_tensor


def coco_collate_fn(batch):
    if len(batch[0]) == 2:
        # (image, caption)
        images = torch.stack([x[0] for x in batch])
        captions = [x[1] for x in batch]
        return images, None, captions  # control_image=None
    else:
        # (image, control, caption)
        images = torch.stack([x[0] for x in batch])
        controls = torch.stack([x[1] for x in batch])
        captions = [x[2] for x in batch]
        return images, controls, captions


def build_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
    use_control=False,
    control_type="edges",
):
    # build augmentations
    # first resize to mid_reso, then crop to final_reso
    mid_reso = round(mid_reso * final_reso)
    train_aug = [
        transforms.Resize(
            mid_reso,
            interpolation=InterpolationMode.LANCZOS,
        ),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug = transforms.Compose(train_aug)

    # build dataset
    train_set = COCODataset(
        data_path,
        subset_name="train2014",
        transform=train_aug,
        max_cnt=None,
    )

    if use_control:
        train_set = ControlDataset(train_set, control_type=control_type)
        print(f"[Dataset] Using control dataset (type={control_type})")

    print(f"[Dataset] {len(train_set)=}")
    print_aug(train_aug, "[train]")

    return train_set


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
