"""
Phase 2: Data Pipeline
Dataset loader + augmentations for reflective/transparent objects.

Recommended public datasets:
  - Trans10K   : https://github.com/xieenze/Trans10K  (transparent objects)
  - GSD        : Glass Surface Detection dataset
  - Omnimatting: semi-transparent matting
  - COCO (subset): bottles, cups, wine glasses
  - Synthetic  : generated via src/synthetic_gen.py
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


CLASS_NAMES = [
    "glass_bottle", "plastic_container", "reflective_surface",
    "semi_transparent_object", "glass_cup", "plastic_bag", "mirror", "window"
]


def get_augmentations(split: str, image_size: int = 640) -> A.Compose:
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3)

    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            # Simulate reflections / glare
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                             num_flare_circles_lower=1, num_flare_circles_upper=3,
                             src_radius=150, p=0.2),
            # Simulate focus blur / camera motion
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=1.0),
            ], p=0.3),
            # Simulate transparency / frosted glass
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox_params)


class ReflectiveObjectDataset(Dataset):
    """
    Expects YOLO-format directory layout:
        data/processed/
            images/train/*.jpg
            images/val/*.jpg
            images/test/*.jpg
            labels/train/*.txt
            labels/val/*.txt
            labels/test/*.txt
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 640):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transforms = get_augmentations(split, image_size)

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) +
            list(self.image_dir.glob("*.jpeg")) +
            list(self.image_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            print(f"[WARNING] No images found in {self.image_dir}. "
                  "Run synthetic_gen.py to generate sample data.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.read().strip().splitlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(cls_id)

        augmented = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        image_tensor = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        targets = torch.zeros((len(aug_bboxes), 6))
        for i, (box, cls) in enumerate(zip(aug_bboxes, aug_labels)):
            targets[i] = torch.tensor([0, cls, *box])

        return image_tensor, targets, str(img_path)


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images, 0)
    for i, t in enumerate(targets):
        t[:, 0] = i
    targets = torch.cat(targets, 0)
    return images, targets, paths


def get_dataloader(root: str, split: str, image_size: int = 640,
                   batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    dataset = ReflectiveObjectDataset(root, split, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
