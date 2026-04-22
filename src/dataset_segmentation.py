"""
dataset_segmentation.py
Dataset COCO para Mask R-CNN.

Clases:
  0 = background
  1 = face
  2 = left_eye
  3 = right_eye
  4 = phone
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from pycocotools import mask as coco_mask_util


class SegmentationDataset(Dataset):
    def __init__(self, dataset_root, split="train", transforms=None):
        self.root  = dataset_root
        self.split = split
        self.transforms = transforms

        ann_path = os.path.join(dataset_root, split, "_annotations.coco.json")
        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images      = {img["id"]: img for img in coco["images"]}
        self.image_ids   = [img["id"] for img in coco["images"]]

        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Mapeo category_id → label 1-indexed
        self.cat_to_label = {
            cat["id"]: idx + 1
            for idx, cat in enumerate(coco["categories"])
        }
        self.label_to_name = {
            idx + 1: cat["name"]
            for idx, cat in enumerate(coco["categories"])
        }
        self.num_classes = len(coco["categories"]) + 1

        print(f"[Seg] split='{split}' | imgs={len(self.image_ids)} | "
              f"clases={self.num_classes-1} | {self.label_to_name}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        # Imagen directamente en dataset/<split>/
        img_path = os.path.join(self.root, self.split, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.anns_by_image.get(img_id, [])
        boxes, labels, masks = [], [], []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            if x2 <= x1 or y2 <= y1:
                continue

            seg  = ann["segmentation"]
            rles = coco_mask_util.frPyObjects(seg, h, w)
            m    = coco_mask_util.decode(rles)
            if m.ndim == 3:
                m = m.any(axis=2)

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_to_label[ann["category_id"]])
            masks.append(m.astype(np.uint8))

        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
            masks_t  = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes_t  = torch.as_tensor(boxes,               dtype=torch.float32)
            labels_t = torch.as_tensor(labels,              dtype=torch.int64)
            masks_t  = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        area = (
            (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        )

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "image_id": torch.tensor([img_id]),
            "area":     area,
            "iscrowd":  torch.zeros((len(labels_t),), dtype=torch.int64),
        }

        img_tensor = TF.to_tensor(img)
        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target