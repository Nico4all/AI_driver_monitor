"""
dataset_classifier.py
Dataset de clasificación para EfficientNet-B0.

El estado del conductor se infiere del PREFIJO del nombre de archivo:
  awake_        → 0
  distracted_   → 1
  drowsy_       → 2
  eyes_closed_  → 3
  phone_        → 4
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

STATES = ["awake", "distracted", "drowsy", "eyes_closed", "phone"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}

# Transforms para entrenamiento con augmentation leve
TRAIN_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def get_state_from_filename(path: str):
    name = os.path.basename(path).lower()
    # Ordenar de mayor a menor longitud para evitar match parcial
    for state in sorted(STATES, key=len, reverse=True):
        if name.startswith(state + "_"):
            return state
    return None


class DriverStateDataset(Dataset):
    """
    Carga todas las imágenes de los splits indicados y las etiqueta
    por el prefijo del nombre de archivo.
    """
    def __init__(self, dataset_root, splits, transform=None):
        self.transform = transform
        self.samples   = []  # lista de (path, label_idx)
        self.skipped   = 0

        for split in splits:
            folder = os.path.join(dataset_root, split)
            paths  = (
                glob.glob(os.path.join(folder, "*.jpg")) +
                glob.glob(os.path.join(folder, "*.jpeg")) +
                glob.glob(os.path.join(folder, "*.png"))
            )
            for p in paths:
                state = get_state_from_filename(p)
                if state is None:
                    self.skipped += 1
                    continue
                self.samples.append((p, STATE_TO_IDX[state]))

        counts = {s: 0 for s in STATES}
        for _, lbl in self.samples:
            counts[STATES[lbl]] += 1

        print(f"[Clf] splits={splits} | total={len(self.samples)} | "
              f"saltadas={self.skipped}")
        print(f"[Clf] distribución: {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_all_samples(dataset_root, splits=("train", "valid")):
    """Devuelve listas de (path, label) para uso en k-fold."""
    samples = []
    for split in splits:
        folder = os.path.join(dataset_root, split)
        paths  = (
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.jpeg")) +
            glob.glob(os.path.join(folder, "*.png"))
        )
        for p in paths:
            state = get_state_from_filename(p)
            if state is not None:
                samples.append((p, STATE_TO_IDX[state]))
    return samples


class SampleListDataset(Dataset):
    """Dataset que recibe directamente una lista de (path, label)."""
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label