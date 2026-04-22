"""
train_maskrcnn.py
Entrena Mask R-CNN (ResNet50-FPN) para segmentación de instancia.
Clases: face, left_eye, right_eye, phone  (+ background)

Uso dentro del contenedor:
    cd /app/src
    python train_maskrcnn.py
"""

import os, sys, time
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor

sys.path.insert(0, os.path.dirname(__file__))
from dataset_segmentation import SegmentationDataset

# ── Hiperparámetros ───────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
NUM_EPOCHS   = 30
BATCH_SIZE   = 2
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP      = 10
LR_GAMMA     = 0.1
NUM_WORKERS  = 2
# ─────────────────────────────────────────────────────────────────────────────


def build_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_box  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)

    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Mask R-CNN — Entrenamiento")
    print(f"  Dispositivo : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*55}\n")

    train_ds = SegmentationDataset(DATASET_ROOT, split="train")
    valid_ds = SegmentationDataset(DATASET_ROOT, split="valid")
    num_classes = train_ds.num_classes

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = build_model(num_classes).to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_GAMMA)

    best_loss = float("inf")
    history   = {"train": [], "valid": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader, 1):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                print(f"  Ep {epoch:>2}/{NUM_EPOCHS} | "
                      f"batch {step:>3}/{len(train_loader)} | "
                      f"loss {loss.item():.4f}")

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        history["train"].append(avg_train)

        # ── Validation loss ───────────────────────────────────────────────────
        model.train()   # Mask R-CNN necesita modo train para calcular loss
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()
        avg_val = val_loss / len(valid_loader)
        history["valid"].append(avg_val)

        elapsed = time.time() - t0
        print(f"\n  ► Epoch {epoch}/{NUM_EPOCHS} | "
              f"train={avg_train:.4f} | val={avg_val:.4f} | {elapsed:.1f}s\n")

        if avg_val < best_loss:
            best_loss = avg_val
            path = os.path.join(MODELS_DIR, "best_maskrcnn.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 best_loss,
                "num_classes":          num_classes,
                "label_to_name":        train_ds.label_to_name,
            }, path)
            print(f"  ✓ Mejor modelo guardado (val={best_loss:.4f}) → {path}\n")

        if epoch % 5 == 0:
            ckpt = os.path.join(MODELS_DIR, f"maskrcnn_ep{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Checkpoint → {ckpt}\n")

    # Guardar curvas de loss
    _save_loss_plot(history)
    print(f"{'='*55}")
    print(f"  Mask R-CNN completo. Mejor val loss: {best_loss:.4f}")
    print(f"{'='*55}\n")


def _save_loss_plot(history):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(history["train"], label="Train loss")
        plt.plot(history["valid"], label="Val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Mask R-CNN — Curva de entrenamiento")
        plt.legend(); plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                           "maskrcnn_loss.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Curva guardada → {out}")
    except Exception as e:
        print(f"  [WARN] No se pudo guardar la curva: {e}")


if __name__ == "__main__":
    train()