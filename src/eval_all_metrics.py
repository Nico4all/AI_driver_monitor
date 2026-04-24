"""
eval_all_metrics.py
Calcula todas las métricas relevantes para ambos modelos:

  EfficientNet-B0  → Accuracy, Precision, Recall, F1 por clase
  Mask R-CNN       → Accuracy (cabeza clf), mAP@0.50, mAP@0.50:0.95, Mean IoU

Nota sobre PCK (Pose Estimation):
  PCK no aplica a este proyecto porque no se implementó un modelo de
  estimación de postura. El proyecto usa segmentación de instancia
  (Mask R-CNN), cuya métrica equivalente es mAP y Mean IoU.

Uso dentro del contenedor:
    cd /app/src
    python eval_all_metrics.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from pycocotools.coco    import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_util
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from dataset_segmentation import SegmentationDataset
from dataset_classifier   import DriverStateDataset, STATES, VAL_TRANSFORMS

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
SCORE_THRESH_MAP = 0.05   # bajo → curva PR completa para mAP
SCORE_THRESH_ACC = 0.50   # normal → accuracy de la cabeza de clasificación
MASK_THRESH      = 0.50

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — EfficientNet-B0: Accuracy + Reporte por clase
# ─────────────────────────────────────────────────────────────────────────────
def eval_efficientnet(device):
    sep("EfficientNet-B0 — Clasificación de Estado")

    path = os.path.join(MODELS_DIR, "best_classifier.pth")
    ckpt = torch.load(path, map_location=device)

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_ds = DriverStateDataset(DATASET_ROOT, splits=["test"],
                                 transform=VAL_TRANSFORMS)
    loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)

    print(f"\n  Accuracy (Test Set) : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Reporte por clase:")
    print(classification_report(all_labels, all_preds,
                                 target_names=STATES, digits=4))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=STATES, yticklabels=STATES)
    plt.title(f"EfficientNet-B0 — Confusion Matrix (Acc={acc:.4f})")
    plt.ylabel("Real"); plt.xlabel("Predicho"); plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "efficientnet_confusion_matrix_final.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"  Matriz guardada → {out}")

    return acc


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Mask R-CNN: Accuracy de la cabeza de clasificación
# ─────────────────────────────────────────────────────────────────────────────
def eval_maskrcnn_accuracy(device, model, label_to_name, test_ds):
    sep("Mask R-CNN — Accuracy de Cabeza de Clasificación")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    tp_total, fp_total, fn_total = 0, 0, 0
    per_class = {name: [0, 0, 0] for name in label_to_name.values()
                 if name != "driver-segmentation"}

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                pred = [label_to_name[int(l)]
                        for l, s in zip(out["labels"], out["scores"])
                        if s >= SCORE_THRESH_ACC and
                           label_to_name.get(int(l)) in per_class]
                true = [label_to_name[int(l)]
                        for l in tgt["labels"]
                        if label_to_name.get(int(l)) in per_class]

                for name in per_class:
                    tp = min(pred.count(name), true.count(name))
                    fp = max(0, pred.count(name) - true.count(name))
                    fn = max(0, true.count(name) - pred.count(name))
                    per_class[name][0] += tp
                    per_class[name][1] += fp
                    per_class[name][2] += fn
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn

    print(f"\n  {'Clase':<20} {'Precisión':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*52}")
    macro_f1 = []
    for name, (tp, fp, fn) in per_class.items():
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2*p*r / (p+r+1e-6)
        macro_f1.append(f1)
        print(f"  {name:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    print(f"  {'─'*52}")

    overall_acc = tp_total / (tp_total + fp_total + fn_total + 1e-6)
    macro       = sum(macro_f1) / len(macro_f1)
    print(f"  Accuracy global     : {overall_acc:.4f}")
    print(f"  Macro F1            : {macro:.4f}")

    return overall_acc


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Mask R-CNN: mAP@0.50 y mAP@0.50:0.95
# ─────────────────────────────────────────────────────────────────────────────
def eval_maskrcnn_map(device, model, label_to_name, test_ds):
    sep("Mask R-CNN — mAP@0.50 y mAP@0.50:0.95")

    loader   = DataLoader(test_ds, batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
    ann_path = os.path.join(DATASET_ROOT, "test", "_annotations.coco.json")
    coco_gt  = COCO(ann_path)

    name_to_catid = {cat["name"]: cat["id"]
                     for cat in coco_gt.dataset["categories"]}

    dt_bbox, dt_segm = [], []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                img_id   = int(tgt["image_id"][0])
                img_info = coco_gt.imgs[img_id]
                h, w     = img_info["height"], img_info["width"]

                for score, lbl, box, mask in zip(
                        out["scores"], out["labels"],
                        out["boxes"],  out["masks"]):

                    if float(score) < SCORE_THRESH_MAP:
                        continue

                    name   = label_to_name.get(int(lbl))
                    cat_id = name_to_catid.get(name)
                    if cat_id is None:
                        continue

                    x1,y1,x2,y2 = box.cpu().numpy()
                    dt_bbox.append({
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":  [float(x1), float(y1),
                                  float(x2-x1), float(y2-y1)],
                        "score": float(score),
                    })

                    mb  = (mask[0].cpu().numpy() > MASK_THRESH).astype(np.uint8)
                    mr  = cv2.resize(mb, (w, h), interpolation=cv2.INTER_NEAREST)
                    rle = coco_mask_util.encode(np.asfortranarray(mr))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    dt_segm.append({
                        "image_id":     img_id,
                        "category_id":  cat_id,
                        "segmentation": rle,
                        "score":        float(score),
                    })

    # BBox mAP
    print("\n  ── BBox ──────────────────────────────────────────────")
    coco_pred_bbox = coco_gt.loadRes(dt_bbox)
    ev_bbox        = COCOeval(coco_gt, coco_pred_bbox, iouType="bbox")
    ev_bbox.evaluate(); ev_bbox.accumulate(); ev_bbox.summarize()

    # Segmentation mAP
    print("\n  ── Segmentación ──────────────────────────────────────")
    coco_pred_segm = coco_gt.loadRes(dt_segm)
    ev_segm        = COCOeval(coco_gt, coco_pred_segm, iouType="segm")
    ev_segm.evaluate(); ev_segm.accumulate(); ev_segm.summarize()

    print(f"\n  {'─'*50}")
    print(f"  BBox  mAP@0.50      : {ev_bbox.stats[1]:.4f}")
    print(f"  BBox  mAP@0.50:0.95 : {ev_bbox.stats[0]:.4f}")
    print(f"  Segm  mAP@0.50      : {ev_segm.stats[1]:.4f}")
    print(f"  Segm  mAP@0.50:0.95 : {ev_segm.stats[0]:.4f}")
    print(f"  {'─'*50}")

    return ev_bbox.stats[1], ev_segm.stats[1]


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Mask R-CNN: Mean IoU por clase
# ─────────────────────────────────────────────────────────────────────────────
def eval_maskrcnn_miou(device, model, label_to_name, test_ds):
    sep("Mask R-CNN — Mean IoU por clase")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    iou_per_class = {name: [] for name in label_to_name.values()
                     if name != "driver-segmentation"}

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                # Para cada GT mask, buscar la predicción de mayor IoU
                gt_labels = tgt["labels"].cpu().numpy()
                gt_masks  = tgt["masks"].cpu().numpy()   # [N, H, W]

                pred_scores = out["scores"].cpu().numpy()
                pred_labels = out["labels"].cpu().numpy()
                pred_masks  = out["masks"].cpu().numpy()  # [N, 1, H, W]

                used_preds = set()

                for gt_lbl, gt_mask in zip(gt_labels, gt_masks):
                    name = label_to_name.get(int(gt_lbl))
                    if name not in iou_per_class:
                        continue

                    best_iou  = 0.0
                    best_pred = -1

                    for pi, (ps, pl, pm) in enumerate(
                            zip(pred_scores, pred_labels, pred_masks)):
                        if pi in used_preds:
                            continue
                        if int(pl) != int(gt_lbl):
                            continue
                        if float(ps) < SCORE_THRESH_ACC:
                            continue

                        pred_bin = (pm[0] > MASK_THRESH).astype(np.uint8)
                        gt_bin   = gt_mask.astype(np.uint8)

                        # Redimensionar pred a tamaño de gt si es necesario
                        if pred_bin.shape != gt_bin.shape:
                            pred_bin = cv2.resize(
                                pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

                        intersection = np.logical_and(pred_bin, gt_bin).sum()
                        union        = np.logical_or(pred_bin, gt_bin).sum()
                        iou          = intersection / (union + 1e-6)

                        if iou > best_iou:
                            best_iou  = iou
                            best_pred = pi

                    iou_per_class[name].append(best_iou)
                    if best_pred >= 0:
                        used_preds.add(best_pred)

    print(f"\n  {'Clase':<20} {'IoU medio':>12} {'# muestras':>12}")
    print(f"  {'─'*46}")

    all_ious = []
    for name, ious in iou_per_class.items():
        if len(ious) == 0:
            print(f"  {name:<20} {'N/A':>12} {'0':>12}")
            continue
        mean_iou = np.mean(ious)
        all_ious.extend(ious)
        print(f"  {name:<20} {mean_iou:>12.4f} {len(ious):>12}")

    miou = np.mean(all_ious) if all_ious else 0.0
    print(f"  {'─'*46}")
    print(f"  {'Mean IoU (mIoU)':<20} {miou:>12.4f}")

    # Gráfica de distribución de IoU por clase
    fig, ax = plt.subplots(figsize=(8, 4))
    data  = [iou_per_class[n] for n in iou_per_class if iou_per_class[n]]
    names = [n for n in iou_per_class if iou_per_class[n]]
    ax.boxplot(data, labels=names)
    ax.set_title("Distribución de IoU por clase — Mask R-CNN (Test Set)")
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1.05)
    ax.axhline(miou, color="red", linestyle="--", label=f"mIoU={miou:.4f}")
    ax.legend(); plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "maskrcnn_iou_distribution.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\n  Gráfica IoU → {out}")

    return miou


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Cargar Mask R-CNN una sola vez ────────────────────────────────────────
    path = os.path.join(MODELS_DIR, "best_maskrcnn.pth")
    ckpt = torch.load(path, map_location=device)
    maskrcnn = maskrcnn_resnet50_fpn(weights=None)
    in_box   = maskrcnn.roi_heads.box_predictor.cls_score.in_features
    maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_box, ckpt["num_classes"])
    in_mask  = maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
    maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, ckpt["num_classes"])
    maskrcnn.load_state_dict(ckpt["model_state_dict"])
    maskrcnn.to(device)
    label_to_name = ckpt["label_to_name"]

    test_ds = SegmentationDataset(DATASET_ROOT, split="test")

    # ── Ejecutar todos los bloques ────────────────────────────────────────────
    clf_acc              = eval_efficientnet(device)
    maskrcnn_acc         = eval_maskrcnn_accuracy(device, maskrcnn, label_to_name, test_ds)
    map50_bbox, map50_sg = eval_maskrcnn_map(device, maskrcnn, label_to_name, test_ds)
    miou                 = eval_maskrcnn_miou(device, maskrcnn, label_to_name, test_ds)

    # ── Tabla resumen final ───────────────────────────────────────────────────
    sep("RESUMEN FINAL — TODAS LAS MÉTRICAS")
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  MODELO             MÉTRICA              VALOR       │
  ├─────────────────────────────────────────────────────┤
  │  EfficientNet-B0    Accuracy (clf)       {clf_acc:.4f}     │
  ├─────────────────────────────────────────────────────┤
  │  Mask R-CNN         Accuracy (cabeza)    {maskrcnn_acc:.4f}     │
  │  Mask R-CNN         mAP@0.50 (bbox)      {map50_bbox:.4f}     │
  │  Mask R-CNN         mAP@0.50 (segm)      {map50_sg:.4f}     │
  │  Mask R-CNN         Mean IoU             {miou:.4f}     │
  ├─────────────────────────────────────────────────────┤
  │  PCK (Pose)         N/A — sin modelo de pose        │
  └─────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()