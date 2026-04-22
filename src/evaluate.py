"""
evaluate.py
Evalúa ambos modelos sobre el test set e imprime métricas.

Uso:
    python evaluate.py --model maskrcnn    # solo Mask R-CNN
    python evaluate.py --model classifier  # solo EfficientNet
    python evaluate.py                     # ambos (default)
"""

import os, sys, argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor
from torchvision.models import efficientnet_b0
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset_segmentation import SegmentationDataset
from dataset_classifier   import DriverStateDataset, STATES, VAL_TRANSFORMS

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
SCORE_THRESH = 0.5


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Mask R-CNN ────────────────────────────────────────────────────────────────
def eval_maskrcnn(device):
    path = os.path.join(MODELS_DIR, "best_maskrcnn.pth")
    ckpt = torch.load(path, map_location=device)

    model = maskrcnn_resnet50_fpn(weights=None)
    in_box  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, ckpt["num_classes"])
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    label_to_name = ckpt["label_to_name"]
    test_ds = SegmentationDataset(DATASET_ROOT, split="test")
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=2, collate_fn=collate_fn)

    stats = {lbl: [0, 0, 0] for lbl in label_to_name}

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                pred = out["labels"][out["scores"] >= SCORE_THRESH].cpu().tolist()
                true = tgt["labels"].tolist()
                for lbl in stats:
                    tp = min(pred.count(lbl), true.count(lbl))
                    fp = max(0, pred.count(lbl) - true.count(lbl))
                    fn = max(0, true.count(lbl) - pred.count(lbl))
                    stats[lbl][0] += tp
                    stats[lbl][1] += fp
                    stats[lbl][2] += fn

    print(f"\n{'─'*58}")
    print(f"  Mask R-CNN — Test Set  (umbral={SCORE_THRESH})")
    print(f"{'─'*58}")
    print(f"  {'Clase':<15} {'Precisión':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'─'*58}")
    macro = []
    for lbl, name in label_to_name.items():
        tp, fp, fn = stats[lbl]
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        macro.append(f1)
        print(f"  {name:<15} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")
    print(f"{'─'*58}")
    print(f"  {'Macro F1':<15} {sum(macro)/len(macro):>32.3f}")
    print(f"{'─'*58}\n")


# ── EfficientNet ──────────────────────────────────────────────────────────────
def eval_classifier(device):
    path = os.path.join(MODELS_DIR, "best_classifier.pth")
    ckpt = torch.load(path, map_location=device)

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"\n  Clasificador cargado | fold={ckpt['fold']} | "
          f"fold_acc={ckpt['fold_acc']:.4f} | "
          f"5-fold mean={ckpt['mean_acc']:.4f} ± {ckpt['std_acc']:.4f}")

    test_ds = DriverStateDataset(DATASET_ROOT, splits=["test"],
                                 transform=VAL_TRANSFORMS)
    loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n{'─'*58}")
    print(f"  EfficientNet-B0 — Test Set  |  Acc = {acc:.4f}")
    print(f"{'─'*58}")
    print(classification_report(all_labels, all_preds, target_names=STATES))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=STATES, yticklabels=STATES)
    plt.title("Confusion Matrix — EfficientNet-B0 (Test Set)")
    plt.ylabel("Real"); plt.xlabel("Predicho"); plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "classifier_confusion_matrix_test.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"  Matriz → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "maskrcnn", "classifier"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if args.model in ("all", "maskrcnn"):
        eval_maskrcnn(device)
    if args.model in ("all", "classifier"):
        eval_classifier(device)