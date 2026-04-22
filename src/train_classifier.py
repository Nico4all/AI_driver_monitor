"""
train_classifier.py
Entrena EfficientNet-B0 para clasificar el estado del conductor.
Usa 5-fold cross-validation sobre train+valid, luego evalúa en test.

Estados: awake(0) distracted(1) drowsy(2) eyes_closed(3) phone(4)

Uso dentro del contenedor:
    cd /app/src
    python train_classifier.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from dataset_classifier import (
    get_all_samples, SampleListDataset,
    DriverStateDataset, STATES,
    TRAIN_TRANSFORMS, VAL_TRANSFORMS,
)

# ── Hiperparámetros ───────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
NUM_CLASSES  = 5
NUM_EPOCHS   = 20
BATCH_SIZE   = 16
LR           = 1e-3
NUM_FOLDS    = 5
NUM_WORKERS  = 2
# ─────────────────────────────────────────────────────────────────────────────


def build_model(device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def run_kfold():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  EfficientNet-B0 — 5-Fold Cross-Validation")
    print(f"  Dispositivo : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"{'='*55}\n")

    # Combinar train + valid para k-fold
    all_samples = get_all_samples(DATASET_ROOT, splits=["train", "valid"])
    labels_arr  = np.array([lbl for _, lbl in all_samples])
    print(f"  Total muestras (train+valid): {len(all_samples)}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels_arr)), labels_arr), start=1):

        print(f"\n  {'─'*50}")
        print(f"  Fold {fold}/{NUM_FOLDS}  |  "
              f"train={len(train_idx)}  val={len(val_idx)}")
        print(f"  {'─'*50}")

        train_samples = [all_samples[i] for i in train_idx]
        val_samples   = [all_samples[i] for i in val_idx]

        train_ds = SampleListDataset(train_samples, transform=TRAIN_TRANSFORMS)
        val_ds   = SampleListDataset(val_samples,   transform=VAL_TRANSFORMS)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        model     = build_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS)

        best_val_acc = 0.0
        best_path    = os.path.join(MODELS_DIR, f"efficientnet_fold{fold}.pth")
        history      = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                              criterion, optimizer, device)
            vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)

            print(f"  Fold {fold} Ep {epoch:>2}/{NUM_EPOCHS} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} | "
                  f"{time.time()-t0:.1f}s")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), best_path)

        print(f"\n  ✓ Fold {fold} mejor val_acc = {best_val_acc:.4f}\n")
        fold_results.append(best_val_acc)
        _save_fold_plot(history, fold)

    # ── Resumen k-fold ────────────────────────────────────────────────────────
    mean_acc = np.mean(fold_results)
    std_acc  = np.std(fold_results)
    print(f"\n{'='*55}")
    print(f"  K-Fold Resultados:")
    for i, acc in enumerate(fold_results, 1):
        print(f"    Fold {i}: {acc:.4f}")
    print(f"  Media  : {mean_acc:.4f}")
    print(f"  Std    : {std_acc:.4f}")
    print(f"{'='*55}\n")

    # ── Guardar el mejor fold como modelo final ───────────────────────────────
    best_fold = int(np.argmax(fold_results)) + 1
    best_fold_path  = os.path.join(MODELS_DIR, f"efficientnet_fold{best_fold}.pth")
    final_path      = os.path.join(MODELS_DIR, "best_classifier.pth")

    model = build_model(device)
    model.load_state_dict(torch.load(best_fold_path, map_location=device))
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes":      NUM_CLASSES,
        "label_to_name":    {i: s for i, s in enumerate(STATES)},
        "fold":             best_fold,
        "fold_acc":         fold_results[best_fold - 1],
        "mean_acc":         mean_acc,
        "std_acc":          std_acc,
    }, final_path)
    print(f"  ✓ Modelo final guardado (fold {best_fold}) → {final_path}\n")

    # ── Evaluar en test set ───────────────────────────────────────────────────
    _eval_on_test(model, device, criterion)


def _eval_on_test(model, device, criterion):
    test_ds = DriverStateDataset(
        os.path.join(os.path.dirname(__file__), "..", "dataset"),
        splits=["test"], transform=VAL_TRANSFORMS,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"  Evaluación en TEST SET | Acc = {acc:.4f}")
    print(f"{'='*55}")
    print(classification_report(labels, preds, target_names=STATES))

    # Matriz de confusión
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=STATES, yticklabels=STATES)
    plt.title("Confusion Matrix — EfficientNet-B0 (Test Set)")
    plt.ylabel("Real"); plt.xlabel("Predicho")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                       "classifier_confusion_matrix.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"\n  Matriz de confusión → {out}\n")


def _save_fold_plot(history, fold):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title(f"Fold {fold} — Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title(f"Fold {fold} — Accuracy")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs",
                       f"classifier_fold{fold}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120); plt.close()


if __name__ == "__main__":
    run_kfold()