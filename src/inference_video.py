"""
inference_video.py — Driver Monitor (2 modelos en tiempo real)
Corre en Windows FUERA del contenedor Docker.

Requiere (instalar localmente con CUDA):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python numpy

Uso:
    python src/inference_video.py
    python src/inference_video.py --camera 1
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor
from torchvision.models import efficientnet_b0

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.join(os.path.dirname(__file__), "..")
MASKRCNN_PATH = os.path.join(ROOT, "models", "best_maskrcnn.pth")
CLF_PATH      = os.path.join(ROOT, "models", "best_classifier.pth")

# ── Configuración ─────────────────────────────────────────────────────────────
SEG_SIZE      = (432, 432)   # igual que Roboflow
CLF_SIZE      = (224, 224)

SCORE_THRESH  = 0.55
MASK_THRESH   = 0.50
MASK_ALPHA    = 0.38

# Máx ratio área del frame para considerar objeto válido (elimina falsos en cara)
MAX_AREA_RATIO = {"phone": 0.18}

STATES = ["awake", "distracted", "drowsy", "eyes_closed", "phone"]

STATE_LABEL = {
    "awake":       "DESPIERTO",
    "distracted":  "DISTRAIDO",
    "drowsy":      "SOMNOLIENTO",
    "eyes_closed": "OJOS CERRADOS",
    "phone":       "USANDO CELULAR",
}

CLASS_COLORS = {
    "face":      (140, 230, 140),   # verde claro
    "left_eye":  (  0, 100, 255),   # rojo-naranja
    "right_eye": (  0, 200, 255),   # amarillo
    "phone":     ( 30, 180, 255),   # azul
}

STATE_COLORS = {
    "awake":       ( 0, 210,   0),
    "distracted":  ( 0, 150, 255),
    "drowsy":      ( 0,   0, 230),
    "eyes_closed": (160,  0, 160),
    "phone":       (  0, 200, 200),
}

CLF_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize(CLF_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# ─────────────────────────────────────────────────────────────────────────────


def load_maskrcnn(path, device):
    ckpt = torch.load(path, map_location=device)
    model = maskrcnn_resnet50_fpn(weights=None)
    in_box  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, ckpt["num_classes"])
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["label_to_name"]


def load_classifier(path, device):
    ckpt  = torch.load(path, map_location=device)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["label_to_name"]


@torch.no_grad()
def run_maskrcnn(model, frame_bgr, device):
    h, w   = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, SEG_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = TF.to_tensor(rgb).unsqueeze(0).to(device)
    return model(tensor)[0], (h, w)


@torch.no_grad()
def run_classifier(model, frame_bgr, device):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = CLF_TRANSFORM(rgb).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


def box_area_ratio(box, frame_area):
    return ((box[2]-box[0]) * (box[3]-box[1])) / frame_area


def draw(frame, seg_out, seg_labels, clf_probs, orig_size):
    h_orig, w_orig = orig_size
    h_in,  w_in   = SEG_SIZE
    overlay = frame.copy()

    scores = seg_out["scores"].cpu().numpy()
    labels = seg_out["labels"].cpu().numpy()
    boxes  = seg_out["boxes"].cpu().numpy()
    masks  = seg_out["masks"].cpu().numpy()

    # ── Verificar si Mask R-CNN detectó un teléfono real ─────────────────────
    phone_label_id = None
    for lbl_id, name in seg_labels.items():
        if name == "phone":
            phone_label_id = lbl_id
            break

    phone_detected = False
    if phone_label_id is not None:
        for score, lbl, box in zip(scores, labels, boxes):
            if int(lbl) == phone_label_id and score >= SCORE_THRESH:
                # Filtro de área: teléfono real < 20% del frame
                area_ratio = ((box[2]-box[0]) * (box[3]-box[1])) / (w_in * h_in)
                if area_ratio < 0.20:
                    phone_detected = True
                    break

    # ── Verificar si Mask R-CNN detectó ojos ─────────────────────────────────
    eye_ids = {lbl_id for lbl_id, name in seg_labels.items()
               if name in ("left_eye", "right_eye")}
    eyes_detected = any(
        int(lbl) in eye_ids and score >= SCORE_THRESH
        for score, lbl in zip(scores, labels)
    )
    
    # ── Ajustar probabilidades del clasificador ───────────────────────────────
    clf_probs = clf_probs.copy()
    phone_idx       = STATES.index("phone")        # índice 4
    eyes_closed_idx = STATES.index("eyes_closed")  # índice 3

    # Regla 1: sin celular detectado → P(phone) = 0
    if not phone_detected:
        clf_probs[phone_idx] = 0.0

    # Regla 2: sin ojos detectados → forzar 90% eyes_closed
    if not eyes_detected:
        clf_probs[eyes_closed_idx] = 0.90
        # Distribuir el 10% restante entre las demás clases (excepto phone si no hay celular)
        remaining_idx = [i for i in range(len(STATES))
                         if i != eyes_closed_idx and clf_probs[i] > 0]
        remaining_sum = sum(clf_probs[i] for i in remaining_idx)
        if remaining_sum > 0:
            for i in remaining_idx:
                clf_probs[i] = clf_probs[i] / remaining_sum * 0.10
        else:
            # Si no queda nada, poner todo en eyes_closed
            clf_probs[eyes_closed_idx] = 1.0

    # Renormalizar para que sume 1
    total = clf_probs.sum()
    if total > 0:
        clf_probs = clf_probs / total

    # ── Dibujar máscaras y contornos ──────────────────────────────────────────
    for score, lbl, box, mask in zip(scores, labels, boxes, masks):
        if score < SCORE_THRESH:
            continue

        name  = seg_labels.get(int(lbl), f"cls{lbl}")
        color = CLASS_COLORS.get(name, (200, 200, 200))

        # Filtro de área (evita phone detectado en la cara)
        if name in MAX_AREA_RATIO:
            if ((box[2]-box[0]) * (box[3]-box[1])) / (w_in * h_in) > MAX_AREA_RATIO[name]:
                continue

        # Escalar al frame original
        x1 = int(box[0] * w_orig / w_in)
        y1 = int(box[1] * h_orig / h_in)
        x2 = int(box[2] * w_orig / w_in)
        y2 = int(box[3] * h_orig / h_in)

        # Máscara semitransparente
        mb = (mask[0] > MASK_THRESH).astype(np.uint8)
        mr = cv2.resize(mb, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        col = np.zeros_like(frame, dtype=np.uint8)
        col[mr == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, col, MASK_ALPHA, 0)

        # Contorno
        contours, _ = cv2.findContours(mr, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Etiqueta pequeña
        txt = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
        cv2.putText(overlay, txt, (x1+2, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Banner de estado (clasificador) ───────────────────────────────────────
    state_idx   = int(np.argmax(clf_probs))
    state_name  = STATES[state_idx]
    state_conf  = clf_probs[state_idx]
    state_color = STATE_COLORS[state_name]
    label_text  = f"{STATE_LABEL[state_name]}   {state_conf:.0%}"

    cv2.rectangle(overlay, (0, 0), (w_orig, 68), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, 0), (w_orig, 68), state_color, 4)
    (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
    cv2.putText(overlay, label_text, ((w_orig-tw)//2, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, state_color, 2)

    # ── Barra de probabilidades ───────────────────────────────────────────────
    bar_y = h_orig - 30
    bw    = w_orig // len(STATES) - 4
    for i, (s, prob) in enumerate(zip(STATES, clf_probs)):
        bx = 4 + i * (bw + 4)
        filled = int(bw * prob)
        cv2.rectangle(overlay, (bx, bar_y), (bx+bw, bar_y+20), (50,50,50), -1)
        cv2.rectangle(overlay, (bx, bar_y), (bx+filled, bar_y+20),
                      STATE_COLORS[s], -1)
        cv2.putText(overlay, s[:4], (bx+2, bar_y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)

    return overlay


def run(camera_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    seg_model, seg_labels = load_maskrcnn(MASKRCNN_PATH, device)
    clf_model, _          = load_classifier(CLF_PATH, device)
    print("Modelos cargados. Abriendo cámara...\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {camera_idx}")
        sys.exit(1)

    print("Presiona 'q' para salir.\n")
    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_size  = frame.shape[:2]
        seg_out, _ = run_maskrcnn(seg_model, frame, device)
        clf_probs  = run_classifier(clf_model, frame, device)
        result     = draw(frame, seg_out, seg_labels, clf_probs, orig_size)

        curr_t = time.time()
        fps    = 1 / (curr_t - prev_t + 1e-6)
        prev_t = curr_t
        dev    = "GPU" if device.type == "cuda" else "CPU"
        cv2.putText(result, f"{fps:.1f} FPS [{dev}]",
                    (10, orig_size[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Driver Monitor — Mask R-CNN + EfficientNet-B0", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()
    run(args.camera)