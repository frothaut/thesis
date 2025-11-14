import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# =========================
# H A R D C O D E D  CONFIG
# =========================
CHECKPOINT_PATH = "best_unet.pth"               # dein gespeichertes Modell
INPUT_DIR       = "predictions/testdaten"       # Eingabe-Bilder
OUTPUT_DIR      = "predictions/predictions"     # Ausgabepfade werden erstellt
GT_DIR          = "predictions/gt"              # Ground-Truth-Masken (Graustufen), gleicher Dateistamm
CLASS_VALUES    = [0, 40, 80, 150, 255]         # Graustufen je Klassenindex (0..N_CLASSES-1)
N_CLASSES       = 5
SCALE_FACTOR    = 0.5                           # wie im Training: Breite/Höhe halbieren
SAVE_OVERLAY    = True                          # optionales Overlay-PNG zusätzlich speichern
OVERLAY_ALPHA   = 0.5                           # Transparenz für Overlay
VALID_EXTS      = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# ==============
# Modell-Definition
# ==============
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes, base_c=64, dropout=0.3):
        super().__init__()
        self.down1 = DoubleConv(3, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_c, base_c*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_c*2, base_c*4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_c*4, base_c*8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_c*8, base_c*16, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_c*16, base_c*8)
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_c*8, base_c*4)
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_c*2, base_c)

        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        c4 = self.dec4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(c4)
        c3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))

        return self.outc(c1)

# ==================
# Hilfsfunktionen
# ==================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def to_grayscale_mask(pred_idx: np.ndarray, class_values: list[int]) -> np.ndarray:
    out = np.zeros_like(pred_idx, dtype=np.uint8)
    for i, val in enumerate(class_values):
        out[pred_idx == i] = val
    return out

def class_index_map(class_values: list[int]) -> dict[int, int]:
    """Erzeugt Mapping von Klasse-WERT -> Klassen-INDEX."""
    return {v: i for i, v in enumerate(class_values)}

def make_overlay(rgb_img: Image.Image, mask_idx: np.ndarray, alpha: float = 0.5) -> Image.Image:
    # Farben sind symbolisch; Kommentare nennen die Zielklassen
    palette = np.array([
        [0,   0,   0],   # index 0 -> 0 (background)
        [255, 255, 0],   # index 1 -> 40
        [0,   0, 255],   # index 2 -> 80
        [0, 255,   0],   # index 3 -> 150
        [255,   0, 0],   # index 4 -> 255
    ], dtype=np.uint8)
    colors = palette[mask_idx % len(palette)]
    color_mask = Image.fromarray(colors, mode="RGB").resize(rgb_img.size, Image.NEAREST)

    overlay = rgb_img.convert("RGBA").copy()
    cm = color_mask.convert("RGBA")
    cm.putalpha(int(alpha * 255))
    overlay.alpha_composite(cm)
    return overlay

def apply_exclusion_rule(
    pred_idx: np.ndarray, 
    class_values: list[int], 
    threshold: float = 0.01
) -> np.ndarray:
    """
    Exclusion rule:
    - If class 40 (yellow) or 80 (blue) reaches at least threshold
      proportion of all pixels, then class 255 (red) is excluded
      (replaced by background 0).
    
    Args:
        pred_idx: ndarray with class indices (not raw grayscale).
        class_values: list of grayscale values used for classes.
        threshold: minimum fraction of pixels (0–1) required to trigger exclusion.
    """
    v2i = class_index_map(class_values)
    idx_yellow = v2i.get(40, None)
    idx_blue   = v2i.get(80, None)
    idx_red    = v2i.get(255, None)
    idx_bg     = v2i.get(0, 0)

    total_pixels = pred_idx.size
    frac_yellow = (
        np.sum(pred_idx == idx_yellow) / total_pixels if idx_yellow is not None else 0
    )
    frac_blue = (
        np.sum(pred_idx == idx_blue) / total_pixels if idx_blue is not None else 0
    )

    if (frac_yellow >= threshold or frac_blue >= threshold) and idx_red is not None:
        pred_idx = pred_idx.copy()
        pred_idx[pred_idx == idx_red] = idx_bg

    return pred_idx

def find_matching_file(dirpath: Path, stem: str):
    for ext in VALID_EXTS:
        p = dirpath / f"{stem}{ext}"
        if p.exists():
            return p
    # häufig sind GT-Masken .png – versuche das als Fallback
    p = dirpath / f"{stem}.png"
    return p if p.exists() else None

def load_gt_index_mask(gt_dir: Path, stem: str, class_values: list[int]) -> np.ndarray | None:
    """
    Lädt eine GT-Maske als Klassenindex-Array (H x W), gemappt via CLASS_VALUES.
    Gibt None zurück, wenn keine Datei gefunden wird.
    """
    gt_path = find_matching_file(gt_dir, stem)
    if gt_path is None:
        return None

    arr = np.array(Image.open(gt_path).convert("L"))
    # Mapping Graustufe -> Klassenindex
    lut = np.full(256, -1, dtype=np.int16)
    for i, v in enumerate(class_values):
        lut[int(v)] = i
    idx = lut[arr]
    # Unbekannte Werte -> Hintergrund (0) und Warnung auf Konsole
    if (idx < 0).any():
        unknown = int((idx < 0).sum())
        print(f"[Warnung] {unknown} Pixel mit unbekanntem GT-Wert in {gt_path.name} -> setze auf Klasse 0")
        idx[idx < 0] = 0
    return idx.astype(np.int32)

@torch.no_grad()
def predict_img(model: nn.Module,
                full_img: Image.Image,
                device: torch.device):
    """
    Returns:
      pred_idx: (H x W) Klassenindizes (np.int32)
      resized_img: ggf. skaliertes PIL-Bild (für Overlay)
      max_probs: (H x W) Top-1-Confidences (np.float32)
      probs: (C x H x W) vollständige Softmax-Wahrscheinlichkeiten (np.float32)
    """
    tf = build_transform()
    model.eval()

    # Resize wie im Training (halbieren)
    if SCALE_FACTOR != 1.0:
        w, h = full_img.size
        full_img = full_img.resize((int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)),
                                   Image.Resampling.LANCZOS)

    img_tensor = tf(full_img).unsqueeze(0).to(device=device, dtype=torch.float32)
    logits = model(img_tensor)  # [1, C, h', w']

    # Sicherheitshalber auf die (ggf. geänderte) Bildgröße bringen
    logits = F.interpolate(logits, size=full_img.size[::-1], mode='bilinear', align_corners=False)
    probs  = F.softmax(logits, dim=1)  # [1, C, H, W]
    max_probs, pred = torch.max(probs, dim=1)  # [1, H, W]

    pred_idx = pred.squeeze(0).cpu().numpy().astype(np.int32)
    pred_idx = apply_exclusion_rule(pred_idx, CLASS_VALUES)

    max_probs_np = max_probs.squeeze(0).cpu().numpy().astype(np.float32)
    probs_np = probs.squeeze(0).cpu().numpy().astype(np.float32)  # (C, H, W)

    return pred_idx, full_img, max_probs_np, probs_np

def compute_image_stats_without_gt(stem: str,
                                   pred_idx: np.ndarray,
                                   max_probs: np.ndarray,
                                   probs: np.ndarray):
    """
    Metriken ohne Ground-Truth (wie zuvor).
    """
    H, W = pred_idx.shape
    total_px = H * W
    flat_probs = probs.reshape(probs.shape[0], -1)  # (C, N)
    eps = 1e-12
    entropy_per_px = -np.sum(flat_probs * np.log(flat_probs + eps), axis=0)
    mean_entropy = float(np.mean(entropy_per_px))

    stats = {
        "file_stem": stem,
        "width": W,
        "height": H,
        "pixels": int(total_px),
        "mean_confidence": float(np.mean(max_probs)),
        "median_confidence": float(np.median(max_probs)),
        "mean_entropy": mean_entropy,
        "num_present_classes_pred": int(len(np.unique(pred_idx)))
    }

    # per Klasse: Anteil & mean Top-1-Conf über vorhergesagte Pixel
    for cls_idx in range(probs.shape[0]):
        mask_pred = (pred_idx == cls_idx)
        pct = float(mask_pred.mean()) if total_px > 0 else 0.0
        stats[f"pct_pred_class_{cls_idx}"] = pct
        stats[f"mean_top1_conf_predclass_{cls_idx}"] = float(np.mean(max_probs[mask_pred])) if mask_pred.any() else np.nan

    return stats

def confusion_from_labels(pred_idx: np.ndarray, gt_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Liefert (C x C) Konfusionsmatrix mit counts.
    rows = GT, cols = PRED
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    flat_pred = pred_idx.reshape(-1)
    flat_gt   = gt_idx.reshape(-1)
    for c_true in range(n_classes):
        mask = (flat_gt == c_true)
        if not mask.any():
            continue
        preds = flat_pred[mask]
        counts = np.bincount(preds, minlength=n_classes)
        cm[c_true, :] += counts
    return cm

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else np.nan

def compute_gt_metrics(pred_idx: np.ndarray,
                       gt_idx: np.ndarray,
                       probs: np.ndarray,
                       max_probs: np.ndarray):
    """
    Berechnet pro Bild:
      - overall_pixel_acc
      - mIoU (über alle Klassen) und mIoU_gt_present (nur Klassen, die im GT vorkommen)
      - pro Klasse: IoU, Dice, Precision, Recall
      - per-Klasse Confidences:
          * mean_top1_conf_predclass_c (bereits oben, aber hier nochmal korrekt/inkorrekt getrennt)
          * mean_top1_conf_correct_c  (GT=c & PRED=c)
          * mean_top1_conf_incorrect_predclass_c (PRED=c & GT!=c)
          * mean_prob_trueclass_c (mittlere Softmax-Wahrscheinlichkeit für die TRUE-Klasse über GT=c)
    """
    C = probs.shape[0]
    cm = confusion_from_labels(pred_idx, gt_idx, C)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    overall_acc = safe_div(correct, total)

    ious, dices, precs, recs = [], [], [], []
    iou_per_class = {}
    dice_per_class = {}
    prec_per_class = {}
    rec_per_class = {}

    # Per-Klasse Confidences
    mean_top1_conf_correct = {}
    mean_top1_conf_incorrect_pred = {}
    mean_prob_trueclass = {}

    for c in range(C):
        TP = int(cm[c, c])
        FP = int(cm[:, c].sum() - TP)
        FN = int(cm[c, :].sum() - TP)
        denom_iou = TP + FP + FN
        denom_dice = 2*TP + FP + FN

        iou = safe_div(TP, denom_iou)
        dice = safe_div(2*TP, denom_dice)
        prec = safe_div(TP, TP + FP)
        rec  = safe_div(TP, TP + FN)

        ious.append(iou); dices.append(dice); precs.append(prec); recs.append(rec)
        iou_per_class[c] = iou
        dice_per_class[c] = dice
        prec_per_class[c] = prec
        rec_per_class[c]  = rec

        # Confidences je Klasse
        mask_true_c = (gt_idx == c)
        mask_pred_c = (pred_idx == c)
        mask_correct = mask_true_c & mask_pred_c
        mask_incorrect_pred = mask_pred_c & (~mask_true_c)

        mean_top1_conf_correct[c] = float(np.mean(max_probs[mask_correct])) if mask_correct.any() else np.nan
        mean_top1_conf_incorrect_pred[c] = float(np.mean(max_probs[mask_incorrect_pred])) if mask_incorrect_pred.any() else np.nan

        # Mittlere Wahrscheinlichkeit für die TRUE-Klasse über alle Pixel mit GT=c
        # probs hat Shape (C, H, W)
        if mask_true_c.any():
            true_probs = probs[c][mask_true_c]
            mean_prob_trueclass[c] = float(np.mean(true_probs))
        else:
            mean_prob_trueclass[c] = np.nan

    # mIoU über alle Klassen:
    miou_all = float(np.nanmean(ious)) if len(ious) else np.nan
    # mIoU nur über Klassen, die im GT vorkamen:
    present_gt = [c for c in range(C) if cm[c, :].sum() > 0]
    miou_gt_present = float(np.nanmean([iou_per_class[c] for c in present_gt])) if present_gt else np.nan

    metrics = {
        "overall_pixel_acc": overall_acc,
        "miou_all": miou_all,
        "miou_gt_present": miou_gt_present,
    }

    # pro Klasse ausgeben
    for c in range(C):
        metrics[f"iou_{c}"] = iou_per_class[c]
        metrics[f"dice_{c}"] = dice_per_class[c]
        metrics[f"precision_{c}"] = prec_per_class[c]
        metrics[f"recall_{c}"] = rec_per_class[c]
        metrics[f"mean_top1_conf_correct_{c}"] = mean_top1_conf_correct[c]
        metrics[f"mean_top1_conf_incorrect_predclass_{c}"] = mean_top1_conf_incorrect_pred[c]
        metrics[f"mean_prob_trueclass_{c}"] = mean_prob_trueclass[c]

    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CHECKPOINT_PATH, N_CLASSES, device)

    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    gt_dir = Path(GT_DIR)
    ensure_dir(out_dir)

    files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in VALID_EXTS]
    if not files:
        print(f"Keine Eingabebilder in {in_dir} gefunden. Erwarte Endungen: {VALID_EXTS}")
        return

    all_rows = []

    for img_path in tqdm(files, desc="Inferenz"):
        img = Image.open(img_path).convert("RGB")
        pred_idx, resized_img, max_probs, probs = predict_img(model, img, device)

        # Graustufenmaske mit CLASS_VALUES
        mask_gray = to_grayscale_mask(pred_idx, CLASS_VALUES)
        mask_img = Image.fromarray(mask_gray).convert('L')

        stem = img_path.stem
        mask_out = out_dir / f"{stem}_mask.png"
        mask_img.save(mask_out)

        if SAVE_OVERLAY:
            overlay_img = make_overlay(resized_img, pred_idx, OVERLAY_ALPHA)
            overlay_out = out_dir / f"{stem}_overlay.png"
            overlay_img.save(overlay_out)

        # --- Basis-Stats ohne GT ---
        row = compute_image_stats_without_gt(stem, pred_idx, max_probs, probs)

        # --- Ground-Truth laden & Metriken ---
        gt_idx = load_gt_index_mask(gt_dir, stem, CLASS_VALUES)
        if gt_idx is not None:
            if gt_idx.shape != pred_idx.shape:
                # Falls nötig, auf gleiche Größe wie Vorhersage bringen (Nearest, da Labels)
                gt_img = Image.fromarray(gt_idx.astype(np.uint8), mode="L")
                gt_img = gt_img.resize(pred_idx.shape[::-1], Image.NEAREST)
                gt_idx = np.array(gt_img, dtype=np.int32)

            gt_metrics = compute_gt_metrics(pred_idx, gt_idx, probs, max_probs)
            # Anzahl Klassen im GT
            row["num_present_classes_gt"] = int(len(np.unique(gt_idx)))
            row.update(gt_metrics)
        else:
            # keine GT gefunden
            row["num_present_classes_gt"] = np.nan
            # setze alle GT-bezogenen Keys auf NaN konsistent zu N_CLASSES
            row.update({
                "overall_pixel_acc": np.nan,
                "miou_all": np.nan,
                "miou_gt_present": np.nan,
            })
            for c in range(N_CLASSES):
                row[f"iou_{c}"] = np.nan
                row[f"dice_{c}"] = np.nan
                row[f"precision_{c}"] = np.nan
                row[f"recall_{c}"] = np.nan
                row[f"mean_top1_conf_correct_{c}"] = np.nan
                row[f"mean_top1_conf_incorrect_predclass_{c}"] = np.nan
                row[f"mean_prob_trueclass_{c}"] = np.nan

        all_rows.append(row)

    # --- Tabelle schreiben ---
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "auswertung.csv"
    df.to_csv(csv_path, index=False)

    with pd.option_context('display.max_columns', None, 'display.width', 220):
        print("\n=== Zusammenfassung (erste Zeilen) ===")
        print(df.head())
        
        
    print(f"\nFertig. Ergebnisse liegen in: {out_dir.resolve()}")
    print(f"Auswertungs-Tabelle gespeichert als: {csv_path.resolve()}")

def load_model(checkpoint_path: str, n_classes: int, device: torch.device) -> nn.Module:
    model = UNet(n_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

if __name__ == "__main__":
    main()