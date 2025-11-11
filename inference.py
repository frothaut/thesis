#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

# =========================
# H A R D C O D E D  CONFIG
# =========================
CHECKPOINT_PATH = "best_unet.pth"               # gespeichertes Modell
INPUT_DIR       = "predictions/testdaten"       # Eingabe-Bilder
OUTPUT_DIR      = "predictions/predictions"     # Ausgabepfade werden erstellt
GT_DIR          = r"E:\Rothaut_Masterthesis\thesis\images\masks"                 # Ground-Truth-Masken im Originalverzeichnis
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def to_grayscale_mask(pred_idx: np.ndarray, class_values: list[int]) -> np.ndarray:
    out = np.zeros_like(pred_idx, dtype=np.uint8)
    for i, val in enumerate(class_values):
        out[pred_idx == i] = val
    return out


def make_overlay(rgb_img: Image.Image, mask_idx: np.ndarray, alpha: float = 0.5) -> Image.Image:
    palette = np.array([
        [0, 0, 0], [255, 255, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]
    ], dtype=np.uint8)
    colors = palette[mask_idx % len(palette)]
    color_mask = Image.fromarray(colors, mode="RGB").resize(rgb_img.size, Image.NEAREST)

    overlay = rgb_img.convert("RGBA").copy()
    cm = color_mask.convert("RGBA")
    cm.putalpha(int(alpha * 255))
    overlay.alpha_composite(cm)
    return overlay


def class_index_map(class_values: list[int]) -> dict[int, int]:
    return {v: i for i, v in enumerate(class_values)}


def apply_exclusion_rule(
    pred_idx: np.ndarray,
    class_values: list[int],
    threshold: float = 0.01
) -> np.ndarray:
    """
    Originale Regel:
    - Wenn Klasse 40 (gelb) ODER 80 (blau) mindestens `threshold` Anteil
      der Pixel erreicht, wird Klasse 255 (rot) ausgeschlossen (auf Hintergrund 0 gesetzt).
    """
    v2i = class_index_map(class_values)
    idx_yellow = v2i.get(40, None)
    idx_blue   = v2i.get(80, None)
    idx_red    = v2i.get(255, None)
    idx_bg     = v2i.get(0, 0)

    total = pred_idx.size
    frac_y = (np.sum(pred_idx == idx_yellow) / total) if idx_yellow is not None else 0.0
    frac_b = (np.sum(pred_idx == idx_blue)   / total) if idx_blue   is not None else 0.0

    if (frac_y >= threshold or frac_b >= threshold) and idx_red is not None:
        pred_idx = pred_idx.copy()
        pred_idx[pred_idx == idx_red] = idx_bg
    return pred_idx


def load_gt_index_mask(gt_dir: Path, stem: str, class_values: list[int]) -> np.ndarray | None:
    nr = int(stem.replace("DJI_","").replace(".jpg",""))
    gt_path = gt_dir / f"mask_0{nr}.png"
    print(gt_path)
    if not gt_path.exists():
        return None
    arr = np.array(Image.open(gt_path).convert("L"))
    lut = np.full(256, -1, dtype=np.int16)
    for i, v in enumerate(class_values):
        lut[int(v)] = i
    idx = lut[arr]
    idx[idx < 0] = 0
    return idx.astype(np.int32)


@torch.no_grad()
def predict_img(model: nn.Module, full_img: Image.Image, device: torch.device):
    tf = build_transform()
    model.eval()
    if SCALE_FACTOR != 1.0:
        w, h = full_img.size
        full_img = full_img.resize((int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)), Image.Resampling.LANCZOS)

    img_tensor = tf(full_img).unsqueeze(0).to(device=device, dtype=torch.float32)
    logits = model(img_tensor)
    logits = F.interpolate(logits, size=full_img.size[::-1], mode='bilinear', align_corners=False)
    probs = F.softmax(logits, dim=1)
    max_probs, pred = torch.max(probs, dim=1)

    pred_idx = pred.squeeze(0).cpu().numpy().astype(np.int32)
    pred_idx = apply_exclusion_rule(pred_idx, CLASS_VALUES)  # <— Regel angewendet
    max_probs_np = max_probs.squeeze(0).cpu().numpy().astype(np.float32)
    return pred_idx, full_img, max_probs_np


def binary_confusion_per_class(pred_idx: np.ndarray, gt_idx: np.ndarray, n_classes: int):
    """Compute TP, FP, FN, TN per class for a single pair of masks."""
    H, W = pred_idx.shape
    assert gt_idx.shape == (H, W)

    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    tn = np.zeros(n_classes, dtype=np.int64)

    for c in range(n_classes):
        pred_c = (pred_idx == c)
        gt_c   = (gt_idx == c)
        tp[c] = int(np.logical_and(pred_c, gt_c).sum())
        fp[c] = int(np.logical_and(pred_c, ~gt_c).sum())
        fn[c] = int(np.logical_and(~pred_c, gt_c).sum())
        tn[c] = int(np.logical_and(~pred_c, ~gt_c).sum())
    return tp, fp, fn, tn

def plot_per_class_binary_confusion(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray, out_path: Path):
    """Plot one subplot per class with bars for TP, FP, FN, TN in percent."""
    C = tp.shape[0]
    cols = min(5, C)
    rows = int(np.ceil(C / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for c in range(C):
        r, k = divmod(c, cols)
        ax = axes[r, k]

        total = tp[c] + fp[c] + fn[c] + tn[c]
        if total == 0:
            values = [0, 0, 0, 0]
        else:
            values = [v / total * 100 for v in [tp[c], fp[c], fn[c], tn[c]]]

        ax.bar(["TP", "FP", "FN", "TN"], values)
        ax.set_title(f"Class {c}")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}%", ha='center', va='bottom')

    # Hide any unused subplots
    total_axes = rows * cols
    for idx in range(C, total_axes):
        r, k = divmod(idx, cols)
        fig.delaxes(axes[r, k])

    fig.suptitle("Binary Confusion (TP/FP/FN/TN) per Class [%]", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: int, out_path: Path):
    """Plot normalized confusion matrix (in percent)."""
    cm_percent = cm.astype(float)
    row_sums = cm_percent.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm_percent, interpolation='nearest')
    ax.set_title('Confusion Matrix (aggregated, %)')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(classes))
    ax.set_yticks(range(classes))
    ax.set_xticklabels([str(i) for i in range(classes)])
    ax.set_yticklabels([str(i) for i in range(classes)])

    # Prozentwerte annotieren
    for i in range(classes):
        for j in range(classes):
            ax.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(N_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    gt_dir = Path(GT_DIR)
    ensure_dir(out_dir)

    files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in VALID_EXTS]
    if not files:
        print(f"Keine Eingabebilder in {in_dir} gefunden. Erwarte Endungen: {VALID_EXTS}")
        return

    tp_sum = np.zeros(N_CLASSES, dtype=np.int64)
    fp_sum = np.zeros(N_CLASSES, dtype=np.int64)
    fn_sum = np.zeros(N_CLASSES, dtype=np.int64)
    tn_sum = np.zeros(N_CLASSES, dtype=np.int64)
    conf_sum_per_class = np.zeros(N_CLASSES, dtype=np.float64)
    conf_count_per_class = np.zeros(N_CLASSES, dtype=np.int64)

    for img_path in files:
        img = Image.open(img_path).convert("RGB")
        pred_idx, resized_img, max_probs = predict_img(model, img, device)

        mask_gray = to_grayscale_mask(pred_idx, CLASS_VALUES)
        mask_img = Image.fromarray(mask_gray).convert('L')

        stem = img_path.stem
        mask_out = out_dir / f"{stem}_mask.png"
        mask_img.save(mask_out)

        if SAVE_OVERLAY:
            overlay_img = make_overlay(resized_img, pred_idx, OVERLAY_ALPHA)
            overlay_out = out_dir / f"{stem}_overlay.png"
            overlay_img.save(overlay_out)

        for c in range(N_CLASSES):
            m = (pred_idx == c)
            if m.any():
                conf_sum_per_class[c] += float(max_probs[m].sum())
                conf_count_per_class[c] += int(m.sum())

        gt_idx = load_gt_index_mask(gt_dir, stem, CLASS_VALUES)
        if gt_idx is not None:
            if gt_idx.shape != pred_idx.shape:
                gt_img = Image.fromarray(gt_idx.astype(np.uint8), mode="L")
                gt_img = gt_img.resize(pred_idx.shape[::-1], Image.NEAREST)
                gt_idx = np.array(gt_img, dtype=np.int32)
            tp, fp, fn, tn = binary_confusion_per_class(pred_idx, gt_idx, N_CLASSES)
            tp_sum += tp; fp_sum += fp; fn_sum += fn; tn_sum += tn
        else:
            print(f"[Hinweis] Keine GT für {stem} gefunden.")

        print(f"Gespeichert: {mask_out}")
        if SAVE_OVERLAY:
            print(f"Gespeichert: {overlay_out}")

    # Plot pro Klasse (TP/FP/FN/TN)
    cm_png = out_dir / "binary_confusion_per_class.png"
    plot_per_class_binary_confusion(tp_sum, fp_sum, fn_sum, tn_sum, cm_png)
    print(f"Binary-Konfusion gespeichert als Bild: {cm_png}")

    # Ein Score (Prozent) pro Klasse: mittlere Top-1-Confidence über alle Pixel der jeweiligen Vorhersageklasse
    mean_conf_per_class = np.full(N_CLASSES, np.nan, dtype=np.float64)
    for c in range(N_CLASSES):
        if conf_count_per_class[c] > 0:
            mean_conf_per_class[c] = 100.0 * (conf_sum_per_class[c] / conf_count_per_class[c])

    print("\nKonfidenz-Score je Klasse (%):")
    for c, v in enumerate(mean_conf_per_class):
        val = f"{v:.2f}%" if not np.isnan(v) else "NaN"
        print(f"Klasse {c}: {val}")

    print(f"\nFertig. Ergebnisse liegen in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
