#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# =========================
# H A R D C O D E D  CONFIG
# =========================
CHECKPOINT_PATH = "best_unet.pth"               # dein gespeichertes Modell
INPUT_DIR       = "predictions/testdaten"       # Eingabe-Bilder
OUTPUT_DIR      = "predictions/predictions"     # Ausgabepfade werden erstellt
CLASS_VALUES    = [0, 40, 80, 150, 255]             # Graustufen je Klassenindex 0 = background, 40 = blue, 80 = yellow, 150= green, 255 = red
N_CLASSES       = 5
SCALE_FACTOR    = 0.5                           # wie im Training: Breite/Höhe halbieren
SAVE_OVERLAY    = True                          # optionales Overlay-PNG zusätzlich speichern
OVERLAY_ALPHA   = 0.5                           # Transparenz für Overlay
VALID_EXTS      = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
GT_DIR = "predictions/test_masks"

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
def load_gt_mask_as_indices(gt_path: Path, target_shape: tuple[int, int], class_values: list[int]) -> np.ndarray:
    """
    Lädt eine GT-Graustufenmaske, resampelt sie (falls nötig) auf target_shape (H, W)
    und gibt ein Array aus Klassenindizes (0..N-1) zurück.
    Pixel mit Werten, die nicht in class_values vorkommen, erhalten Index -1.
    """
    img = Image.open(gt_path).convert("L")

    h, w = target_shape
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)

    gt_gray = np.array(img, dtype=np.uint8)
    v2i = class_index_map(class_values)

    gt_idx = np.full(gt_gray.shape, -1, dtype=np.int32)
    for i, v in enumerate(class_values):
        gt_idx[gt_gray == v] = i

    return gt_idx

def update_confusion(conf_counts: np.ndarray,
                     pred_idx: np.ndarray,
                     gt_idx: np.ndarray):
    """
    Aktualisiert die Konfusionsmatrix (TP, FP, TN, FN) pro Klasse.
    conf_counts: shape (N_CLASSES, 4) mit Spalten [TP, FP, TN, FN].
    """
    assert conf_counts.shape[1] == 4, "conf_counts muss Form (N_CLASSES, 4) haben"

    # Nur Pixel mit gültigen GT-Klassen (gt_idx >= 0) berücksichtigen
    valid = (gt_idx >= 0) & (gt_idx != 0)
    if not np.any(valid):
        return

    n_classes = conf_counts.shape[0]
    for c in range(1, n_classes):
        pred_c = (pred_idx == c)
        gt_c   = (gt_idx == c)

        tp = np.sum(pred_c & gt_c & valid)
        fp = np.sum(pred_c & (~gt_c) & valid)
        fn = np.sum((~pred_c) & gt_c & valid)
        tn = np.sum((~pred_c) & (~gt_c) & valid)

        conf_counts[c, 0] += tp
        conf_counts[c, 1] += fp
        conf_counts[c, 2] += tn
        conf_counts[c, 3] += fn


def print_confusion(conf_counts: np.ndarray, class_values: list[int]):
    """
    Gibt TP, FP, TN, FN in Prozent für jede Klasse aus.
    Prozentwerte beziehen sich jeweils auf TP+FP+TN+FN der Klasse.
    """
    n_classes = conf_counts.shape[0]
    print("\nKonfusionsmatrix (Pixel-basiert, pro Klasse):")
    print("Klasse (Grauwert) | TP%    | FP%    | TN%    | FN%    | (Counts TP/FP/TN/FN)")

    for c in range(1, n_classes):
        tp, fp, tn, fn = conf_counts[c]
        total = tp + fp + tn + fn
        if total == 0:
            tp_p = fp_p = tn_p = fn_p = 0.0
        else:
            tp_p = 100.0 * tp / total
            fp_p = 100.0 * fp / total
            tn_p = 100.0 * tn / total
            fn_p = 100.0 * fn / total

        gv = class_values[c] if c < len(class_values) else c
        print(f"{c} ({gv:3d})          | "
              f"{tp_p:6.2f} | {fp_p:6.2f} | {tn_p:6.2f} | {fn_p:6.2f} | "
              f"({tp}/{fp}/{tn}/{fn})")
def make_overlay(rgb_img: Image.Image, mask_idx: np.ndarray, alpha: float = 0.5) -> Image.Image:
    palette = np.array([
        [0,   0,   0],     # index 0 -> 0 (background) -> black
        [255, 255, 0],     # index 1 -> 40 (blue)      -> blue
        [0, 0,   255],     # index 2 -> 80 (yellow)    -> yellow
        [0,  255, 0],      # index 3 -> 150 (green)    -> green
        [255, 0,   0],     # index 4 -> 255 (red)      -> red
    ], dtype=np.uint8)
    colors = palette[mask_idx % len(palette)]
    color_mask = Image.fromarray(colors, mode="RGB").resize(rgb_img.size, Image.NEAREST)

    overlay = rgb_img.convert("RGBA").copy()
    cm = color_mask.convert("RGBA")
    cm.putalpha(int(alpha * 255))
    overlay.alpha_composite(cm)
    return overlay

def apply_exclusion_rule(pred_idx: np.ndarray, class_values: list[int],threshold: float = 0.01) -> np.ndarray:
    """
    Wenn 40 (blue) ODER 80 (yellow) irgendwo vorhergesagt wurden,
    dann DARF 255 (rot) NICHT im Ergebnis vorkommen -> setze 150 -> 0 (background).
    """
    v2i = class_index_map(class_values)
    idx_blue   = v2i.get(80, None)
    idx_yellow = v2i.get(40, None)
    idx_red  = v2i.get(255, None)
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

@torch.no_grad()
def predict_img(model: nn.Module,
                full_img: Image.Image,
                device: torch.device) -> np.ndarray:
    """
    Gibt die vorhergesagten Klassenindizes (H x W) als numpy.int32 zurück.
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
    logits = F.interpolate(logits, size=full_img.size[::-1], mode='bilinear', align_corners=False)
    probs  = F.softmax(logits, dim=1)
    pred   = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    pred = apply_exclusion_rule(pred, CLASS_VALUES)
    return pred, full_img  # Bild ggf. skaliert zurückgeben fürs Overlay


def load_model(checkpoint_path: str, n_classes: int, device: torch.device) -> nn.Module:
    model = UNet(n_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model
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

    # Konfusionsmatrix: pro Klasse [TP, FP, TN, FN]
    conf_counts = np.zeros((N_CLASSES, 4), dtype=np.int64)
    num_with_gt = 0

    for img_path in tqdm(files, desc="Inferenz"):
        img = Image.open(img_path).convert("RGB")
        pred_idx, resized_img = predict_img(model, img, device)

        # Graustufenmaske mit CLASS_VALUES
        mask_gray = to_grayscale_mask(pred_idx, CLASS_VALUES)
        mask_img = Image.fromarray(mask_gray)
        mask_img = mask_img.convert('L')  # echte Graustufe

        stem = img_path.stem
        nr = stem.replace("DJI_", "")
        gt_path = gt_dir / f"mask_{nr}.png"
        mask_out = out_dir / f"{stem}_mask.png"
        mask_img.save(mask_out)

        # Overlay speichern (optional)
        if SAVE_OVERLAY:
            overlay_img = make_overlay(resized_img, pred_idx, OVERLAY_ALPHA)
            overlay_out = out_dir / f"{stem}_overlay.png"
            overlay_img.save(overlay_out)

        
        if gt_path.exists():
            print("Found GT: ", gt_path)
            gt_idx = load_gt_mask_as_indices(gt_path,
                                             target_shape=pred_idx.shape,
                                             class_values=CLASS_VALUES)
            update_confusion(conf_counts, pred_idx, gt_idx)
            num_with_gt += 1
        # Falls es keine passende GT gibt, wird das Bild einfach für die Metrik ignoriert.
        else:
            print("Found no GT for ", stem)
    print(f"Fertig. Ergebnisse liegen in: {out_dir.resolve()}")
    print(f"Anzahl ausgewerteter Bilder mit GT: {num_with_gt}")

    print_confusion(conf_counts, CLASS_VALUES)



if __name__ == "__main__":
    main()