# %%
import os
import time
import math
import h5py
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

Image.MAX_IMAGE_PIXELS = None
torch.backends.cudnn.benchmark = True

# =========================
#   MODELL (unverändert)
# =========================
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
            layers.insert(3, nn.Dropout2d(dropout))  # nach der ersten ReLU
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
        u4 = self.up4(bn); c4 = self.dec4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(c4); c3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3); c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2); c1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.outc(c1)

# =========================
#   H5 ERZEUGEN
# =========================

def build_h5_dataset(
    img_dir, mask_dir, out_path="dataset.h5",
    class_values=(0,80,150,255),
    patch_size=512, overlap=0.5, resize_half=False,
    val_split=0.2, seed=42, compression="lzf"):
    print("Starting to Build H5 Dataset")
    step = int(patch_size * (1 - overlap))

    # --- Alle Bilder und Masken indexieren ---
    img_files = {re.search(r"(\d+)", f).group(1): os.path.join(img_dir, f)
                 for f in os.listdir(img_dir) if f.lower().endswith(".jpg")}
    mask_files = {re.search(r"(\d+)", f).group(1): os.path.join(mask_dir, f)
                  for f in os.listdir(mask_dir) if f.lower().endswith(".png")}

    # --- Match über Nummer ---
    common_ids = sorted(set(img_files.keys()) & set(mask_files.keys()))
    assert len(common_ids)>0, "Keine passenden Paare gefunden!"

    # --- Patches zählen ---
    def count_patches(img_path):
        w,h = Image.open(img_path).size
        if resize_half: w//=2; h//=2
        nx = 1 + max(0, (w - patch_size) // step)
        ny = 1 + max(0, (h - patch_size) // step)
        return nx*ny

    N = sum(count_patches(img_files[i]) for i in common_ids)
    print(f"Insgesamt {N} Patches aus {len(common_ids)} Bild/Masken-Paaren.")

    # --- H5 anlegen ---
    with h5py.File(out_path, "w") as f:
        img_ds  = f.create_dataset("images", (N, patch_size, patch_size, 3), dtype="uint8",
                                   chunks=(1, patch_size, patch_size, 3), compression=compression, shuffle=True)
        mask_ds = f.create_dataset("masks",  (N, patch_size, patch_size), dtype="uint8",
                                   chunks=(1, patch_size, patch_size), compression=compression, shuffle=True)
        names   = f.create_dataset("orig_id", (N,), dtype=h5py.string_dtype())
        split   = f.create_dataset("split", (N,), dtype="uint8")
        f.create_dataset("class_values", data=np.array(class_values, dtype="int32"))

        i = 0
        rng = np.random.default_rng(seed)
        for cid in common_ids:
            img_path = img_files[cid]
            msk_path = mask_files[cid]

            img = Image.open(img_path).convert("RGB")
            msk = Image.open(msk_path).convert("L")

            if resize_half:
                w,h = img.size
                img = img.resize((w//2, h//2), Image.Resampling.LANCZOS)
                msk = msk.resize((w//2, h//2), Image.Resampling.LANCZOS)

            w,h = img.size
            for y in range(0, h - patch_size + 1, step):
                for x in range(0, w - patch_size + 1, step):
                    ip = np.array(img.crop((x,y,x+patch_size,y+patch_size)), dtype=np.uint8)
                    mp = np.array(msk.crop((x,y,x+patch_size,y+patch_size)), dtype=np.uint8)
                    img_ds[i]  = ip
                    mask_ds[i] = mp
                    names[i]   = cid
                    i+=1

        # Train/Val Split setzen
        idx = np.arange(N); rng.shuffle(idx)
        n_val = int(N*val_split)
        val_idx = set(idx[:n_val])
        for j in range(N):
            split[j] = 1 if j in val_idx else 0

    print(f"Fertig: {out_path}")

# =========================
#   PYTORCH DATASET (H5)
# =========================
class H5SegDataset(Dataset):
    def __init__(self, h5_path, split=0, class_values=(0,80,150,255)):
        super().__init__()
        self.h5_path = h5_path
        self.split_id = split
        self.class_values = np.array(class_values, dtype=np.uint8)

        # Nur einmal Indexe vorbereiten, ohne Handle offen zu lassen
        with h5py.File(self.h5_path, "r") as f:
            splits = f["split"][...]
        self.indices = np.where(splits == self.split_id)[0].astype(np.int64)

        # Lookup-Tabelle für Maske->Klasse
        self.lut = np.full(256, -1, dtype=np.int16)
        for cls_id, v in enumerate(self.class_values):
            self.lut[int(v)] = cls_id

        # Keine Datei hier offen halten!
        self.h5 = None

        self.tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    def _ensure_open(self):
        # Jeder Worker öffnet sein eigenes Handle beim ersten Zugriff
        if self.h5 is None:
            # SWMR + libver helfen bei stabilen parallelen Reads
            self.h5 = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
            self.images = self.h5["images"]
            self.masks  = self.h5["masks"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        i = self.indices[idx]
        img = self.images[i]  # uint8 HWC
        msk = self.masks[i]   # uint8 HW

        aug = self.tf(image=img, mask=msk)
        x   = aug["image"]    # FloatTensor CHW
        m   = aug["mask"].numpy().astype(np.uint8)  # HW

        y_np = self.lut[m]
        if (y_np < 0).any():
            y_np = np.where(y_np < 0, 0, y_np)
        y = torch.from_numpy(y_np.astype(np.int64))
        return x, y

    # Wichtig: Pickling-freundlich machen (Handle nicht mitsenden)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5"] = None
        # images/masks sind nur Views – auch entfernen
        state.pop("images", None)
        state.pop("masks", None)
        return state

    def __del__(self):
        try:
            if getattr(self, "h5", None) is not None:
                self.h5.close()
        except Exception:
            pass

# =========================
#   KLASSEGEWICHTE
# =========================
def compute_class_weights_from_h5(h5_path, split=0, n_classes=4, class_values=(0,80,150,255)):
    with h5py.File(h5_path, "r") as f:
        masks  = f["masks"]
        splits = f["split"][...]
        indices = np.where(splits == split)[0]
        # LUT wie oben
        lut = np.full(256, -1, dtype=np.int16)
        for cls_id, v in enumerate(class_values):
            lut[int(v)] = cls_id

        counts = np.zeros(n_classes, dtype=np.int64)
        for i in tqdm(indices, desc="Class count (train)"):
            m = masks[i]
            mapped = lut[m]
            mapped = np.where(mapped < 0, 0, mapped)  # robustness
            bc = np.bincount(mapped.ravel(), minlength=n_classes)
            counts += bc

    total = counts.sum()
    # inverse frequency
    weights = np.array([ (total/c) if c>0 else 1.0 for c in counts ], dtype=np.float32)
    # optional: normalisieren
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

# =========================
#   TRAINING
# =========================
def dice_coef(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    return ((2 * intersection + eps) / (union + eps)).mean()

def weighted_cross_entropy_loss(logits, masks, class_weights):
    return F.cross_entropy(logits, masks, weight=class_weights)

# =========================
#   MAIN
# =========================
if __name__ == "__main__":
    # --- Parameter ---
    img_dir      = "images/imgs"
    mask_dir     = "images/masks"
    h5_path      = "dataset.h5"
    n_classes    = 5
    class_values = (0, 40, 80, 150, 255)

    patch_size   = 512
    overlap      = 0.5
    resize_half  = False
    val_split    = 0.2
    seed         = 42

    lr       = 1e-4
    epochs   = 10
    bs       = 8
    nw       = 4  # num_workers
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp      = torch.cuda.is_available()

    # --- H5 erzeugen (einmalig) ---
    build_h5_dataset(
            img_dir, mask_dir, out_path=h5_path,
            class_values=class_values,
            patch_size=patch_size, overlap=overlap, resize_half=resize_half,
            val_split=val_split, seed=seed, compression="lzf")

    # --- Dataset / Loader ---
    train_ds = H5SegDataset(h5_path, split=0, class_values=class_values)
    val_ds   = H5SegDataset(h5_path, split=1, class_values=class_values)


    #Variante B – schneller (nachdem es läuft, gern aufdrehen)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                               num_workers=nw, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,
                               num_workers=nw, pin_memory=True,
                               persistent_workers=True)

    # --- Modell/Optim/Scheduler ---
    model = UNet(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    checkpoint = "best_unet.pth"
    best_iou = 0.0

    # --- Class Weights (aus H5, nur Train-Split) ---
    class_weights = compute_class_weights_from_h5(h5_path, split=0, n_classes=n_classes, class_values=class_values).to(device)
    print("Class weights:", class_weights.tolist())

    # --- Optional: vorhandenes Modell laden ---
    if os.path.exists(checkpoint):
        print(f"Loading saved model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

    # --- Training ---
    scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu", enabled=amp)
    losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Train Ep {epoch}"):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(imgs)
                loss = weighted_cross_entropy_loss(logits, masks, class_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        losses.append(train_loss)

        # ------ Validation ------
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=amp):
            for imgs, masks in tqdm(val_loader, desc="Val"):
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                logits = model(imgs)
                preds = F.softmax(logits, dim=1)

                val_loss += F.cross_entropy(logits, masks, weight=class_weights).item()
                masks_one_hot = F.one_hot(masks, num_classes=n_classes).permute(0, 3, 1, 2).float()
                val_dice += dice_coef(preds, masks_one_hot).item()

        val_loss /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        with open("log_filip.txt","+a") as f:
            f.write(f"Epoch {epoch}  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}\n")
            f.close()
        if val_dice > best_iou:
            best_iou = val_dice
            torch.save(model.state_dict(), checkpoint)

# %%
