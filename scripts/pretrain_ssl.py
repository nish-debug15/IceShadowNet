"""
scripts/pretrain_ssl.py
=======================
Self-Supervised Contrastive Pretraining (SimCLR) for Lunar SAR.

Novel Contribution for Viva:
- Replaces mismatched optical ImageNet pretraining with in-domain radar contrastive pretraining.
- Pretrains ResNet-18 on unlabeled SAR patches from TRAIN + VAL splits ONLY.
- Hard assertion: strictly ZERO test patches used (no transductive data leakage).
- SAR-specific augmentations: multiplicative speckle noise, gain/offset jitter,
  random resized crop, flips, and 90-degree rotations. NO optical color jitter.

Outputs:
- models_real/resnet18_ssl_pretrained.pth
- reports/real/figures/ssl_pretrain_loss.png
"""

import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.simclr_ssl import SARSimCLR, NTXentLoss

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED_REAL = "data/processed_real"
MODELS_DIR     = "models_real"
FIGURES_DIR    = "reports/real/figures"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 35
LR         = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.2
SEED       = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# SAR-Specific Augmentation Pair Generator
# ──────────────────────────────────────────────────────────────────────────────
def sar_augment(img: np.ndarray) -> np.ndarray:
    """
    Apply physics-grounded radar augmentations to (3, H, W) patch.
    1. Random horizontal flip (p=0.5)
    2. Random vertical flip (p=0.5)
    3. Random 90-degree rotation (k in {0, 1, 2, 3})
    4. Random resized crop (scale 0.75-1.0, crop and resize back to 64x64)
    5. Multiplicative speckle noise (log-domain additive noise)
    6. Gain / log-offset jitter
    Strictly NO RGB color jitter!
    """
    x = img.copy()  # (3, H, W)
    C, H, W = x.shape

    # 1. Flips
    if random.random() < 0.5:
        x = np.flip(x, axis=2).copy()
    if random.random() < 0.5:
        x = np.flip(x, axis=1).copy()

    # 2. Rotations (90, 180, 270)
    k = random.randint(0, 3)
    if k > 0:
        x = np.rot90(x, k, axes=(1, 2)).copy()

    # 3. Random resized crop (scale 0.75 to 1.0)
    scale = random.uniform(0.75, 1.0)
    crop_h = max(16, int(H * scale))
    crop_w = max(16, int(W * scale))
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    cropped = x[:, top:top+crop_h, left:left+crop_w]

    # Bilinear interpolation back to (H, W) using torch
    crop_t = torch.from_numpy(cropped).unsqueeze(0).float()
    resized_t = F.interpolate(crop_t, size=(H, W), mode="bilinear", align_corners=False)
    x = resized_t.squeeze(0).numpy()

    # 4. Multiplicative speckle noise simulation
    # In log10 domain: log10(I * (1 + eta)) = log10(I) + log10(1 + eta)
    if random.random() < 0.8:
        speckle_std = random.uniform(0.05, 0.15)
        eta = np.random.normal(0, speckle_std, size=x.shape).astype(np.float32)
        # Ensure 1 + eta > 0
        eta = np.clip(eta, -0.8, 0.8)
        speckle_log = np.log10(1.0 + eta + 1e-6)
        x = x + speckle_log

    # 5. Gain / calibration offset jitter
    if random.random() < 0.8:
        offset = random.uniform(-0.15, 0.15)
        x = x + offset

    # Per-channel standard normalization
    for c in range(C):
        mu = x[c].mean()
        std = x[c].std()
        x[c] = (x[c] - mu) / (std + 1e-8)

    return x.astype(np.float32)


class SimCLRDataset(Dataset):
    """
    Dataset returning two independently augmented views of each unlabeled patch.
    """
    def __init__(self, images: np.ndarray):
        self.images = images  # (N, 3, 64, 64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_patch = self.images[idx]
        v1 = sar_augment(raw_patch)
        v2 = sar_augment(raw_patch)
        return torch.from_numpy(v1), torch.from_numpy(v2)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pretraining Loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("SELF-SUPERVISED CONTRASTIVE PRETRAINING (SimCLR for Lunar SAR)")
    print("=" * 70)

    # 1. Load Train and Val splits ONLY (strictly no test split)
    train_path = os.path.join(PROCESSED_REAL, "train.npz")
    val_path   = os.path.join(PROCESSED_REAL, "val.npz")
    test_path  = os.path.join(PROCESSED_REAL, "test.npz")

    train_data = np.load(train_path)
    val_data   = np.load(val_path)
    test_data  = np.load(test_path)

    # Hard assertion against transductive data leakage
    train_coords = set(map(tuple, train_data["coords"]))
    val_coords   = set(map(tuple, val_data["coords"]))
    test_coords  = set(map(tuple, test_data["coords"]))

    assert len(train_coords.intersection(test_coords)) == 0, "LEAKAGE: Train coords overlap Test coords!"
    assert len(val_coords.intersection(test_coords)) == 0, "LEAKAGE: Val coords overlap Test coords!"
    print(f"Hard Assertion Passed: Zero test coordinates in pretraining set.")

    # Combine train + val images for self-supervised pretraining (labels discarded)
    unlabeled_images = np.concatenate([train_data["images"], val_data["images"]], axis=0)
    print(f"Pretraining Dataset: {len(unlabeled_images):,d} unlabeled radar patches (Train: {len(train_data['images'])}, Val: {len(val_data['images'])})")
    print(f"Held-out Test Split: {len(test_data['images'])} patches strictly isolated.")

    ds = SimCLRDataset(unlabeled_images)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 2. Model & Loss setup
    model = SARSimCLR(in_channels=3, proj_dim=128, hidden_dim=512).to(DEVICE)
    criterion = NTXentLoss(temperature=TEMPERATURE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    print(f"\nBeginning SimCLR Contrastive Pretraining on {DEVICE} for {EPOCHS} epochs (Batch Size={BATCH_SIZE})...")

    epoch_losses = []
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x1, x2 in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                z1 = model(x1)
                z2 = model(x2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  Epoch [{epoch:2d}/{EPOCHS:2d}] | NT-Xent Contrastive Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # 3. Save pretrained encoder weights
    # Extract only the backbone weights so it can directly initialize IceResNet
    checkpoint_path = os.path.join(MODELS_DIR, "resnet18_ssl_pretrained.pth")
    # Save both the full SimCLR model and the backbone alone for convenient loading
    torch.save({
        "backbone_state_dict": model.backbone.state_dict(),
        "full_state_dict": model.state_dict(),
        "epoch_losses": epoch_losses,
        "best_loss": best_loss,
        "epochs": EPOCHS,
    }, checkpoint_path)
    print(f"\nPretraining Complete! Saved encoder checkpoint to: {checkpoint_path}")

    # 4. Save Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker="o", color="#2563eb", linewidth=2, label="NT-Xent Loss")
    plt.xlabel("Pretraining Epoch", fontsize=11)
    plt.ylabel("Contrastive Loss", fontsize=11)
    plt.title("SimCLR Contrastive Pretraining Loss on Lunar DFSAR Patches", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    fig_path = os.path.join(FIGURES_DIR, "ssl_pretrain_loss.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pretraining loss curve to: {fig_path}")
    print(f"Total Pretraining Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
