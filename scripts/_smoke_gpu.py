"""Smoke test: 1 epoch, seed 42, CustomCNN on GPU."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert DEVICE.type == "cuda", f"CUDA not available: {DEVICE}"
    print(f"Device : {DEVICE} ({torch.cuda.get_device_name(0)})")
    torch.cuda.reset_peak_memory_stats()

    from src.datasets.ice_dataset import IcePatchDataset
    from src.models.custom_cnn import CustomCNN

    PROCESSED_DIR = "data/processed_real"
    BATCH_SIZE = 16
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    X, Y, C = data["images"], data["labels"], data["coords"]
    patches = list(zip(X, Y.tolist(), [tuple(c) for c in C]))

    train_ds = IcePatchDataset(patches, augment=True, normalize=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    in_channels = train_ds.patches[0][0].shape[0]
    model = CustomCNN(in_channels=in_channels, num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    model.train()
    running_loss = 0.0
    n = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        n += x.size(0)

    elapsed = time.time() - t0
    avg_loss = running_loss / n
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Loss   : {avg_loss:.4f}")
    print(f"Time   : {elapsed:.1f}s / epoch")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")
    print("SMOKE TEST PASSED")
