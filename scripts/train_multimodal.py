"""
scripts/train_multimodal.py
==============================
Block 3 training: EarlyFusionCNN + LateFusionResNet on DFSAR+OHRC data.
Produces reports/training_summary_multimodal.md and model checkpoints.

Usage:
    python -m scripts.train_multimodal
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.datasets.multimodal_dataset import MultimodalIceDataset
from src.models.multimodal_fusion import EarlyFusionCNN, LateFusionResNet
from scripts.train_all import (
    select_loss, read_pos_weight, save_loss_curve,
    save_confusion_matrix, save_roc_curve, flag_suspicious_metrics,
    BATCH_SIZE, EPOCHS, LR, SEED, DEVICE, MODELS_DIR, REPORTS_DIR,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = "data/processed"


def load_mm_split(name):
    path = os.path.join(PROCESSED_DIR, f"{name}_multimodal.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Multimodal split '{path}' not found. "
            "Run `python -m scripts.ingest_multimodal` first."
        )
    return MultimodalIceDataset(path, augment=(name == "train"), normalize=True)


def evaluate_mm(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for d, o, y in loader:
            d, o = d.to(DEVICE), o.to(DEVICE)
            logits = model(d, o).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    m = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "cm":        confusion_matrix(labels, preds),
        "probs":     probs,
        "labels":    labels,
    }
    try:
        m["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


def train_mm_model(model, train_loader, val_loader, criterion, name):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Training (multimodal): {name}  ({params:,} params)")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    train_losses, val_losses = [], []
    best_f1 = -1.0
    best_ckpt = os.path.join(MODELS_DIR, f"{name}.pth")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for d, o, y in train_loader:
            d, o, y = d.to(DEVICE), o.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(d, o), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * d.size(0)
        avg_train = epoch_loss / max(len(train_loader.dataset), 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for d, o, y in val_loader:
                d, o, y = d.to(DEVICE), o.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(d, o), y).item() * d.size(0)
        avg_val = val_loss / max(len(val_loader.dataset), 1)
        val_losses.append(avg_val)

        val_m = evaluate_mm(model, val_loader)
        scheduler.step(val_m["f1"])
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), best_ckpt)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:3d}/{EPOCHS} | Train {avg_train:.4f} | "
                  f"Val {avg_val:.4f} | F1 {val_m['f1']:.3f}")

    return best_f1, train_losses, val_losses, time.time()-t0, params


def main():
    train_ds = load_mm_split("train")
    val_ds   = load_mm_split("val")
    test_ds  = load_mm_split("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight_val = read_pos_weight()
    criterion      = select_loss(pos_weight_val)

    model_defs = [
        ("EarlyFusionCNN",   EarlyFusionCNN(dfsar_channels=2, ohrc_channels=1)),
        ("LateFusionResNet", LateFusionResNet(dfsar_channels=2, ohrc_channels=1, pretrained=False)),
    ]

    summary_rows = []
    for name, model in model_defs:
        model = model.to(DEVICE)
        best_f1, tl, vl, tt, params = train_mm_model(
            model, train_loader, val_loader, criterion, name
        )
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}.pth"), map_location=DEVICE))
        test_m = evaluate_mm(model, test_loader)

        flag_suspicious_metrics(test_m, name)
        save_loss_curve(tl, vl, name)
        save_confusion_matrix(test_m["cm"], name)
        save_roc_curve(test_m["labels"], test_m["probs"], name)

        row = {
            "model":      name,
            "accuracy":   test_m["accuracy"],
            "precision":  test_m["precision"],
            "recall":     test_m["recall"],
            "f1":         test_m["f1"],
            "roc_auc":    test_m["roc_auc"],
            "params":     params,
            "train_time": round(tt, 1),
        }
        summary_rows.append(row)
        print(f"\n  Test metrics — {name}:")
        for k, v in row.items():
            if k not in ("model", "params", "train_time"):
                print(f"    {k:12s}: {v:.4f}")

    md = (
        "# Multimodal Training Summary\n\n"
        "> **DATA SOURCE:** SYNTHETIC. See reports/data_notes.md.\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Time (s) |\n"
        "|-------|----------|-----------|--------|----|---------|--------|----------|\n"
    )
    for r in summary_rows:
        md += (
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['params']:,} | {r['train_time']} |\n"
        )
    with open(os.path.join(REPORTS_DIR, "training_summary_multimodal.md"), "w") as f:
        f.write(md)

    with open(os.path.join(REPORTS_DIR, "training_summary_multimodal.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nMultimodal results saved to {REPORTS_DIR}/")
    print("Block 3 training complete.")
    return summary_rows


if __name__ == "__main__":
    main()
