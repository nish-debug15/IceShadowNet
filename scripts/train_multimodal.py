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
        "cm":        confusion_matrix(labels, preds, labels=[0, 1]),
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
    SEEDS = [42, 123, 2024]
    
    train_ds = load_mm_split("train")
    val_ds   = load_mm_split("val")
    test_ds  = load_mm_split("test")

    # Support checks
    val_pos = sum(train_ds.labels) if hasattr(val_ds, 'labels') else sum(val_ds.labels) 
    # Wait, in Multimodal dataset, labels are precomputed array. `sum(val_ds.labels)`
    val_pos = sum(val_ds.labels)
    test_pos = sum(test_ds.labels)
    print(f"\nVal Positives: {val_pos}, Test Positives: {test_pos}")
    
    low_support_flag = ""
    if val_pos < 10 or test_pos < 10:
        low_support_flag = (
            f"\n[WARNING] LOW SUPPORT -- METRICS ON THIS SPLIT ARE NOISY\n"
            f"   Val Positives: {val_pos}, Test Positives: {test_pos}\n"
        )
        print(low_support_flag)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight_val = read_pos_weight()
    criterion      = select_loss(pos_weight_val)

    def get_models():
        return [
            ("EarlyFusionCNN",   lambda: EarlyFusionCNN(dfsar_channels=2, ohrc_channels=1)),
            ("LateFusionResNet", lambda: LateFusionResNet(dfsar_channels=2, ohrc_channels=1, pretrained=False)),
        ]

    summary_rows = []
    circularity_flags = []
    if low_support_flag:
        circularity_flags.append(low_support_flag)

    for name, factory in get_models():
        seed_results = []
        for seed in SEEDS:
            print(f"\n=== Training {name} (Seed {seed}) ===")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = factory().to(DEVICE)
            
            best_f1, tl, vl, tt, params = train_mm_model(
                model, train_loader, val_loader, criterion, name
            )
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}.pth"), map_location=DEVICE))
            test_m = evaluate_mm(model, test_loader)
            test_m["train_time"] = tt
            test_m["val_f1"] = best_f1
            seed_results.append(test_m)

        # Aggregate across seeds
        avg_acc = np.mean([r["accuracy"] for r in seed_results])
        std_acc = np.std([r["accuracy"] for r in seed_results])
        avg_prec = np.mean([r["precision"] for r in seed_results])
        std_prec = np.std([r["precision"] for r in seed_results])
        avg_rec = np.mean([r["recall"] for r in seed_results])
        std_rec = np.std([r["recall"] for r in seed_results])
        avg_f1 = np.mean([r["f1"] for r in seed_results])
        std_f1 = np.std([r["f1"] for r in seed_results])
        avg_auc = np.mean([r["roc_auc"] for r in seed_results])
        std_auc = np.std([r["roc_auc"] for r in seed_results])
        avg_val_f1 = np.mean([r["val_f1"] for r in seed_results])
        std_val_f1 = np.std([r["val_f1"] for r in seed_results])
        
        cm = seed_results[-1]["cm"]
        tn, fp, fn, tp = cm.ravel()
        support_str = f"TP:{tp} FP:{fp} FN:{fn} TN:{tn} (Pos:{tp+fn} Neg:{tn+fp})"
        
        flag = flag_suspicious_metrics(seed_results[-1], name)
        if flag:
            circularity_flags.append(flag)

        row = {
            "model":      name,
            "accuracy":    f"{avg_acc:.4f}±{std_acc:.4f}",
            "precision":   f"{avg_prec:.4f}±{std_prec:.4f}",
            "recall":      f"{avg_rec:.4f}±{std_rec:.4f}",
            "f1":          f"{avg_f1:.4f}±{std_f1:.4f}",
            "roc_auc":     f"{avg_auc:.4f}±{std_auc:.4f}",
            "val_f1":      f"{avg_val_f1:.4f}±{std_val_f1:.4f}",
            "support":     support_str,
            "params":     params,
            "train_time": round(np.mean([r["train_time"] for r in seed_results]), 1),
        }
        summary_rows.append(row)
        print(f"\n  Test metrics (Mean±Std) -- {name}:")
        for k, v in row.items():
            if k not in ("model", "params", "train_time", "val_f1"):
                print(f"    {k:12s}: {v}")

    md = (
        "# Multimodal Training Summary\n\n"
        "> **DATA SOURCE:** SYNTHETIC. See reports/data_notes.md.\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |\n"
        "|-------|----------|-----------|--------|----|---------|---------|--------|----------|\n"
    )
    for r in summary_rows:
        md += (
            f"| {r['model']} | {r['accuracy']} | {r['precision']} | "
            f"{r['recall']} | {r['f1']} | {r['roc_auc']} | {r['support']} | "
            f"{r['params']:,} | {r['train_time']} |\n"
        )
    if circularity_flags:
        md += "\n## [WARNING] Circularity / Low Support Flags\n\n"
        for flag in list(set(circularity_flags)):
            md += flag + "\n"

    with open(os.path.join(REPORTS_DIR, "training_summary_multimodal.md"), "w") as f:
        f.write(md)

    with open(os.path.join(REPORTS_DIR, "training_summary_multimodal.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nMultimodal results saved to {REPORTS_DIR}/")
    print("Block 3 training complete.")
    return summary_rows

if __name__ == "__main__":
    main()
