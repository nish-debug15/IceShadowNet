"""
src/evaluate.py
================
Standardized evaluation metrics, threshold tuning, and block-bootstrap CI calculation
for IceShadowNet lunar polar SAR classification.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay,
)


def tune_threshold_on_val(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    threshold_range: tuple = (0.05, 0.95),
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Tune binary classification threshold strictly on validation set by maximizing F1 score.
    Test labels and test predictions are NEVER accessed or passed to this function.

    Args:
        val_labels: 1D binary array of ground-truth validation labels.
        val_probs: 1D float array of predicted probabilities on validation set.
        threshold_range: (min_threshold, max_threshold) search range.
        step: Grid search step size.

    Returns:
        (best_threshold, best_val_f1): Optimal threshold and corresponding validation F1.
    """
    candidates = np.round(np.arange(threshold_range[0], threshold_range[1] + 1e-5, step), 4)
    best_f1 = -1.0
    best_thresh = 0.5

    for th in candidates:
        preds = (val_probs >= th).astype(int)
        score = f1_score(val_labels, preds, zero_division=0)
        # Prefer higher score; tie-break towards threshold closest to 0.5
        if score > best_f1 or (np.isclose(score, best_f1) and abs(th - 0.5) < abs(best_thresh - 0.5)):
            best_f1 = float(score)
            best_thresh = float(th)

    return best_thresh, best_f1


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive classification metrics including PR-AUC and test prevalence.

    Args:
        y_true: 1D binary array of true labels.
        y_probs: 1D float array of predicted probabilities.
        threshold: Decision threshold for binary prediction.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, pr_auc,
        prevalence, threshold, cm, and support.
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_probs = np.asarray(y_probs).flatten().astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = float(np.mean(y_true))

    prevalence = float(np.mean(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_str = f"TP:{tp} FP:{fp} FN:{fn} TN:{tn} (Pos:{tp+fn} Neg:{tn+fp})"

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "prevalence": float(prevalence),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "support": support_str,
    }


def block_bootstrap_evaluation(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    strip_ids: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Perform block-bootstrap evaluation by resampling entire spatial strips with replacement.
    Accounts for spatial correlation among patches within the same along-track strip.

    Args:
        y_true: 1D array of ground truth binary labels.
        y_probs: 1D array of predicted probabilities.
        strip_ids: 1D array indicating the along-track strip ID for each patch.
        threshold: Classification decision threshold (e.g. tuned on val).
        n_bootstraps: Number of bootstrap iterations (default: 1000).
        confidence_level: Confidence level for intervals (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing point estimates and 95% CIs for F1, ROC-AUC, and PR-AUC.
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_probs = np.asarray(y_probs).flatten().astype(float)
    strip_ids = np.asarray(strip_ids).flatten()

    unique_strips = np.unique(strip_ids)
    strip_map = {s: np.where(strip_ids == s)[0] for s in unique_strips}

    rng = np.random.default_rng(seed)

    f1_list = []
    roc_auc_list = []
    pr_auc_list = []

    for _ in range(n_bootstraps):
        sampled_strips = rng.choice(unique_strips, size=len(unique_strips), replace=True)
        sampled_indices = np.concatenate([strip_map[s] for s in sampled_strips])

        y_boot = y_true[sampled_indices]
        p_boot = y_probs[sampled_indices]

        # Require both classes to evaluate ranking metrics
        if len(np.unique(y_boot)) < 2:
            continue

        pred_boot = (p_boot >= threshold).astype(int)
        f1_list.append(f1_score(y_boot, pred_boot, zero_division=0))
        roc_auc_list.append(roc_auc_score(y_boot, p_boot))
        pr_auc_list.append(average_precision_score(y_boot, p_boot))

    alpha = (1.0 - confidence_level) / 2.0
    low_pct = 100.0 * alpha
    high_pct = 100.0 * (1.0 - alpha)

    if len(f1_list) > 0:
        f1_ci = (float(np.percentile(f1_list, low_pct)), float(np.percentile(f1_list, high_pct)))
        roc_auc_ci = (float(np.percentile(roc_auc_list, low_pct)), float(np.percentile(roc_auc_list, high_pct)))
        pr_auc_ci = (float(np.percentile(pr_auc_list, low_pct)), float(np.percentile(pr_auc_list, high_pct)))
        f1_mean = float(np.mean(f1_list))
        roc_auc_mean = float(np.mean(roc_auc_list))
        pr_auc_mean = float(np.mean(pr_auc_list))
    else:
        # Fallback in extreme edge case
        point = compute_comprehensive_metrics(y_true, y_probs, threshold=threshold)
        f1_ci = (point["f1"], point["f1"])
        roc_auc_ci = (point["roc_auc"], point["roc_auc"])
        pr_auc_ci = (point["pr_auc"], point["pr_auc"])
        f1_mean = point["f1"]
        roc_auc_mean = point["roc_auc"]
        pr_auc_mean = point["pr_auc"]

    return {
        "f1_ci": f1_ci,
        "roc_auc_ci": roc_auc_ci,
        "pr_auc_ci": pr_auc_ci,
        "f1_mean": f1_mean,
        "roc_auc_mean": roc_auc_mean,
        "pr_auc_mean": pr_auc_mean,
        "valid_rounds": len(f1_list),
        "total_rounds": n_bootstraps,
    }


def get_strip_ids_from_coords(coords: np.ndarray, grid_size: tuple = (1, 20)) -> np.ndarray:
    """
    Derive spatial along-track strip index from patch coordinates (r, c).
    Uses full real DFSAR swath length (40,000 lines, max r = 39,936).
    """
    coords = np.asarray(coords)
    block_width = (39936.0 + 1e-5) / grid_size[1]
    return np.minimum((coords[:, 0] / block_width).astype(int), grid_size[1] - 1)


def evaluate_model(model, dataloader, device, threshold: float = 0.5):
    """
    Evaluates a trained PyTorch model on a DataLoader.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs).astype(float)

    metrics = compute_comprehensive_metrics(all_labels, all_probs, threshold=threshold)
    metrics["probs"] = all_probs
    metrics["labels"] = all_labels
    return metrics


def save_confusion_matrix(cm, save_path):
    """Saves the confusion matrix as an image."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Ice", "Potential Ice"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
