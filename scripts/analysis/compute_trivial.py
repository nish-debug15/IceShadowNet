import os, sys, numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

PROCESSED_DIR = "data/processed_real"
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
Y = test_data["labels"]
ice_fracs = test_data["ice_fracs"]

# Full test
n_total = len(Y)
n_pos = int(np.sum(Y == 1))
n_neg = int(np.sum(Y == 0))
prev_full = n_pos / n_total

# All-positive classifier on full test:
# TP = n_pos, FP = n_neg, FN = 0, TN = 0
tp_full = n_pos
fp_full = n_neg
fn_full = 0
prec_full = tp_full / (tp_full + fp_full)
rec_full = 1.0
f1_full = 2 * prec_full * rec_full / (prec_full + rec_full)

# Clean test
pos_px = np.round(ice_fracs * (64 * 64)).astype(int)
amb_mask = (pos_px >= 1) & (pos_px <= 4)
clean_mask = ~amb_mask

Y_clean = Y[clean_mask]
n_clean_total = len(Y_clean)
n_clean_pos = int(np.sum(Y_clean == 1))
n_clean_neg = int(np.sum(Y_clean == 0))
prev_clean = n_clean_pos / n_clean_total

tp_clean = n_clean_pos
fp_clean = n_clean_neg
fn_clean = 0
prec_clean = tp_clean / (tp_clean + fp_clean)
rec_clean = 1.0
f1_clean = 2 * prec_clean * rec_clean / (prec_clean + rec_clean)

# Ambiguity breakdown
neg_indices = np.where(Y == 0)[0]
pos_indices = np.where(Y == 1)[0]
amb_negs = int(np.sum(amb_mask[neg_indices]))
amb_pos = int(np.sum(amb_mask[pos_indices]))

print(f"Full Test Set (N={n_total}):")
print(f"  Positives: {n_pos}, Negatives: {n_neg}, Prevalence: {prev_full*100:.2f}%")
print(f"  All-Positive Classifier: Precision = {prec_full:.4f}, Recall = {rec_full:.4f}, F1 = {f1_full:.4f}")
print(f"  Random Score Classifier: ROC-AUC = 0.5000, PR-AUC = {prev_full:.4f}")

print(f"\nAmbiguity Band (1-4 positive pixels):")
print(f"  Total ambiguous patches: {int(amb_mask.sum())}")
print(f"  Ambiguous negatives: {amb_negs} / {n_neg} ({100*amb_negs/n_neg:.1f}%)")
print(f"  Ambiguous positives: {amb_pos} / {n_pos} ({100*amb_pos/n_pos:.1f}%)")

print(f"\nClean Test Set (N={n_clean_total}):")
print(f"  Positives: {n_clean_pos}, Negatives: {n_clean_neg}, Prevalence: {prev_clean*100:.2f}%")
print(f"  All-Positive Classifier: Precision = {prec_clean:.4f}, Recall = {rec_clean:.4f}, F1 = {f1_clean:.4f}")
print(f"  Random Score Classifier: ROC-AUC = 0.5000, PR-AUC = {prev_clean:.4f}")
