"""
scripts/run_gradcam.py
========================
Block 3: generate Grad-CAM overlays on REAL trained models using real
(or synthetic-but-real-pipeline) patches from data/processed/.

Saves overlays to reports/gradcam/<model_name>_sample<i>.png

Usage:
    python -m scripts.run_gradcam
"""

import os, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _HAS_GRADCAM_PKG = True
except ImportError:
    _HAS_GRADCAM_PKG = False
    warnings.warn(
        "pytorch-grad-cam not installed (pip install grad-cam). "
        "Falling back to manual Grad-CAM implementation.",
        RuntimeWarning,
    )

PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
GRADCAM_DIR    = "reports/gradcam"
DEVICE         = torch.device("cpu")   # Grad-CAM works fine on CPU for inference
N_SAMPLES      = 5                     # overlays per model

os.makedirs(GRADCAM_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Manual Grad-CAM (no external package dependency)
# ──────────────────────────────────────────────────────────────────────────────
class ManualGradCAM:
    """
    Hook-based Grad-CAM without the pytorch-grad-cam package.

    Algorithm (Selvaraju et al. 2017):
    1. Forward pass; capture feature maps A^k at target conv layer.
    2. Backward pass; capture gradients ∂y/∂A^k at same layer.
    3. Weight each feature map channel by global-average-pooled gradient.
    4. ReLU + normalise → heatmap H (H, W).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model         = model
        self.target_layer  = target_layer
        self._features     = None
        self._gradients    = None
        self._fwd_hook     = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook     = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, inp, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return normalised Grad-CAM heatmap (H_feat, W_feat) in [0,1]."""
        self.model.zero_grad()
        self.model.eval()
        out = self.model(input_tensor)             # (1,1)
        out.backward()                              # trigger hooks

        weights   = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam       = (weights * self._features).sum(dim=1).squeeze()  # (H_f, W_f)
        cam       = torch.relu(cam).cpu().numpy()
        cam_range = cam.max() - cam.min()
        if cam_range > 1e-8:
            cam = (cam - cam.min()) / cam_range
        else:
            cam = np.zeros_like(cam)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_patches(split="test"):
    path = os.path.join(PROCESSED_DIR, f"{split}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' not found. Run ingest_and_split first.")
    data   = np.load(path)
    images = data["images"]   # (N, C, H, W)
    labels = data["labels"]   # (N,)
    return images, labels


def load_model(model_class, model_path, device):
    model = model_class
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint '{model_path}' not found. "
            "Run train_all.py (Block 2) before run_gradcam.py (Block 3)."
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def make_vis_image(img_chw: np.ndarray) -> np.ndarray:
    """Convert (C,H,W) float patch to (H,W,3) uint8 for matplotlib / show_cam_on_image."""
    ch0 = img_chw[0]
    # Normalise each channel for display
    lo, hi = ch0.min(), ch0.max()
    vis = (ch0 - lo) / (hi - lo + 1e-8)
    return np.stack([vis, vis, vis], axis=-1).astype(np.float32)


def run_gradcam_on_model(model, target_layer, images, labels, name, n_samples=N_SAMPLES):
    """Generate and save Grad-CAM overlays for a set of sample patches."""
    # Pick n_samples ice patches first (label==1), then non-ice if needed
    ice_idx   = np.where(labels == 1)[0][:n_samples]
    noice_idx = np.where(labels == 0)[0][:max(0, n_samples - len(ice_idx))]
    sample_idx = list(ice_idx) + list(noice_idx)

    if _HAS_GRADCAM_PKG:
        cam_obj = GradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(0)]

    for i, idx in enumerate(sample_idx):
        inp        = torch.from_numpy(images[idx:idx+1]).to(DEVICE).requires_grad_(True)
        vis_img    = make_vis_image(images[idx])     # (H,W,3) float32 [0,1]
        pred_label = labels[idx]

        if _HAS_GRADCAM_PKG:
            heatmap = cam_obj(input_tensor=inp, targets=targets)[0]   # (H_f, W_f)
        else:
            gc   = ManualGradCAM(model, target_layer)
            heatmap = gc.generate(inp)
            gc.remove_hooks()

        # Resize heatmap to patch spatial size for overlay
        import cv2
        H, W = vis_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H))

        # Colour overlay
        heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3].astype(np.float32)
        overlay = 0.5 * vis_img + 0.5 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(vis_img);     axes[0].set_title("Original (Ch0 norm)")
        axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay);     axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        truth_str = "ICE" if pred_label == 1 else "NON-ICE"
        fig.suptitle(f"{name} | patch #{idx} | label={truth_str}", fontsize=11)

        save_path = os.path.join(GRADCAM_DIR, f"{name}_sample{i}_label{pred_label}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    images, labels = load_patches("test")
    print(f"Loaded test split: {images.shape}, {labels.shape}")
    in_channels = images.shape[1]

    # 1. CustomCNN
    cnn = load_model(
        CustomCNN(in_channels=in_channels, num_classes=1),
        os.path.join(MODELS_DIR, "CustomCNN.pth"),
        DEVICE,
    )
    print("\nGrad-CAM: CustomCNN (target=conv3)")
    run_gradcam_on_model(cnn, cnn.conv3, images, labels, "CustomCNN")

    # 2. ResNet (scratch)
    rn_scratch = load_model(
        IceResNet(in_channels=in_channels, num_classes=1, pretrained=False),
        os.path.join(MODELS_DIR, "ResNet_scratch.pth"),
        DEVICE,
    )
    print("\nGrad-CAM: ResNet_scratch (target=layer4[-1])")
    run_gradcam_on_model(rn_scratch, rn_scratch.model.layer4[-1], images, labels, "ResNet_scratch")

    # 3. ResNet (pretrained)
    rn_pt = load_model(
        IceResNet(in_channels=in_channels, num_classes=1, pretrained=True),
        os.path.join(MODELS_DIR, "ResNet_pretrained.pth"),
        DEVICE,
    )
    print("\nGrad-CAM: ResNet_pretrained (target=layer4[-1])")
    run_gradcam_on_model(rn_pt, rn_pt.model.layer4[-1], images, labels, "ResNet_pretrained")

    print(f"\nAll Grad-CAM overlays saved to {GRADCAM_DIR}/")
    print("Block 3 Grad-CAM complete.")


if __name__ == "__main__":
    main()
