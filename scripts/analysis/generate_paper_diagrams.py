"""
scripts/analysis/generate_paper_diagrams.py
===========================================
Generates three clean publication figures in reports/real/figures/ (all <500 KB):
  1. scene_strips_overview.png   — Plotted strictly from data/processed_real/split_manifest.json
  2. pipeline_overview.png       — End-to-end anti-circularity ML & georef pipeline diagram
  3. iceunet_architecture.png    — Exact 4-level IceUNet architecture matching src/models/unet.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "processed_real", "split_manifest.json")
OUT_DIR = os.path.join(REPO_ROOT, "reports", "real", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Warm Anthropic editorial palette
BG = "#FAF9F5"
SURF = "#F0EEE6"
INK = "#141413"
MUTED = "#5E5D59"
BORDER = "#D4D0C5"
ACCENT = "#D97757"
VAL_BLUE = "#3D6B8E"
TRAIN_SAGE = "#5A7D65"
AMBIG_GOLD = "#C99B3B"


def make_scene_strips_overview():
    """Plot exact per-strip patch breakdown (positive, ambiguity 1-4px, zero-px) and split assignment."""
    manifest = json.load(open(MANIFEST_PATH, "r", encoding="utf-8"))
    by_strip = {}
    for p in manifest["patches"]:
        s = p["strip_id"]
        by_strip.setdefault(s, {"split": p["split"], "pos": 0, "ambig": 0, "zero": 0, "total": 0})
        by_strip[s]["total"] += 1
        if p["label"] == 1:
            by_strip[s]["pos"] += 1
        elif 1 <= p["pos_px_count"] <= 4:
            by_strip[s]["ambig"] += 1
        else:
            by_strip[s]["zero"] += 1

    strips = sorted(by_strip.keys())
    pos_vals = [by_strip[s]["pos"] for s in strips]
    amb_vals = [by_strip[s]["ambig"] for s in strips]
    zero_vals = [by_strip[s]["zero"] for s in strips]
    rates = [100.0 * by_strip[s]["pos"] / by_strip[s]["total"] for s in strips]
    splits = [by_strip[s]["split"] for s in strips]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.4), dpi=150, facecolor=BG, gridspec_kw={"height_ratios": [1.5, 1.0]})
    fig.subplots_adjust(hspace=0.35)

    x = np.arange(len(strips))
    w = 0.72
    ax1.set_facecolor(BG)
    ax1.bar(x, pos_vals, width=w, color=ACCENT, label="Positive (>=5 px CPR>1 & DOP<0.13)")
    ax1.bar(x, amb_vals, bottom=pos_vals, width=w, color=AMBIG_GOLD, alpha=0.85, label="Ambiguity Band (1-4 pos px, y=0)")
    bot2 = [a + b for a, b in zip(pos_vals, amb_vals)]
    ax1.bar(x, zero_vals, bottom=bot2, width=w, color="#D8D4C9", label="Clean Negative (0 pos px, y=0)")

    for i, sp in enumerate(splits):
        if sp == "test":
            ax1.axvspan(i - 0.45, i + 0.45, color=ACCENT, alpha=0.09, zorder=0)
            ax1.text(i, by_strip[strips[i]]["total"] + 18, "TEST", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=ACCENT)
        elif sp == "val":
            ax1.axvspan(i - 0.45, i + 0.45, color=VAL_BLUE, alpha=0.09, zorder=0)
            ax1.text(i, by_strip[strips[i]]["total"] + 18, "VAL", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=VAL_BLUE)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in strips], fontsize=8.5, color=INK)
    ax1.set_ylim(0, 960)
    ax1.set_ylabel("Post-Buffer Patches", fontsize=9.5, color=INK)
    ax1.set_title("DFSAR Scene Strips (0-19) Across Active Rows [90624, 319812] with 256-Row Guard Buffers", fontsize=11, fontweight="bold", color=INK, pad=10)
    ax1.legend(loc="upper center", ncol=3, frameon=True, facecolor=SURF, edgecolor=BORDER, fontsize=7.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.5, color=MUTED)

    ax2.set_facecolor(BG)
    bar_colors = [ACCENT if sp == "test" else (VAL_BLUE if sp == "val" else TRAIN_SAGE) for sp in splits]
    ax2.bar(x, rates, width=w, color=bar_colors, alpha=0.9)
    train_rate = 100.0 * manifest["counts"]["train"]["positives"] / manifest["counts"]["train"]["patches"]
    ax2.axhline(train_rate, color=INK, linestyle="--", linewidth=1.1, label=f"Train positive rate ({train_rate:.2f}%)")
    ax2.axhspan(0.7 * train_rate, 1.4 * train_rate, color=TRAIN_SAGE, alpha=0.12, label="Deterministic split selection band (0.7x-1.4x)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{s}" for s in strips], fontsize=8.5, color=INK)
    ax2.set_ylim(0, 74)
    ax2.set_ylabel("Positive Rate (%)", fontsize=9.5, color=INK)
    ax2.set_xlabel("Along-Track Strip Index (Azimuth Order)", fontsize=9.5, color=INK)
    ax2.legend(loc="upper left", ncol=2, frameon=True, facecolor=SURF, edgecolor=BORDER, fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle=":", alpha=0.5, color=MUTED)

    out_path = os.path.join(OUT_DIR, "scene_strips_overview.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def draw_box(ax, x, y, w, h, title, subtitle="", fc=SURF, ec=BORDER, title_color=INK, lw=1.4):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=9, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=7.6, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=title_color)


def draw_arrow(ax, x0, y0, x1, y1, color=INK, lw=1.3, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def make_pipeline_overview():
    """Draw the end-to-end anti-circularity pipeline overview schematic."""
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(5.5, 4.92, "IceShadowNet Pipeline Overview (Anti-Circularity Split & Dual-Level Evaluation)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

    # Column 1: Raw Scene
    draw_box(ax, 0.20, 1.85, 2.15, 1.55, "Chandrayaan-2 DFSAR\nL1A Complex SLI", "Rows [90624, 319812]\nShh, Shv, Svv (0.505x9.593 m)", fc="#EDEAE0", ec="#8C8982")

    # Branch A (Labels) & Branch B (Inputs)
    draw_box(ax, 2.75, 3.35, 2.55, 1.15, "Proxy Label Generator", "7x7 Boxcar Stokes Matrix\nCPR > 1.0 & DOP < 0.13 (>=5 px)", fc="#FDF1EC", ec=ACCENT, title_color=ACCENT)
    draw_box(ax, 2.75, 0.70, 2.55, 1.15, "Strict Model Inputs (3-ch)", "Log-Intensity [HH, HV, VV]\nPer-patch z-scored (NO CPR/DOP)", fc="#EDF3F8", ec=VAL_BLUE, title_color=VAL_BLUE)

    draw_arrow(ax, 2.35, 2.9, 2.75, 3.85, color=ACCENT)
    draw_arrow(ax, 2.35, 2.35, 2.75, 1.30, color=VAL_BLUE)
    ax.text(4.02, 2.60, "Strict Anti-Circularity Firewall\n(Zero Derived Pol. Features in Inputs)", ha="center", va="center", fontsize=7.5, style="italic", color=ACCENT)

    # Column 3: Spatial Split
    draw_box(ax, 5.70, 1.75, 2.25, 1.75, "Deterministic Strip Split\n(256-Row Guard Buffer)", "Train: 14 strips (9,828)\nVal: [2,6,17] | Test: [0,8,15]\nZero extent leakage", fc=SURF, ec=BORDER)
    draw_arrow(ax, 5.30, 3.85, 5.70, 2.95)
    draw_arrow(ax, 5.30, 1.30, 5.70, 2.30)

    # Column 4: Models & Diagnostics
    draw_box(ax, 8.35, 3.45, 2.40, 1.05, "Patch Classification", "LogReg, CustomCNN, ResNet-18\nSimCLR SSL, IceUNet mean_prob", fc=SURF, ec=TRAIN_SAGE)
    draw_box(ax, 8.35, 2.05, 2.40, 1.05, "Pixel Localization", "IceUNet (3 seeds), 7x7+15x15\nHistGBDT/LR, 4-ch Phase Oracle*", fc=SURF, ec=ACCENT)
    draw_box(ax, 8.35, 0.65, 2.40, 1.05, "Reliability & GeoTIFFs", "MC Dropout / Deep Ensemble\nStructured Pruning & 676-GCP TPS", fc=SURF, ec=VAL_BLUE)

    draw_arrow(ax, 7.95, 2.95, 8.35, 3.95)
    draw_arrow(ax, 7.95, 2.62, 8.35, 2.57)
    draw_arrow(ax, 7.95, 2.25, 8.35, 1.15)

    out_path = os.path.join(OUT_DIR, "pipeline_overview.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def make_iceunet_architecture():
    """Draw exact 4-level IceUNet U-shaped encoder-decoder architecture from src/models/unet.py."""
    fig, ax = plt.subplots(figsize=(11.0, 5.3), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.35, 5.0)
    ax.axis("off")

    ax.text(5.5, 4.72, "IceUNet Architecture (src/models/unet.py · base_ch=24 · 1,084,681 Parameters)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

    # Input
    draw_box(ax, 0.25, 3.45, 1.45, 0.90, "Input Patch", "(B, 3, 64, 64)\n[HH, HV, VV]", fc="#EDF3F8", ec=VAL_BLUE)
    # Encoder
    draw_box(ax, 1.95, 3.45, 1.65, 0.90, "enc1 (24 ch)", "64x64 Conv-BN-ReLU x2", fc=SURF, ec=TRAIN_SAGE)
    draw_box(ax, 2.40, 2.25, 1.65, 0.90, "enc2 (48 ch)", "32x32 MaxPool -> Conv x2", fc=SURF, ec=TRAIN_SAGE)
    draw_box(ax, 2.90, 1.05, 1.65, 0.90, "enc3 (96 ch)", "16x16 MaxPool -> Conv x2", fc=SURF, ec=TRAIN_SAGE)

    # Bottleneck
    draw_box(ax, 4.70, 0.20, 1.85, 0.90, "bottleneck (192 ch)", "8x8 + Dropout2d(0.15)", fc="#FDF1EC", ec=ACCENT, title_color=ACCENT)

    # Decoder
    draw_box(ax, 6.70, 1.05, 1.75, 0.90, "dec3 (96 ch)", "up3 16x16 + Dropout2d", fc="#FDF1EC", ec=ACCENT)
    draw_box(ax, 7.20, 2.25, 1.75, 0.90, "dec2 (48 ch)", "up2 32x32 + Dropout2d", fc="#FDF1EC", ec=ACCENT)
    draw_box(ax, 7.65, 3.45, 1.70, 0.90, "dec1 (24 ch)", "up1 64x64 Conv x2", fc=SURF, ec=TRAIN_SAGE)

    # Output Head
    draw_box(ax, 9.55, 3.45, 1.25, 0.90, "1x1 Head", "(B, 1, 64, 64)\nLogits", fc="#EDEAE0", ec=INK)

    # Forward encoder-decoder path arrows
    draw_arrow(ax, 1.70, 3.90, 1.95, 3.90)
    draw_arrow(ax, 2.75, 3.45, 3.00, 3.15)
    draw_arrow(ax, 3.22, 2.25, 3.48, 1.95)
    draw_arrow(ax, 3.95, 1.05, 4.70, 0.65)
    draw_arrow(ax, 6.55, 0.65, 7.15, 1.05)
    draw_arrow(ax, 7.75, 1.95, 7.95, 2.25)
    draw_arrow(ax, 8.20, 3.15, 8.40, 3.45)
    draw_arrow(ax, 9.35, 3.90, 9.55, 3.90)

    # Skip connections
    draw_arrow(ax, 3.60, 3.90, 7.65, 3.90, color=MUTED, lw=1.1)
    ax.text(5.62, 4.03, "Skip concat (64x64, +24 ch)", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    draw_arrow(ax, 4.05, 2.70, 7.20, 2.70, color=MUTED, lw=1.1)
    ax.text(5.62, 2.83, "Skip concat (32x32, +48 ch)", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    draw_arrow(ax, 4.55, 1.50, 6.70, 1.50, color=MUTED, lw=1.1)
    ax.text(5.62, 1.63, "Skip concat (16x16, +96 ch)", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    ax.text(5.5, -0.22, "Loss: 0.5 x BCEWithLogitsLoss(pos_weight=50) + 0.5 x SoftDiceLoss  |  MC Dropout active in bottleneck, dec3, dec2 (T=20 stochastic passes)", ha="center", va="bottom", fontsize=7.8, color=MUTED)

    out_path = os.path.join(OUT_DIR, "iceunet_architecture.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    make_scene_strips_overview()
    make_pipeline_overview()
    make_iceunet_architecture()
