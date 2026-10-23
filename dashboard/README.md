# IceShadowNet Dashboard

Interactive Next.js visual dashboard for Chandrayaan-2 DFSAR lunar polar subsurface ice detection benchmarking.

## Features

- **Overview:** Project summary, rotated 90° pure SAR hero visualization, headline ROC-AUC metrics, and negative result banner.
- **How It Works:** 8-stage physical radar preprocessing pipeline (HH, HV, VV, Multilook, CPR, DOP, Label Mask, Patch Grid) with pure pixel imagery, min/max legends, and interactive Untrained vs. Trained Grad-CAM slider.
- **Results:** 4 headline model verdict cards, ROC-AUC SVG chart with Logistic Regression and coin-flip baselines, 3 key takeaways from Section 3, and collapsible detailed views (All runs with nowrap CIs, Training details, Pretrained experiments, and Per-strip metrics).
- **Explore:** Complete 338-patch contact sheet with plain English filter buttons (All, Correct, False alarm, Missed ice, Strips 6/10/14), multi-channel detail view (HH, HV, VV), and advanced threshold slider.
- **What Went Wrong:** Highest-confidence failure case diagnostics (top-2 false positives, top-2 false negatives) side-by-side with pure 384×384 WebP Grad-CAM overlays, overlapping false negative analysis, and methodological limitations.

## Local Development

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to view the application.

## Static Build & Export

```bash
npm run build
```

Generates static export files in `dashboard/out/`.

## Deployment

### Vercel
- **Framework Preset:** Next.js
- **Root Directory:** `dashboard`
- **Build Command:** `next build`
- **Output Directory:** Next.js default (`out` / static output)
- **Environment Variables:** None required. All benchmark results, manifests, and WebP imagery are pre-packaged under `public/data/`.
