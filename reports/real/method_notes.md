# DFSAR Polarimetric Processing: Methodology & Theoretical Justification
_Generated: 2026-10-18 (Block 2 of Phase 5)_

---

## 1. Product Selection: SLI vs. SRI vs. GRI

Chandrayaan-2 DFSAR distributes data at multiple product levels:

| Product Code | Processing Level | Data Type | Geometry | Description |
|---|---|---|---|---|
| **SLI** (`d_sli_xx_fp_*`) | **L1A** | **`ComplexLSB8`** (float32 I & Q) | Slant Range | **Single-look complex (SLC)** scattering matrix ($S_{HH}, S_{HV}, S_{VH}, S_{VV}$) |
| **SRI** (`d_sri_xx_fp_*`) | **L2** | `UnsignedLSB2` (uint16) | Seleno-Referenced (Polar Stereographic, 25 m/px) | Multilooked detected magnitude/intensity |
| **GRI** (`d_gri_xx_fp_*`) | **L2** | `UnsignedLSB2` (uint16) | Ground Range | Detected ground-range intensity |

### Physical Justification for Product Selection

> **CRITICAL DECISION: The SLI (L1A complex) product MUST be used. CPR computation from SRI or GRI (magnitude-only) is physically invalid.**

### Why Magnitude-Only CPR is Physically Invalid
Circular Polarization Ratio (CPR) is defined as:
$$\text{CPR} = \frac{\sigma_{SC}}{\sigma_{OC}} = \frac{\langle |S_{RR}|^2 \rangle}{\langle |S_{RL}|^2 \rangle}$$
where $SC$ denotes same-sense circular receive and $OC$ denotes opposite-sense circular receive for a circularly polarized transmit wave.

Transforming the complex Sinclair scattering matrix $\mathbf{S} = \begin{pmatrix} S_{HH} & S_{HV} \\ S_{VH} & S_{VV} \end{pmatrix}$ from the linear horizontal/vertical basis $(H, V)$ to the circular basis $(R, L)$ yields:
$$S_{RR} = \frac{1}{2} \left[ (S_{HH} - S_{VV}) + i (S_{HV} + S_{VH}) \right]$$
$$S_{RL} = \frac{1}{2} \left[ (S_{HH} + S_{VV}) + i (S_{VH} - S_{HV}) \right]$$

For reciprocal media ($S_{HV} = S_{VH}$):
$$S_{RR} = \frac{1}{2} \left[ (S_{HH} - S_{VV}) + 2i S_{HV} \right]$$
$$S_{RL} = \frac{1}{2} (S_{HH} + S_{VV})$$

Notice that $|S_{RR}|^2$ and $|S_{RL}|^2$ depend strictly on the **coherent interference and relative phase difference** $\Delta \phi = \angle S_{HH} - \angle S_{VV}$ between channels:
$$|S_{RL}|^2 = \frac{1}{4} \left( |S_{HH}|^2 + |S_{VV}|^2 + 2 |S_{HH}| |S_{VV}| \cos(\angle S_{HH} - \angle S_{VV}) \right)$$
$$|S_{RR}|^2 = \frac{1}{4} \left( |S_{HH} - S_{VV}|^2 + 4 |S_{HV}|^2 + 4 \text{Im}\{(S_{HH} - S_{VV}) S_{HV}^*\} \right)$$

In detected magnitude products (SRI and GRI), only $|S_{HH}|$ and $|S_{HV}|$ are recorded; **all phase angles ($\angle S_{HH}, \angle S_{VV}, \angle S_{HV}$) are permanently discarded**. Without complex phase:
1. Coherent cancellation between $S_{HH}$ and $S_{VV}$ cannot occur.
2. The co-pol phase difference $\Delta \phi_{co}$, which distinguishes surface bounce ($\Delta \phi \approx 0^\circ$) from double bounce ($\Delta \phi \approx 180^\circ$), is lost.
3. Approximating CPR by the linear ratio $\frac{\sigma_{HV}}{\sigma_{HH}}$ is a severe error that conflates cross-polarization with circular polarization helicity reversal.

Therefore, the **complex SLI dataset** is the only scientifically valid product for CPR and DOP estimation.

---

## 2. Mathematical Formulation: Multilooked Stokes Parameters & DOP

### The Mathematical Necessity of Multilooking
For any single pixel in a single-look complex SAR image, the complex field components $E_R = S_{RR}$ and $E_L = S_{RL}$ are deterministic scalars. The instantaneous Stokes vector is:
$$S_0 = |E_R|^2 + |E_L|^2$$
$$S_1 = 2 \text{Re}(E_R E_L^*)$$
$$S_2 = 2 \text{Im}(E_R E_L^*)$$
$$S_3 = |E_R|^2 - |E_L|^2$$

By algebraic identity:
$$S_1^2 + S_2^2 + S_3^2 = 4 |E_R|^2 |E_L|^2 + (|E_R|^2 - |E_L|^2)^2 = (|E_R|^2 + |E_L|^2)^2 = S_0^2$$
Consequently:
$$\text{DOP}_{\text{single-look}} = \frac{\sqrt{S_1^2 + S_2^2 + S_3^2}}{S_0} \equiv 1.0000 \quad \text{for EVERY pixel!}$$

**A single-look radar image cannot measure depolarization.** Depolarization is a statistical property of partially coherent waves, manifesting as cross-channel decorrelation. Multilook spatial averaging over a window $W \times W$ is required **BEFORE** computing DOP:
$$\langle \sigma_{RR} \rangle = \frac{1}{W^2} \sum_{i,j \in W} |S_{RR}(i,j)|^2$$
$$\langle \sigma_{RL} \rangle = \frac{1}{W^2} \sum_{i,j \in W} |S_{RL}(i,j)|^2$$
$$\langle P_{RL} \rangle = \frac{1}{W^2} \sum_{i,j \in W} 2 S_{RR}(i,j) S_{RL}^*(i,j)$$

From these multilooked quantities:
$$S_0 = \langle \sigma_{RR} \rangle + \langle \sigma_{RL} \rangle$$
$$S_1 = \text{Re}\langle P_{RL} \rangle$$
$$S_2 = \text{Im}\langle P_{RL} \rangle$$
$$S_3 = \langle \sigma_{RR} \rangle - \langle \sigma_{RL} \rangle$$
$$\text{CPR} = \frac{\langle \sigma_{RR} \rangle}{\langle \sigma_{RL} \rangle + \epsilon}$$
$$\text{DOP} = \frac{\sqrt{S_1^2 + S_2^2 + S_3^2}}{S_0 + \epsilon}$$

---

## 3. Configuration & Label Rules

- **Window Size ($W$):** $W = 7$ pixels (configurable boxcar filter). Provides $7 \times 7 = 49$ independent looks, reducing speckle variance and enabling accurate DOP estimation down to $< 0.10$.
- **Ice Anomaly Criteria:**
  - $\text{CPR} > 1.0$: Same-sense circular backscatter exceeds opposite-sense circular backscatter (Coherent Backscatter Opposition Effect — CBOE).
  - $\text{DOP} < 0.13$: Low degree of polarization, indicative of multiple volume scattering inside low-loss dielectric media (water ice).
- **Patch Label Rule:** A patch is assigned label $y = 1$ (ice candidate) if the fraction of pixels meeting both criteria ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$) exceeds `patch_label_fraction = 0.05` (5%). Otherwise, $y = 0$.
- **Model Input Constraint:** Model input tensors contain **raw radar channels only** ($\log_{10}(|S_{HH}|^2 + \epsilon)$, $\log_{10}(|S_{HV}|^2 + \epsilon)$, $\log_{10}(|S_{VV}|^2 + \epsilon)$). CPR and DOP are strictly used for label generation and evaluation baseline comparison, **NEVER** as model inputs.
