# 18 · Independent Component Analysis — Latent Source Separation in Pharma Batch Manufacturing

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ICA_Pharma_Features/blob/main/18_ICA_Pharma_Features.ipynb)

> *"PCA finds directions of maximum variance. ICA finds directions of maximum independence. In a real process, what's driving quality is almost never just variance — it's hidden causes."*

---

## 🎯 The Batch Failed. But Which Cause?

The quality report says: *viscosity out of specification.*

The quality engineer writes a corrective action: *review mixing procedure.* The next batch runs with closer attention to mixing. Viscosity comes back in spec. Two batches later, it drifts again — from a completely different direction. Active concentration this time. Then pH. Then turbidity.

The corrective actions were not wrong. They were answering the wrong question.

**Viscosity is not a cause. It is a symptom.** Every quality sensor on this syrup batch line — pH, viscosity, conductivity, density, turbidity, reaction temperature, reaction time, active concentration — is measuring a mixture of three independent process drivers that cannot be observed directly:

- **F1** — variation in active reagent dosage and purity
- **F2** — variation in solvent quality and ionic balance
- **F3** — variation in agitation efficiency and mixing kinetics

When F1 drifts, viscosity, density, and active concentration all shift together — because they share the same root cause. The quality report will flag three out-of-spec readings. But there is one corrective action: fix the reagent dosage.

A quality system that reads sensor deviations and writes corrective actions for each one is not doing root cause analysis. It is writing eight different descriptions of the same event.

**Independent Component Analysis (ICA)** solves the prior problem. Before you can decide what to fix, you need to know what is broken. ICA reverses the mixing process — it separates the 8 observed sensor signals back into 3 independent source signals, one per root cause, without ever measuring those sources directly. The output is not a deviation report. It is a **root cause attribution**: IC3 deviated on batch 847 — check agitation speed and temperature profile.

This project applies FastICA to **1,583 pharmaceutical syrup batch records** from 8 quality sensors, recovering 3 independent components that align with the known process root causes: IC1 → Active Reagent (r = 0.74), IC2 → Solvent / Ionic Balance (r = 0.70), IC3 → Agitation & Kinetics (r = 0.85). A single **IC distance metric** classifies 90% of batches as in-spec, 5% as warning, and 5% as alert — with a direction for each alert that points directly to the responsible process driver.

> 📦 **Full Project Pack** — complete dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 📊 Dataset

**File:** `pharma_data.csv` — 1,583 pharmaceutical syrup batch records from 8 quality sensors and 3 latent process drivers.

**Observed Sensors (model input — 8 columns):**

| Column | Unit | Description | Dominant Source |
|--------|------|-------------|----------------|
| `ph_final` | — | Final batch pH | F2 (Solvent / Ionic) |
| `viscosity_cp` | cP | Product viscosity | F1 (Active Reagent) |
| `conductivity_ms_cm` | mS/cm | Ionic conductivity | F2 (Solvent / Ionic) |
| `density_g_ml` | g/mL | Product density | F1 (Active Reagent) |
| `turbidity_ntu` | NTU | Turbidity (clarity) | F3 (Agitation) |
| `reaction_temp_c` | °C | Peak reaction temperature | F3 (Agitation) |
| `reaction_time_min` | min | Total batch duration | F3 (Agitation) |
| `active_conc_mg_ml` | mg/mL | Active ingredient concentration | F1 (Active Reagent) |

**Latent Sources (validation only — 3 columns):**

| Column | Description | ICA Recovery |
|--------|-------------|--------------|
| `f1_active_reagent` | True active reagent dosage / purity variation | IC1, r = 0.74 |
| `f2_solvent_salts` | True solvent quality / ionic balance variation | IC2, r = 0.70 |
| `f3_agitation_kinetics` | True agitation / mixing kinetics variation | IC3, r = 0.85 |

> ⚠️ **In real production, only the 8 sensor columns are available.** The true source columns exist here for educational validation only. FastICA recovers them purely from the observed sensor readings.

**Data Origin — Real-World Source Systems:**

| Sensor Group | Source System | Instrument |
|---|---|---|
| `ph_final`, `conductivity_ms_cm` | PAT / In-line analyzer | pH electrode + conductivity cell at batch end |
| `viscosity_cp` | Quality Lab / In-line viscometer | Rotational viscometer or in-line Coriolis |
| `density_g_ml` | Quality Lab | Oscillating U-tube density meter |
| `turbidity_ntu` | PAT | In-line nephelometer |
| `reaction_temp_c`, `reaction_time_min` | DCS / Batch execution system | RTD at reactor jacket + MES timestamps |
| `active_conc_mg_ml` | HPLC / LIMS | Post-batch high-performance liquid chromatography |

**Key EDA Findings:**

The sensor correlation structure is not noise — it is the fingerprint of the mixing process. Viscosity and active concentration are correlated (both driven by F1). pH and conductivity are correlated (both driven by F2). Reaction time and turbidity are correlated (both driven by F3). These pairwise correlations are not causal relationships between sensors — they are reflections of a common hidden driver. ICA's job is to find those drivers by maximizing the statistical independence of the recovered signals.

---

## 🤖 Model

### Why ICA, not PCA?

The answer to this question is the entire point of the project.

PCA (Project 17) finds the directions of maximum **variance** in the data. On a motor with 10 sensors, the first principal component captured 54.4% of variance because the degradation cascade made all sensors move together in the same direction. PCA was effective because variance and the signal of interest were aligned.

In a pharma batch process, variance and root cause are not aligned. F1, F2, and F3 are designed to be independent — they are driven by different supply chain inputs, different process units, different control loops. A reagent dosage drift has nothing to do with a solvent ionic balance variation. These are orthogonal events, not correlated ones.

When the root causes are independent, **PCA will mix them into its components** — because PCA finds orthogonal directions of maximum variance, not independent directions. The ICA-vs-PCA source recovery comparison confirms this directly:

| Method | Component | Best Source Match | Recovery |r| |
|--------|-----------|------------------|----------|
| **ICA** | IC1 | F1 — Active Reagent | **0.7354** |
| **ICA** | IC2 | F2 — Solvent / Ionic | **0.6951** |
| **ICA** | IC3 | F3 — Agitation | **0.8490** |
| PCA | PC1 | F2 — Solvent / Ionic | 0.5808 |
| PCA | PC2 | F1 — Active Reagent | 0.8069 |
| PCA | PC3 | F3 — Agitation | 0.6760 |

ICA achieves higher alignment with the true physical causes for 2 of 3 components, and — critically — **maps each component to a unique root cause**. PCA's PC2 achieves 0.80 with F1, but PC1 and PC3 are partial mixtures. The PCA decomposition is usable for anomaly detection; it is not usable for root cause attribution. That distinction is the entire operational difference between the two methods.

### Why FastICA with k = 3?

The component count selection combines two inputs:

**PCA variance bound:** Running PCA on all 8 components, 3 principal components explain 94.3% of total variance. This means the 8 sensor signals carry roughly 3 dimensions of independent information — consistent with 3 root causes.

**Domain knowledge:** The process is known to have three independent input variability sources: reagent dosage, solvent preparation, and reactor agitation. These are separate process steps with separate control loops. k = 3 is not a statistical guess — it is a physical fact that the variance analysis confirms.

FastICA maximizes the **non-Gaussianity** (statistical independence) of each recovered component using a fixed-point iteration algorithm. Convergence was reached in 7 iterations. The algorithm performs internal whitening (decorrelation) before applying the independence criterion, so the ordering and scaling of ICs may differ between runs — but the recovered sources are stable.

### The Mixing Matrix — Reading the Root Cause Map

The ICA mixing matrix A describes how each source contributes to each observed sensor. It is the Rosetta Stone between sensor deviations and process causes:

| Sensor | IC1 (Active Reagent) | IC2 (Solvent / Ionic) | IC3 (Agitation) |
|--------|---------------------|----------------------|-----------------|
| `ph_final` | −0.242 | **+0.766** | −0.579 |
| `viscosity_cp` | **−0.865** | −0.450 | +0.029 |
| `conductivity_ms_cm` | −0.562 | **+0.723** | −0.378 |
| `density_g_ml` | **−0.964** | −0.032 | +0.192 |
| `turbidity_ntu` | −0.663 | −0.332 | −0.573 |
| `reaction_temp_c` | −0.487 | −0.224 | **−0.777** |
| `reaction_time_min` | −0.255 | −0.289 | **−0.878** |
| `active_conc_mg_ml` | **−0.854** | −0.354 | +0.343 |

**IC1 (Active Reagent):** dominant in density (−0.964), active concentration (−0.854), viscosity (−0.865). When IC1 deviates, expect these three sensors to co-deviate. Root action: review reagent batch certificate, check dosing pump calibration.

**IC2 (Solvent / Ionic Balance):** dominant in pH (+0.766), conductivity (+0.723). When IC2 deviates, pH and conductivity shift together. Root action: review water quality, check buffer preparation, inspect ion exchange resins.

**IC3 (Agitation & Kinetics):** dominant in reaction time (−0.878), reaction temperature (−0.777). When IC3 deviates, the batch runs hotter and faster or cooler and slower. Root action: inspect agitator speed control, check impeller wear, review cooling jacket flow.

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **FastICA convergence** | 5 iterations | Fast, stable solution |
| **PCA variance (3 PCs)** | 94.3% | Confirms k=3 covers the information space |
| **IC1 → F1 recovery** | r = 0.74 | Active reagent source well-recovered |
| **IC2 → F2 recovery** | r = 0.70 | Solvent source recovered (most overlap with F3) |
| **IC3 → F3 recovery** | r = 0.85 | Agitation source best-recovered |
| **Warning threshold (p90)** | dist ≥ 2.54 | 5.0% of batches (79 records) |
| **Alert threshold (p95)** | dist ≥ 2.86 | 5.1% of batches (80 records) |
| **In-spec batches** | 1,424 (90.0%) | Within normal IC-distance envelope |

**Batch Quality Distribution:**

| Status | Count | Share | Threshold |
|--------|-------|-------|-----------|
| ✅ In-spec | 1,424 | 90.0% | IC dist < 2.54 |
| ⚠️ Warning | 79 | 5.0% | 2.54 ≤ dist < 2.86 |
| 🔴 Alert | 80 | 5.1% | dist ≥ 2.86 |

---

## 🔍 The IC Distance Metric

The IC distance combines the three recovered source scores into a single batch health indicator:

```
IC dist = √(IC1² + IC2² + IC3²)
```

An in-spec batch sits near the origin in IC space — all three sources are within their normal ranges. An out-of-spec batch deviates radially from the origin in the direction of the responsible source. The **direction** of the deviation is the root cause attribution: a batch that drifts primarily along the IC1 axis failed because of reagent dosage variation, not solvent quality and not agitation.

This is what separates ICA-based quality monitoring from threshold-based alarm systems. A threshold system says: *viscosity = 162 cP, limit = 155 cP, out of spec.* An ICA system says: *IC1 distance = 3.1, IC2 = 0.4, IC3 = 0.6 — this is an active reagent event. Check lot certificate and dosing pump.*

---

## 🗂️ Repository Structure

```
ICA_Pharma_Features/
├── 18_ICA_Pharma_Features.ipynb    # Full educational notebook
├── pharma_data.csv                  # 250-row sample dataset
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (1,583 records), notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab (no installation):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ICA_Pharma_Features/blob/main/18_ICA_Pharma_Features.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/ICA_Pharma_Features.git
cd ICA_Pharma_Features
pip install -r requirements.txt
jupyter notebook 18_ICA_Pharma_Features.ipynb
```

**Option 3 — Streamlit Simulator:**

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

1. **Correlation is not causation — it is mixing.** When pH and conductivity are correlated in batch quality data, it does not mean one causes the other. It means they share a hidden driver — in this case, solvent ionic balance. ICA finds the driver. Conventional SPC finds the correlation and flags two non-causal problems.

2. **ICA's objective is statistical independence, not variance.** This distinction determines which algorithm to use. PCA is optimal when the signal of interest is the direction of maximum variance — degradation, drift, magnitude. ICA is optimal when the signal of interest is a set of independent causes — process drivers, root causes, supply chain inputs. Choose based on the problem, not habit.

3. **The mixing matrix is the deliverable, not the IC scores.** The IC scores tell you *that* a source deviated. The mixing matrix tells you *which sensors* will move when it does — and by how much. A quality engineer armed with the mixing matrix can design targeted corrective actions before the next batch runs.

4. **Sign and order of ICs are arbitrary — physical interpretation is not.** FastICA may return IC1 with a negative sign compared to F1, or reorder IC1 and IC2 between runs. The mathematical content is identical — only the labeling convention changes. The physical interpretation (which sensors load onto which component) is stable and recoverable by inspecting the mixing matrix against domain knowledge.

5. **ICA and PCA are complementary, not competing.** PCA revealed that 94.3% of the variance in 8 sensors is explained by 3 components — that information motivated k = 3 for ICA. The two methods answer different questions from the same data. A complete quality monitoring system would run both: PCA for anomaly detection and health trending, ICA for root cause attribution when an anomaly is confirmed.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
