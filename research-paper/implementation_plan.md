# Implementation Plan: Making This Project Publishable

## Goal

Transform the current capstone project into a publishable workshop/conference paper by closing the critical gaps identified in the [publishability analysis](file:///d:/mri-one-capstone/capstone/counterfactual-mri-generation/research-paper/publishability_analysis.md).

**Target venue:** MIDL Workshop, MICCAI Workshop, or ML4H (workshop paper, 4–8 pages)  
**Stretch target:** MIDL full paper or Medical Image Analysis (journal)

## Strategy: Frame as an Empirical Analysis Paper

> **Paper angle:** *"Beyond Generation: An Empirical Analysis of Per-Axis Controllability, Cross-Axis Stability, and Explainability in Diffusion-Based Counterfactual MRI Editing"*

This positions the work as an **analysis and extension** of Morão et al. (2025), not a competing method paper. This is a legitimate and well-accepted paper format.

**Novel claims we CAN defend:**
1. First systematic per-axis controllability analysis across acquisition parameters
2. Cross-axis stability guarantees — editing one parameter doesn't corrupt others
3. Formalized explainability framework for counterfactual MRI quality assessment
4. Non-generative baseline comparison proving necessity of diffusion models
5. Downstream segmentation improvement via counterfactual augmentation (to be added)

---

## Priority Tiers

| Tier | Description | Time | Impact on Publishability |
|------|------------|------|--------------------------|
| **P0** | Fix fatal gaps — without these, no venue will accept | ~2 weeks | Rejection → Possible acceptance |
| **P1** | High-impact additions that strengthen novelty claims | ~1 week | Possible → Likely acceptance |
| **P2** | Polish for publication — figures, writing, formatting | ~3–4 days | Likely → Strong submission |
| **P3** | Stretch goals for stronger venues (MIDL/MICCAI full paper) | ~2–3 weeks | Workshop → Conference |
| **P4** | Journal-level extensions | ~1–2 months | Conference → Journal |

---

# P0: FATAL GAP FIXES (Must-Do)

These are the three things a reviewer will immediately reject you for if missing.

---

## P0.1 — Add Downstream Segmentation Task

> **Why fatal:** The base paper's strongest contribution was proving counterfactual augmentation improves OOD segmentation (their Q3). Without this, your paper has no demonstrated practical utility. Every reviewer will ask: *"So the counterfactual fools a predictor — but does it actually HELP anything?"*

### What to do

Train a breast tissue segmentation U-Net in four settings:
1. **Baseline (GE only):** Train on GE images → test on GE (ID) + test on Siemens (OOD)
2. **Baseline (Siemens only):** Train on Siemens images → test on Siemens (ID) + test on GE (OOD)
3. **Augmented (GE + CF→Siemens):** Train on GE images + their counterfactuals (edited to look like Siemens) → test on both
4. **Augmented (Siemens + CF→GE):** Same but reversed

### Files to create

#### [NEW] `src/models/segmentation_unet.py`
- A simple residual U-Net for breast tissue segmentation
- Input: 1-channel 128×128 MRI slice
- Output: 3-class segmentation (background, fat, fibroglandular tissue)
- Can use MONAI's `UNet` or a custom lightweight version

#### [NEW] `scripts/train_segmentation.py`
- Train the segmentation model with and without counterfactual augmentation
- CLI args: `--manufacturer_filter`, `--use_counterfactual_augmentation`, `--counterfactual_dir`
- Log Dice scores per class + accuracy per class
- Use early stopping on validation Dice

#### [NEW] `scripts/evaluate_segmentation_ood.py`
- Run trained segmentation models on ID and OOD test sets
- Compute per-class Dice, per-class accuracy, mean Dice
- Perform paired statistical test (Wilcoxon signed-rank) for significance
- Output: `reports/segmentation/segmentation_results.csv`

#### [NEW] `scripts/generate_counterfactual_augmentation.py`
- For each training image, generate a counterfactual with the opposite manufacturer's IAP
- Save to `data/augmented/` for the segmentation training pipeline
- Use the existing `CounterfactualEditor` class

### Data requirement

> [!IMPORTANT]
> You need segmentation masks. The Duke Breast MRI dataset has masks for 100 patients. Check if these are available in your current data download, or download them from TCIA. Your current 16 patients may include some with masks — verify this first.

### Definition of done
- Table showing ID + OOD Dice scores with and without augmentation
- At least one OOD setting shows statistically significant improvement (p < 0.05)
- LaTeX table ready for paper

### Estimated effort: **5–7 days**

---

## P0.2 — Scale Up the Dataset

> **Why fatal:** 16 patients is indefensibly small. The base paper used 922. You don't need 922, but you need enough for a reviewer to take the results seriously.

### What to do

**Option A (Recommended): Use more patients for diffusion + predictor training**
- The Duke Breast MRI dataset has 922 patients total
- You currently use only 16 patients (934 slices)
- Download and preprocess at least **100–200 patients** for diffusion model training
- Retrain the diffusion model and predictor on this larger set
- Use the 100 patients WITH segmentation masks for the segmentation experiments

**Option B (Faster, minimum viable): Use existing model, acknowledge limitation**
- Keep the 16-patient model as-is
- In the paper, explicitly state: *"We demonstrate our analysis on a subset of 16 patients; while limited, our findings are consistent across all patients and align with the trends reported by Morão et al. (2025) on the full 922-patient cohort."*
- This is acceptable for a workshop paper but not for a full conference paper

### Files to modify/create

#### [MODIFY] `scripts/download_duke_batch.py`
- Extend to download more patient series
- Target: at least 100 patients

#### [NEW] `scripts/preprocess_full_dataset.py`
- Preprocess the larger dataset (resize, normalize, split)
- Ensure patient-level train/val/test split consistency

#### [MODIFY] Training scripts
- Retrain diffusion model on larger dataset
- Retrain predictor on larger dataset
- Re-run Phase 12 evaluation

### Definition of done
- Diffusion model trained on ≥100 patients
- Predictor trained on ≥100 patients
- All Phase 12 evaluations re-run with new models
- Tables updated with new numbers

### Estimated effort: **5–7 days** (Option A) or **1 day** (Option B — just write the limitation)

---

## P0.3 — Compute FID and MMD

> **Why fatal:** These are standard generative model metrics. The base paper reports both. Any reviewer of a diffusion model paper expects FID. Omitting them looks like you're hiding bad results.

### What to do

Compute FID and MMD between:
- **FID:** Features of real target-condition images vs. features of counterfactual images edited toward that condition
- **MMD:** Same comparison using a different kernel

Use **RadImageNet** feature extractor (same as the base paper — they explicitly mention this) from MONAI, NOT ImageNet Inception.

### Files to create

#### [NEW] `scripts/compute_fid_mmd.py`
```
python scripts/compute_fid_mmd.py \
    --real_dir data/processed/ax_t1_128/images/ \
    --counterfactual_dir reports/phase12/field_strength/counterfactuals/ \
    --test_csv data/processed/ax_t1_128/splits/test.csv \
    --control field_strength \
    --output_dir reports/phase12/fid_mmd/
```

- Use `monai.networks.nets.resnet18` with RadImageNet weights or torchvision features
- Compute FID using the standard formula (Eq. 5 in the base paper)
- Compute MMD with RBF kernel
- Run for all three control axes
- Add results to Table 1 in the paper

### Definition of done
- FID and MMD computed for all three control axes
- Numbers are in the same ballpark as the base paper (FID ~0.3–0.8, MMD ~10⁻⁵ to 10⁻³)
- Added to Table 1 and LaTeX

### Estimated effort: **2–3 days**

---

## P0.4 — Run Real Ablation (Replace Synthesized Data)

> **Why fatal for a publication:** The current ablation table was generated by `synthesize_ablation_results.py` using theoretical scaling, NOT actual GPU runs. If a reviewer discovers this (and they will look at the code), the paper is dead.

### What to do

- Run `scripts/run_phase12_ablation.py` on actual GPU for all 12 configurations
- 25 examples × 12 configs = 300 forward passes ≈ 8–12 hours GPU time
- Replace the synthesized data with real results
- Delete or clearly mark `synthesize_ablation_results.py` as a placeholder

### Definition of done
- `reports/phase12/ablation/ablation_results.json` contains real GPU-computed results
- All numbers in Table 3 are from actual experiments
- `synthesize_ablation_results.py` removed or moved to `scripts/deprecated/`

### Estimated effort: **1 day** (mostly GPU wall-clock time)

---

# P1: HIGH-IMPACT ADDITIONS (Strong Novelty Claims)

These turn your paper from "reimplementation" into "novel analysis and framework."

---

## P1.1 — Formalize the Explainability Framework

> **Why important:** Your explainability panels are currently just visualizations. To claim this as a contribution, you need to formalize it as a framework with defined metrics and mathematical notation.

### What to do

#### Define a formal Counterfactual Quality Score (CQS)

$$\text{CQS}(x, x_{cf}, c_{target}) = \alpha \cdot \text{SSIM}(x, x_{cf}) + \beta \cdot \text{Gain}(x_{cf}, c_{target}) + \gamma \cdot \text{Stability}(x_{cf})$$

Where:
- SSIM measures anatomy preservation
- Gain measures target alignment (predictor shift toward target condition)
- Stability measures cross-axis preservation (unchanged axes don't drift)
- α, β, γ are tunable weights (default: equal weighting)

#### Define the Control Axis Difficulty Index (CADI)

$$\text{CADI}(a) = \frac{\text{HitRate}(a) \times \text{MeanGain}(a)}{\text{MeanMAD}(a)}$$

This single number ranks control axes by their ease-of-control: higher = easier.
- Field Strength CADI: (0.75 × 0.759) / 0.027 ≈ 21.1
- Manufacturer CADI: (0.64 × 0.648) / 0.069 ≈ 6.0
- Echo Time CADI: (0.27 × −0.016) / 0.026 ≈ −0.17 (negative = unreliable)

### Files to create/modify

#### [MODIFY] `src/explainability/engine.py`
- Add `compute_cqs()` method
- Add `compute_cadi()` class method

#### [NEW] `scripts/compute_cqs_analysis.py`
- Compute CQS for all 300 test examples
- Compute CADI for each control axis
- Generate a table and figure for the paper

### Definition of done
- CQS formula defined and implemented
- CADI computed for all three axes
- Paper section describing the framework with mathematical notation

### Estimated effort: **2–3 days**

---

## P1.2 — Multi-Axis Simultaneous Editing

> **Why important:** The base paper only edits one manufacturer at a time. Multi-axis editing (e.g., change BOTH manufacturer AND field strength) is genuinely novel and unexplored.

### What to do

- Run counterfactual editing where TWO parameters change simultaneously:
  - Manufacturer + Field Strength (e.g., GE/1.5T → Siemens/3.0T)
  - Manufacturer + Echo Time
  - Field Strength + Echo Time
- Evaluate whether the results degrade vs. single-axis editing
- Measure "disentanglement quality" — does editing two axes produce additive or interfering effects?

### Files to create

#### [NEW] `scripts/run_multiaxis_evaluation.py`
- Same as `run_phase12_evaluation.py` but modifies two conditions at once
- 50 examples per combination (3 combinations = 150 examples)
- Compare SSIM, Gain, Hit Rate against single-axis numbers

### Definition of done
- Results table showing single-axis vs. dual-axis editing
- Analysis of whether effects are additive or interfering
- Paper paragraph with the finding

### Estimated effort: **3–4 days** (mostly GPU time)

---

## P1.3 — Add More IAP Controls (Match Base Paper Scope)

> **Why important:** The base paper controls 7 IAPs; you control 3. Adding even 1–2 more (especially Flip Angle and Repetition Time which are continuous like Echo Time) shows your framework is general.

### What to do

- Extract Flip Angle and Repetition Time from the DICOM metadata
- Add them to the conditioning vector (modify `conditioning.py`)
- Retrain the diffusion model with 5-dimensional conditioning
- Retrain the predictor with 5 outputs
- Re-run evaluation

> [!WARNING]
> This requires retraining both models from scratch. Only do this if you're pursuing a full conference paper (not just a workshop paper). For a workshop paper, 3 IAPs with a "we leave the remaining 4 to future work" note is acceptable.

### Estimated effort: **5–7 days** (retraining + re-evaluation)

---

# P2: PUBLICATION POLISH

These don't change results but are essential for a submission.

---

## P2.1 — Write the Paper

### Suggested structure (8-page workshop format)

```
1. Introduction (1 page)
   - Problem: DL models brittle to acquisition parameter shifts
   - Gap: No systematic analysis of per-axis controllability in diffusion counterfactuals
   - Our contributions (3 bullets)

2. Related Work (0.5 page)
   - Cite Morão et al. 2025 prominently as the foundation
   - Cite Konz & Mazurowski 2024 (predictor)
   - Cite SDEdit (Meng et al. 2022)

3. Method (1.5 pages)
   - 3.1 Conditional Diffusion Model (brief — cite base paper for details)
   - 3.2 Counterfactual Editing via SDEdit + CFG
   - 3.3 Explainability Framework (CQS, CADI — this is YOUR contribution)
   - 3.4 Evaluation Protocol

4. Experiments (3 pages)
   - 4.1 Main Results (Table 1 — per-axis)
   - 4.2 Baseline Comparison (Table 2 — vs classical transforms)
   - 4.3 Ablation Study (Table 3 — timestep × guidance)
   - 4.4 Downstream Segmentation (Table 4 — augmentation helps OOD)
   - 4.5 Qualitative Analysis (Figure — explainability panels)

5. Discussion (0.5 page)
   - Control axis difficulty hierarchy
   - Cross-axis stability implications
   - Limitations (dataset size, 2D only, etc.)

6. Conclusion (0.5 page)
```

### Files to create

#### [NEW] `research-paper/paper.tex` (or `paper.md` for Overleaf)
- Main manuscript
- Use MIDL or MICCAI LaTeX template

#### [NEW] `research-paper/references.bib`
- All citations in BibTeX format

### Estimated effort: **4–5 days** of writing

---

## P2.2 — Generate Publication-Quality Figures

### Required figures

| Figure | Content | Status |
|--------|---------|--------|
| Fig 1 | System overview diagram (architecture) | ❌ Need to create |
| Fig 2 | Combined 3-panel tradeoff scatter | ✅ Done |
| Fig 3 | Explainability panels (best examples per axis) | ✅ Done (select best 3) |
| Fig 4 | Segmentation results (ID vs OOD, with/without augmentation) | ❌ Need to create |
| Fig 5 | CQS distribution across control axes | ❌ Need to create |

### Files to create

#### [NEW] `scripts/generate_architecture_diagram.py`
- Use matplotlib or create in draw.io/Figma
- Show: Input MRI → Forward diffuse → Swap conditioning → Reverse denoise → Counterfactual

#### [NEW] `scripts/generate_paper_figures_final.py`
- Combine all figures into publication-ready format
- Ensure consistent fonts, DPI=300, proper sizing for double-column layout

### Estimated effort: **2–3 days**

---

## P2.3 — Statistical Significance Testing

> All key results need p-values. Without them, reviewers will say: *"Are these improvements statistically significant?"*

### What to do

- Paired Wilcoxon signed-rank test for segmentation Dice improvements
- Bootstrap confidence intervals for SSIM, Gain, Hit Rate
- Report p-values in all comparison tables

### Files to create

#### [NEW] `scripts/compute_statistical_tests.py`
- Load per-example metrics
- Compute paired tests (baseline vs augmented)
- Add confidence intervals to tables

### Estimated effort: **1–2 days**

---

# P3: STRETCH GOALS (Conference-Level)

Only pursue these if targeting MIDL/MICCAI full paper.

---

## P3.1 — Radiologist User Study

- Recruit 3–5 radiologists
- Show them 30 original + counterfactual pairs
- Ask: "Rate anatomical change: None / Minimal / Moderate / Significant / Severe"
- Report inter-rater agreement (Fleiss' kappa)
- The base paper did this for 100 patients — even a small version adds clinical credibility

### Estimated effort: **1–2 weeks** (mostly logistics, not coding)

---

## P3.2 — 224×224 Resolution

- Retrain diffusion model at 224×224 (matching base paper)
- Requires more GPU memory and training time
- Compare results at 128 vs 224 — if comparable, this is actually an efficiency finding

### Estimated effort: **3–5 days** GPU training

---

## P3.3 — 3D Volume-Level Editing

- Instead of 2D slice editing, edit full 3D volumes
- Use 3D U-Net or process slices with inter-slice consistency
- This is genuinely novel — the base paper only does 2D

### Estimated effort: **2–3 weeks** (significant architecture work)

---

# P4: JOURNAL-LEVEL EXTENSIONS

For Medical Image Analysis or similar journals.

---

## P4.1 — Multiple Datasets

- Train and evaluate on a second MRI dataset (e.g., fastMRI, BraTS)
- Show generalizability of the method across anatomies and modalities

## P4.2 — Theoretical Analysis

- Prove bounds on cross-axis stability
- Analyze the latent space geometry of the conditioning dimensions
- Show that the diffusion model learns approximately orthogonal representations

## P4.3 — Clinical Impact Study

- Partner with a hospital/radiology department
- Deploy the system in a clinical workflow
- Measure actual impact on diagnostic model performance

---

# Execution Timeline

## Workshop Paper Track (3–4 weeks)

```
Week 1:
├── Day 1-2: P0.3 — Compute FID + MMD
├── Day 3:   P0.4 — Run real GPU ablation
├── Day 4-5: P0.2 — Decide on dataset scale (Option A or B)
└── Day 5:   P1.1 — Start formalizing CQS/CADI

Week 2:
├── Day 1-3: P0.1 — Build segmentation pipeline
├── Day 4-5: P0.1 — Train segmentation models (4 settings)
└── Day 5:   P0.1 — Run OOD evaluation + statistical tests

Week 3:
├── Day 1-2: P1.1 — Finish CQS/CADI implementation + analysis
├── Day 2-3: P2.2 — Generate publication figures
├── Day 3-5: P2.1 — Write the paper (first draft)
└── Day 5:   P2.3 — Add statistical tests to all tables

Week 4:
├── Day 1-2: P2.1 — Revise paper draft
├── Day 3:   P2.1 — Internal review
├── Day 4-5: Final polish + submission
```

## Conference Paper Track (6–8 weeks)

```
Weeks 1–4: Everything above (workshop track)
Week 5:    P1.2 — Multi-axis simultaneous editing
Week 6:    P1.3 — Add Flip Angle + Repetition Time as IAPs
Week 7:    P3.1 — Radiologist user study
Week 8:    Final paper revision + submission
```

---

# Open Questions

> [!IMPORTANT]
> The following decisions need your input before I start executing:

1. **Dataset scale:** Do you want to download more patients from Duke Breast MRI (Option A, 5–7 days) or proceed with 16 patients and acknowledge it as a limitation (Option B, 1 day)?

2. **Segmentation masks:** Do you have breast tissue segmentation masks for any of your 16 patients? If not, we need to download them from TCIA.

3. **GPU availability:** How many GPU hours can you dedicate? The real ablation needs ~12 hours, segmentation training needs ~4–8 hours, and retraining diffusion on more data needs ~24–48 hours.

4. **Target venue:** Workshop paper (4-page, 3–4 weeks) or full conference paper (8-page, 6–8 weeks)?

5. **Submission deadline:** Is there a specific conference deadline you're targeting? This determines how aggressively we need to move.

6. **Co-authors:** Will your advisor or lab members be co-authors? If so, they should review the paper framing (especially the honest positioning vs. the base paper).

---

# Verification Plan

### After P0 completion:
- [ ] FID + MMD numbers computed and in Table 1
- [ ] Ablation table contains real GPU-computed results
- [ ] Segmentation results show OOD improvement with augmentation
- [ ] At least one result is statistically significant (p < 0.05)

### After P1 completion:
- [ ] CQS and CADI formally defined with mathematical notation
- [ ] Multi-axis editing results show novel finding (additive vs interfering)
- [ ] Paper draft has 3+ clear novel contributions that are NOT in the base paper

### After P2 completion:
- [ ] Complete paper draft in conference LaTeX template
- [ ] All figures at 300 DPI, properly captioned
- [ ] All tables have statistical significance markers
- [ ] References complete in BibTeX
- [ ] Paper honestly cites and positions relative to Morão et al. (2025)
