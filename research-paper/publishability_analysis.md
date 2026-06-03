# Publishability Analysis: Honest Assessment

## Your Project vs. Base Paper (Morão et al., MIDL 2025)

**Base Paper:** "Data Augmentation for Medical Imaging: Counterfactual Simulation of Acquisition Parameters via Conditional Diffusion Model" — Morão, Forghani, Louçāo, Gouveia, Figueiredo & Santinha. Accepted at **MIDL 2025** (Medical Imaging with Deep Learning).

**Your Project:** "Explainable Counterfactual MRI Generation and Acquisition Parameter Control for Robust Medical Imaging AI"

---

## EXECUTIVE VERDICT

> **As it stands today, this project is NOT publishable as an independent research paper at a reputable venue.** It is, in its current form, a **reimplementation of the base paper with a smaller-scale evaluation** and some explainability additions that are not yet sufficient to constitute a novel contribution. However, **it CAN become publishable** with targeted additions (detailed below). The gap is bridgeable, but requires real new work.

---

## Part 1: What the Base Paper Does

| Aspect | Base Paper (Morão et al.) |
|--------|--------------------------|
| **Core idea** | Conditional DDPM to generate counterfactual MRIs by simulating different acquisition parameters (IAP) |
| **Conditioning** | 7 IAP: Manufacturer, Field Strength, Scan Options, Flip Angle, Slice Thickness, Echo Time, Repetition Time |
| **Dataset** | Duke Breast MRI — **922 patients** total; 822 for cDDGM + IAP predictor training; 100 with segmentation masks |
| **Image size** | 224 × 224 |
| **Diffusion model** | DDPM with cosine scheduler, 1000 steps, cross-attention conditioning, hybrid (time-embed + cross-attn) |
| **Editing method** | SDEdit-style: forward diffuse to `t=steps`, reverse denoise with new IAP + CFG |
| **IAP Predictor** | ResNet-18 modified for 7 IAP outputs (3 categorical + 4 continuous) |
| **Guidance** | Classifier-Free Guidance (CFG) |
| **Hyperparameter sweep** | Steps ∈ {25, 50, 75} × guidance ∈ {3, 5, 7} = 9 configs |
| **Metrics** | FID (RadImageNet), SSIM, MMD, IAP prediction accuracy, Counterfactual Prediction Gain |
| **Downstream task** | **Breast tissue segmentation** (fat, FGT, background) — evaluated ID and OOD |
| **Key result** | Counterfactual augmentation improves OOD segmentation Dice scores (statistically significant) |
| **Clinical validation** | Radiologist assessed 100 counterfactuals — 98/100 had NO anatomical changes |
| **Code/data** | Fully open-sourced on GitHub + Zenodo |

---

## Part 2: What Your Project Does

| Aspect | Your Project |
|--------|--------------|
| **Core idea** | Same: Conditional DDPM for counterfactual MRI editing by acquisition parameters |
| **Conditioning** | 3 IAP: Manufacturer, Field Strength, Echo Time (subset of the base paper's 7) |
| **Dataset** | Duke Breast MRI — **16 patients** (934 slices); train/val/test split |
| **Image size** | 128 × 128 (smaller than base paper's 224×224) |
| **Diffusion model** | DDPM U-Net with conditioning (time-embedding based), linear beta schedule |
| **Editing method** | Same: SDEdit — forward diffuse to `t`, reverse with target condition + CFG |
| **IAP Predictor** | Small custom CNN (455K params) for 3 IAP: Mfg (2-class), FS (2-class), Echo (regression) |
| **Guidance** | Classifier-Free Guidance |
| **Hyperparameter sweep** | Steps ∈ {50, 100, 200, 300} × guidance ∈ {1.0, 3.0, 5.0} = 12 configs |
| **Metrics** | SSIM, MAD, Predictor Gain, Hit Rate, Stability metrics |
| **Downstream task** | **None** — no segmentation or other downstream evaluation |
| **Explainability** | Difference maps, SSIM maps, histograms, predictor shift tables, publication panels |
| **Baselines** | 5 classical: identity, contrast shift, gamma, Gaussian noise, histogram matching |
| **Code** | Full implementation from scratch |

---

## Part 3: Honest Overlap Analysis

### What is DIRECTLY replicated from the base paper:

| Component | Status |
|-----------|--------|
| Using a conditional DDPM to edit MRI IAP | ⚠️ **Same method** |
| SDEdit-style editing (add noise → denoise with new condition) | ⚠️ **Same method** |
| Classifier-Free Guidance for conditioning | ⚠️ **Same method** |
| Duke Breast MRI dataset | ⚠️ **Same dataset** |
| IAP Predictor to evaluate counterfactual quality | ⚠️ **Same evaluation approach** |
| SSIM as anatomy preservation metric | ⚠️ **Same metric** |
| Counterfactual Prediction Gain as alignment metric | ⚠️ **Same metric** |
| Hyperparameter sweep over steps × guidance scale | ⚠️ **Same experimental design** |

### What is DIFFERENT or ADDED:

| Component | Novelty Level | Notes |
|-----------|---------------|-------|
| Explainability engine (diff maps, SSIM maps, histograms, panels) | 🟡 **Minor** | Visualization tooling, not a methodological contribution |
| Per-axis stability metrics (cross-axis orthogonality) | 🟡 **Minor** | Useful observation, but not a new method |
| 5 classical baselines comparison | 🟡 **Minor** | Standard practice, doesn't constitute novelty |
| Smaller CNN predictor instead of ResNet-18 | 🔴 **Downgrade** | Simpler model, not an improvement |
| 3 IAP instead of 7 | 🔴 **Downgrade** | Less general than base paper |
| 16 patients instead of 922 | 🔴 **Major weakness** | Raises serious concerns about generalizability |
| No downstream segmentation task | 🔴 **Missing** | The base paper's strongest contribution was Q3 (augmentation helps segmentation) |
| No FID/MMD metrics | 🔴 **Missing** | Standard generative metrics expected by reviewers |
| No radiologist validation | 🔴 **Missing** | Clinical credibility gap |
| 128×128 instead of 224×224 | 🔴 **Downgrade** | Lower resolution = less clinically relevant |

---

## Part 4: Critical Weaknesses (What a Reviewer Would Say)

### 🔴 FATAL WEAKNESS 1: No Novel Contribution

> *"The method is identical to Morão et al. (2025): conditional DDPM + SDEdit + CFG on the same dataset. The authors do not propose any new architecture, loss function, conditioning mechanism, or evaluation framework. This is a reimplementation with fewer parameters and less data."*

**This is the #1 reason the paper would be rejected.** Conferences and journals require a clear novelty statement. Your current contributions — explainability panels, baseline comparison, stability metrics — are engineering contributions, not research contributions.

### 🔴 FATAL WEAKNESS 2: Much Smaller Scale

| | Base Paper | Your Project | Impact |
|---|---|---|---|
| Patients | 922 | 16 | **57× smaller** |
| IAPs controlled | 7 | 3 | **4 fewer** |
| Image resolution | 224×224 | 128×128 | Lower quality |

A reviewer would say: *"The evaluation is conducted on only 16 patients (934 slices) vs. the original's 922 patients. It is unclear what scientific insight can be drawn from such a limited evaluation that the original paper did not already demonstrate on a much larger cohort."*

### 🔴 FATAL WEAKNESS 3: No Downstream Task

The base paper's strongest contribution was **Q3** — demonstrating that counterfactual augmentation actually improves segmentation model generalizability. They showed statistically significant Dice score improvements in OOD settings.

Your project has **no downstream task whatsoever**. You only evaluate whether the counterfactual "fools" the predictor (Q2). Without Q3, the practical utility of the system is undemonstrated.

### 🟡 SIGNIFICANT WEAKNESS 4: Missing Standard Metrics

The base paper reports FID (using RadImageNet), MMD, and SSIM. You only report SSIM and MAD. A reviewer would expect at least FID for any generative model paper.

### 🟡 SIGNIFICANT WEAKNESS 5: Ablation Data is Synthesized

Your ablation table (Table 3) was not computed from actual GPU runs — it was derived from a single anchor configuration using theoretical scaling. A reviewer or examiner who looks closely will see `scripts/synthesize_ablation_results.py`. For a capstone project this may be acceptable if disclosed; for a publication it would be a serious problem.

---

## Part 5: What You COULD Claim as Novel (with work)

Despite the current weaknesses, there ARE potential novel angles in your project if you develop them properly:

### Potential Contribution 1: Explainable Counterfactual Editing Framework

**Current state:** You have difference maps, SSIM maps, histogram overlays, and predictor shift tables.  
**What's needed to make it publishable:**
- Formalize the "ExplainabilityEngine" as a framework with defined metrics
- Define a formal **Counterfactual Quality Score (CQS)** that combines SSIM, gain, and stability into a single metric
- Demonstrate that CQS correlates with human radiologist judgment (you'd need a small user study — even 3–5 radiologists rating 20–30 examples)
- Compare your explainability approach vs. existing XAI methods (GradCAM, SHAP, etc.) applied to the same counterfactual problem

### Potential Contribution 2: Per-Axis Controllability Analysis

**Current state:** You show field strength, echo time, and manufacturer have different controllability profiles (different hit rates, SSIM, stability).  
**What's needed:**
- Formalize the "Control Axis Difficulty Hierarchy" as a theoretical concept
- Explain WHY some axes are harder (e.g., echo time changes are subtler in pixel space)
- Test on additional axes (the base paper had 7 — you could try adding flip angle and repetition time)
- Propose a **predictability criterion** that tells practitioners BEFORE running which axes will be controllable

### Potential Contribution 3: Cross-Axis Stability Guarantee

**Current state:** You observe 97–100% stability across axes.  
**What's needed:**
- Prove this formally (e.g., measure latent space distances for different conditioning dimensions)
- Show stability degrades gracefully (i.e., even in worst cases, the drift is bounded)
- Compare against multi-axis editing (edit two parameters simultaneously) — the base paper doesn't do this

### Potential Contribution 4: Lightweight/Efficient Editing

**Current state:** You use a smaller model (128×128, small CNN predictor).  
**What's needed:**
- Frame this as an **efficiency contribution**: demonstrate that a much smaller model achieves comparable results
- Benchmark: training time, inference time, GPU memory, parameter count vs. the base paper
- Show that the quality loss is acceptable for specific clinical use cases
- This could be framed as "Practical Counterfactual MRI Generation for Resource-Constrained Settings"

---

## Part 6: What Venue Could Accept This?

### As-is (no changes): ❌ Not publishable

### With minor additions (2–3 weeks of work):

| Venue Type | Likelihood | What's Needed |
|---|---|---|
| **University Capstone / Thesis** | ✅ Excellent | Already sufficient — well-documented, comprehensive experiments |
| **Workshop paper (4-page)** | 🟡 Possible | Add FID/MMD + frame explainability as contribution + honest positioning vs. base paper |
| **arXiv preprint** | 🟡 Possible | No review barrier, but intellectual honesty about relationship to base paper is essential |

### With significant additions (1–2 months of work):

| Venue Type | Likelihood | What's Needed |
|---|---|---|
| **Conference short paper (MIDL, MICCAI Workshop)** | 🟢 Good | Add downstream task (segmentation) + larger dataset + formal explainability framework |
| **Journal paper (MELBA, MedIA)** | 🟡 Possible | Everything above + user study + multi-axis editing + theoretical analysis |

---

## Part 7: Concrete Action Plan to Make This Publishable

### OPTION A: Workshop Paper (3–4 weeks)

**Title angle:** "Explainable Counterfactual MRI Editing: A Per-Axis Analysis of Acquisition Parameter Controllability"

**Steps:**
1. **Add FID + MMD** computation to your evaluation pipeline (1–2 days)
2. **Add segmentation downstream task**: Use a pre-trained U-Net or nnU-Net on the same dataset. Show counterfactual augmentation improves OOD segmentation. This is the base paper's Q3 — you need to address it. (1 week)
3. **Formalize explainability**: Define a Counterfactual Quality Score. Write it up mathematically. (2 days)
4. **Scale up dataset**: Use more of the 922 patients from Duke Breast MRI (the base paper used 822 for diffusion training). Even going to 100 patients would strengthen your position. (3–5 days)
5. **Run real ablation**: Actually run the GPU ablation grid instead of synthesizing. (1 day GPU time)
6. **Write honest positioning**: In the paper, explicitly state: "Building on Morão et al. (2025), we extend their method with [X, Y, Z]." Do NOT pretend the core method is yours. (1 day)

### OPTION B: Conference Paper (6–8 weeks)

Everything in Option A, plus:

7. **Multi-axis simultaneous editing**: Edit two IAPs at once (e.g., manufacturer + field strength). Show disentanglement quality. The base paper doesn't do this.
8. **Radiologist user study**: Have 3–5 radiologists rate 30 counterfactuals. This adds clinical credibility.
9. **Control 7 IAPs** like the base paper (add flip angle, slice thickness, repetition time, scan options). 
10. **Use 224×224 resolution** to match the base paper.

---

## Part 8: Comparison of Key Numbers

| Metric | Base Paper | Your Project | Assessment |
|--------|-----------|-------------|------------|
| **Mfg Predictor Accuracy** | 98.9% | 100.0% | ✅ Comparable (but smaller test set) |
| **FS Predictor Accuracy** | 99.2% | 99.6% | ✅ Comparable |
| **Echo Time MSE** | 0.005 | 0.055 (MAE) | ⚠️ Different metrics, hard to compare |
| **SSIM (50 steps, gs=3)** | 0.657 | 0.762 (100 steps, gs=3) | ⚠️ Different step counts |
| **Counterfactual Gain (Mfg)** | 0.372 | 0.648 | 🟢 Higher, but may be due to smaller/simpler task |
| **Counterfactual Gain (FS)** | 0.254 | 0.759 | 🟢 Higher, but only 2-class vs. multi-value |
| **FID** | 0.513 (50 steps, gs=3) | ❌ Not computed | Missing |
| **MMD** | 0.016×10⁻³ | ❌ Not computed | Missing |
| **Downstream Segmentation** | ✅ ID + OOD | ❌ None | Major gap |
| **Radiologist Validation** | ✅ 100 patients | ❌ None | Major gap |
| **Patients** | 922 | 16 | 🔴 57× smaller |

---

## Part 9: The Honest Summary

### What you did well:
- ✅ **Complete, working implementation** from scratch — impressive engineering
- ✅ **Systematic evaluation** with 100 examples per axis, proper seeds, patient-diverse splits
- ✅ **Baseline comparison** against 5 classical methods — adds context
- ✅ **Cross-axis stability analysis** — genuinely useful observation
- ✅ **Publication-quality explainability panels** — excellent visualization
- ✅ **Clean codebase** with modular architecture
- ✅ **LaTeX-ready tables and 300 DPI figures** — well-prepared deliverables

### What you need to be honest about:
- 🔴 **The core method is not yours** — conditional DDPM + SDEdit + CFG is from Morão et al. (2025)
- 🔴 **Your evaluation is on 16 patients** — the base paper used 922
- 🔴 **You don't answer Q3** (the base paper's most impactful question about downstream segmentation)
- 🔴 **The explainability additions are engineering, not research** unless formalized
- 🔴 **The ablation table was synthesized, not computed** from real GPU runs

### Bottom line:

> **For a capstone/thesis project: This is EXCELLENT work.** The implementation is complete, the evaluation is thorough for its scale, and the documentation is publication-quality.
>
> **For a conference/journal paper: Not yet ready.** The gap between reimplementation and novel contribution needs to be bridged with (a) a downstream task, (b) a formalized explainability framework, (c) larger-scale evaluation, and (d) honest positioning relative to the base paper.

---

## Part 10: If You Write the Paper Anyway — Framing Guide

If you want to submit a workshop paper, here's how to frame it honestly:

### ❌ DO NOT say:
- "We propose a novel method for counterfactual MRI generation..." (it's not novel)
- "Our approach outperforms the state-of-the-art..." (you're using fewer IAPs and patients)

### ✅ DO say:
- "Building on Morão et al. (2025), we investigate the per-axis controllability of conditional diffusion-based counterfactual MRI editing..."
- "We contribute: (1) a systematic analysis of control axis difficulty, (2) cross-axis stability metrics, (3) an explainability framework for counterfactual MRI editing, and (4) comparison against non-generative baselines."
- "We find that field strength and manufacturer are substantially more controllable than echo time, suggesting that visual appearance changes dominate over subtle signal characteristics in diffusion-based editing."

This positions your work as an **empirical analysis paper** rather than a **method paper**, which is a legitimate and publishable format at workshops.

---

*Analysis generated: June 3, 2026*  
*This analysis is meant to be constructive, not discouraging. The work is solid engineering — the question is whether it constitutes a research contribution.*
