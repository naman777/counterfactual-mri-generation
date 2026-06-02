# Phase 10 + Phase 12 Implementation Plan
## Explainable Counterfactual MRI — Paper Publication Track

---

## Overview

Phase 10 = Explainability module integration (makes the system complete)
Phase 12 = Quantitative evaluation experiments (makes the paper defensible)

These two phases must be done in order. Phase 12 depends on Phase 10 being stable.

Estimated time:
- Phase 10: 4–5 days
- Phase 12: 4–5 days
- Total: ~9–10 working days to paper-ready results

---

# PHASE 10: Explainability Module — Full Integration

## What Phase 10 Must Produce for the Paper

You already have individual explainability functions from Phase 7:
- `src/explainability/difference_map.py`
- `src/explainability/ssim_map.py`
- `src/explainability/histogram.py`

What Phase 10 adds is:
1. A unified `ExplainabilityEngine` class that wraps all four functions
2. A `predictor_shift` function that runs the IAP predictor before and after
3. A `generate_panel()` function that produces a publication-quality figure
4. A batch evaluation runner that saves panels + JSON metrics for all test examples
5. A validation summary that aggregates results across the full test set

---

## Phase 10 — Step-by-Step

### Step 10.1 — Build the Unified ExplainabilityEngine

**File:** `src/explainability/engine.py`

This class takes an original image and a counterfactual image as inputs and
returns all four explainability outputs in one call.

```python
class ExplainabilityEngine:
    def __init__(self, predictor_checkpoint, label_maps_dir, device='cuda'):
        # Load predictor
        # Load label maps for manufacturer/field_strength decoding
        pass

    def compute_all(self, original, counterfactual,
                    source_condition, target_condition):
        """
        Inputs:
            original: numpy array [H, W] normalized [0,1]
            counterfactual: numpy array [H, W] normalized [0,1]
            source_condition: dict with keys manufacturer, field_strength, echo_time_norm
            target_condition: dict with keys manufacturer, field_strength, echo_time_norm

        Returns:
            results: dict with keys:
                diff_map         -> numpy [H, W]
                ssim_map         -> numpy [H, W]
                global_ssim      -> float
                mean_abs_diff    -> float
                histogram_data   -> dict (bins, orig_hist, cf_hist)
                hist_l1          -> float
                hist_intersection-> float
                predictor_before -> dict (mfg_pred, fs_pred, echo_pred)
                predictor_after  -> dict (mfg_pred, fs_pred, echo_pred)
                predictor_gain   -> dict (mfg_gain, fs_gain, echo_improvement)
                target_hit       -> dict (mfg_hit, fs_hit, echo_moved_closer)
        """
        pass

    def _run_predictor(self, image):
        # Convert numpy image to tensor
        # Run predictor forward pass
        # Return decoded predictions + raw probabilities
        pass

    def _compute_predictor_gain(self, before, after, target_condition):
        # For categorical: gain = after_target_prob - before_target_prob
        # For continuous: improvement = abs(before_error) - abs(after_error)
        pass
```

**Definition of done:** `engine.compute_all()` runs on a single pair and returns
all keys without errors.

---

### Step 10.2 — Build the Publication Panel Generator

**File:** `src/explainability/panel_generator.py`

This is the most important output for your paper figures.

```python
def generate_publication_panel(
    original,
    counterfactual,
    results,           # output of ExplainabilityEngine.compute_all()
    source_condition,
    target_condition,
    save_path,
    title=None
):
    """
    Layout (1 row, 6 columns):
    [Original] [Counterfactual] [Diff Map] [SSIM Map] [Histogram] [Predictor Table]

    All images at same scale.
    Diff map: red heatmap (0=no change, bright=large change)
    SSIM map: grayscale (bright=high similarity preserved)
    Histogram: overlaid original (blue) vs counterfactual (orange)
    Predictor table: text panel showing before/after/gain for all three controls
    """
```

Panel specifications for paper quality:
- Figure DPI: 300
- Image size per panel: 128x128 displayed at 2x for visibility
- Colormap for diff map: 'hot' or 'Reds'
- Colormap for SSIM map: 'gray'
- Font size: 8pt for labels, 10pt for titles
- Save as: `.png` (for reports) and `.pdf` (for paper submission)
- Background: white

The predictor table panel should show:

```
Control        | Before      | After       | Target      | Gain
Manufacturer   | GE          | SIEMENS     | SIEMENS     | +0.999
Field Strength | 1.5T        | 1.5T        | 1.5T        | stable
EchoTime norm  | 0.706       | 0.256       | 0.150       | -0.431
```

**Definition of done:** Running `generate_publication_panel()` on your Phase 9
best-case examples produces a figure you would be comfortable putting in a paper.

---

### Step 10.3 — Build the Batch Panel Runner

**File:** `scripts/run_explainability_batch.py`

This script runs the explainability engine on a list of test examples and saves:
- One panel per example
- One JSON metrics file per example
- One aggregated summary JSON

```
python scripts/run_explainability_batch.py \
    --test_csv data/processed/ax_t1_128/splits/test.csv \
    --counterfactual_dir reports/counterfactuals/phase12/ \
    --output_dir reports/explainability/batch/ \
    --predictor_checkpoint checkpoints/predictor/ax_t1_128/best.pt \
    --label_maps_dir data/processed/ax_t1_128/label_maps/
```

Output structure:
```
reports/explainability/batch/
├── panels/
│   ├── field_strength/
│   │   ├── example_001_panel.png
│   │   ├── example_001_panel.pdf
│   │   ├── example_001_metrics.json
│   │   └── ...
│   ├── echo_time/
│   └── manufacturer/
├── summaries/
│   ├── field_strength_summary.json
│   ├── echo_time_summary.json
│   └── manufacturer_summary.json
└── paper_figures/
    ├── fig_field_strength_best3.pdf
    ├── fig_echo_time_best3.pdf
    └── fig_manufacturer_tradeoff.pdf
```

**Definition of done:** All test examples produce panels and the summary JSON
files contain aggregated mean ± std for all metrics.

---

### Step 10.4 — Build the Paper Figure Selector

**File:** `scripts/select_paper_figures.py`

This script automatically ranks your test examples by quality and selects the
best N for inclusion in the paper.

Selection criteria per control axis:

For field strength:
- Sort by: target_field_gain DESC, then global_ssim DESC
- Select top 3 examples

For EchoTime:
- Sort by: echo_improvement DESC, then global_ssim DESC
- Select top 3 examples

For manufacturer:
- Select one "balanced" example (moderate gain, moderate SSIM)
- Select one "high-gain" example (highest gain regardless of SSIM)
- This honestly represents the tradeoff

```python
python scripts/select_paper_figures.py \
    --summary_dir reports/explainability/batch/summaries/ \
    --output_dir reports/paper_figures/ \
    --n_per_axis 3
```

This script also produces a combined multi-row figure:
- Row 1: Field strength best case
- Row 2: EchoTime best case
- Row 3: Manufacturer balanced case
- Row 4: Manufacturer high-gain case

This becomes Figure 2 or Figure 3 in your paper.

**Definition of done:** `paper_figures/combined_main_figure.pdf` exists and
looks publication-quality.

---

### Step 10.5 — Validate the Predictor Shift Logic

Before Phase 12, manually verify the predictor shift computation on 5 examples.

For each example print:
```
Original image predictor output:
  Manufacturer: GE (prob=0.9987)
  Field Strength: 1.5T (prob=0.9991)
  EchoTime norm: 0.7058

Counterfactual image predictor output:
  Manufacturer: GE (prob=0.9832)       <- stayed stable, good
  Field Strength: 3.0T (prob=0.9999)   <- shifted to target, good
  EchoTime norm: 0.6821                <- small change, acceptable

Target condition was:
  Manufacturer: GE (same)
  Field Strength: 3.0T
  EchoTime: same
```

Check that:
- Categorical gain = after_target_prob - before_target_prob is positive
  when editing worked
- EchoTime improvement = abs(before - target) - abs(after - target) is
  positive when editing worked
- Unchanged controls have near-zero gain (stability check)

If any of these fail, fix the logic before running Phase 12.

**Definition of done:** Manual verification on 5 examples matches expected
behavior for all three axes.

---

# PHASE 12: Quantitative Evaluation — Paper Results

## What Phase 12 Must Produce for the Paper

Phase 12 must produce three things:

1. **Main results table** — systematic metrics across 50–100 test examples
   for all three control axes
2. **Baseline comparison table** — your diffusion results vs. histogram
   matching, contrast shift, and identity baselines on the same metrics
3. **Ablation table** — effect of edit timestep and guidance scale on the
   quality-alignment tradeoff

These three tables are the core of your results section.

---

## Phase 12 — Step-by-Step

### Step 12.1 — Define the Evaluation Protocol

Before writing any code, lock the evaluation settings.

**Test set:** All 16 test patients (934 slices). For each experiment, sample
up to 100 examples ensuring patient-level diversity (max 10 slices per patient).

**Fixed settings per axis (from your Phase 9 best results):**

| Control | Edit Timestep | Guidance Scale | Justification |
|---|---|---|---|
| Field Strength | 100 | 3.0 | Best SSIM + gain tradeoff |
| EchoTime | 100 | 1.0 | Best improvement + SSIM |
| Manufacturer | 300 | 5.0 | Balanced tradeoff point |

**Metrics to report for every example:**

```
global_ssim
mean_absolute_difference
max_absolute_difference
hist_l1_distance
hist_intersection
target_categorical_gain   (for manufacturer / field strength)
target_hit_after_edit     (binary: did predictor switch to target?)
echo_time_improvement     (for EchoTime only)
echo_moved_closer         (binary)
mfg_stability             (did manufacturer stay when not being edited?)
fs_stability              (did field strength stay when not being edited?)
```

---

### Step 12.2 — Build the Main Evaluation Runner

**File:** `scripts/run_phase12_evaluation.py`

This is the core Phase 12 script. It runs counterfactual editing on N test
examples for a given control axis and saves all metrics.

```
python scripts/run_phase12_evaluation.py \
    --control field_strength \
    --test_csv data/processed/ax_t1_128/splits/test.csv \
    --diffusion_checkpoint checkpoints/diffusion/ax_t1_128/latest.pt \
    --predictor_checkpoint checkpoints/predictor/ax_t1_128/best.pt \
    --label_maps_dir data/processed/ax_t1_128/label_maps/ \
    --output_dir reports/phase12/field_strength/ \
    --n_examples 100 \
    --edit_timestep 100 \
    --guidance_scale 3.0 \
    --seed 42
```

Run this three times, once per axis:
```
--control field_strength  --edit_timestep 100 --guidance_scale 3.0
--control echo_time       --edit_timestep 100 --guidance_scale 1.0
--control manufacturer    --edit_timestep 300 --guidance_scale 5.0
```

Internal logic of the runner:

```python
for example_idx, row in enumerate(test_examples):
    # 1. Load original image
    # 2. Get source condition from metadata
    # 3. Build target condition (flip the control being tested)
    # 4. Run counterfactual_editor.edit()
    # 5. Run ExplainabilityEngine.compute_all()
    # 6. Save metrics to JSON
    # 7. Save panel image
    # 8. Log progress

# After loop:
# Aggregate all JSON files into summary CSV
# Compute mean, std, median for all metrics
# Save summary JSON and CSV
```

**Target condition construction logic:**

For field_strength:
```python
# Flip field strength, keep everything else the same
if source['field_strength'] == '1.5T':
    target['field_strength'] = '3.0T'
else:
    target['field_strength'] = '1.5T'
target['manufacturer'] = source['manufacturer']  # unchanged
target['echo_time_norm'] = source['echo_time_norm']  # unchanged
```

For echo_time:
```python
# Use the opposite end of the EchoTime range
echo_range_min = 0.0   # from your label maps
echo_range_max = 1.0
if source['echo_time_norm'] > 0.5:
    target['echo_time_norm'] = 0.15  # move to low end
else:
    target['echo_time_norm'] = 0.85  # move to high end
target['manufacturer'] = source['manufacturer']
target['field_strength'] = source['field_strength']
```

For manufacturer:
```python
# Flip manufacturer
if source['manufacturer'] == 'GE MEDICAL SYSTEMS':
    target['manufacturer'] = 'SIEMENS'
else:
    target['manufacturer'] = 'GE MEDICAL SYSTEMS'
target['field_strength'] = source['field_strength']
target['echo_time_norm'] = source['echo_time_norm']
```

**Definition of done:** Three output directories exist, each with 100 example
metrics JSON files and one summary CSV.

---

### Step 12.3 — Build the Baseline Comparison Runner

**File:** `scripts/run_phase12_baselines.py`

This runs the same test examples through your Phase 7 baseline transformations
(identity, contrast_shift, gamma, gaussian_noise, histogram_matching) and
computes the same metrics so you can fill a comparison table.

```
python scripts/run_phase12_baselines.py \
    --test_csv data/processed/ax_t1_128/splits/test.csv \
    --predictor_checkpoint checkpoints/predictor/ax_t1_128/best.pt \
    --label_maps_dir data/processed/ax_t1_128/label_maps/ \
    --output_dir reports/phase12/baselines/ \
    --n_examples 100 \
    --seed 42
```

Important: use the exact same 100 examples and same seed as the main
evaluation runner so the comparison is fair.

For each baseline, the "target condition" is the same as the diffusion
experiments. This lets you compare:

| Method | SSIM | MAD | Target Hit Rate | Predictor Gain |
|---|---|---|---|---|
| Identity | 1.000 | 0.000 | 0% | 0.000 |
| Contrast Shift | ~0.993 | ~0.014 | 0% | ~0.000 |
| Gamma | ~0.959 | ~0.016 | 0% | ~0.000 |
| Gaussian Noise | ~0.740 | ~0.020 | 0% | ~0.000 |
| Histogram Match | ~0.744 | ~0.066 | ~15–30%? | small |
| **Ours (Diffusion)** | **~0.75** | **~0.03** | **100%** | **~0.99** |

The key insight this table will show: only the diffusion model achieves
consistent target hit. Baselines may accidentally shift appearance, but they
do not controllably reach the target acquisition condition.

**Definition of done:** `reports/phase12/baselines/baseline_summary.csv`
exists with all baselines and all metrics.

---

### Step 12.4 — Build the Ablation Table

**File:** `scripts/run_phase12_ablation.py`

This runs field strength editing (your strongest axis) across a grid of
edit timesteps and guidance scales on a smaller subset (20–30 examples)
to produce the ablation table.

```
python scripts/run_phase12_ablation.py \
    --test_csv data/processed/ax_t1_128/splits/test.csv \
    --diffusion_checkpoint checkpoints/diffusion/ax_t1_128/latest.pt \
    --predictor_checkpoint checkpoints/predictor/ax_t1_128/best.pt \
    --output_dir reports/phase12/ablation/ \
    --n_examples 25 \
    --control field_strength \
    --timestep_grid 100 200 300 400 \
    --guidance_grid 1.0 3.0 5.0 \
    --seed 42
```

This produces a 4×3 = 12 configuration table:

| Edit Steps | Guidance | SSIM↑ | MAD↓ | Target Gain↑ | Hit Rate↑ |
|---|---|---|---|---|---|
| 100 | 1.0 | | | | |
| 100 | 3.0 | | | | |
| 100 | 5.0 | | | | |
| 200 | 1.0 | | | | |
| ... | ... | | | | |
| 400 | 5.0 | | | | |

This becomes Table 2 or Table 3 in your paper and directly supports the
"tradeoff characterization" contribution claim.

**Definition of done:** `reports/phase12/ablation/ablation_table.csv` exists
with all 12 configurations filled.

---

### Step 12.5 — Build the Results Aggregation Script

**File:** `scripts/aggregate_phase12_results.py`

This script reads all Phase 12 output directories and produces the final
tables formatted for the paper.

```
python scripts/aggregate_phase12_results.py \
    --phase12_dir reports/phase12/ \
    --output_dir reports/paper_tables/
```

Output files:
```
reports/paper_tables/
├── table1_main_results.csv        <- mean ± std for all axes
├── table2_baseline_comparison.csv <- diffusion vs baselines
├── table3_ablation.csv            <- timestep x guidance grid
└── table_latex/
    ├── table1.tex
    ├── table2.tex
    └── table3.tex
```

The LaTeX files should be directly pasteable into your paper.

Format for table1_main_results.csv:

```
Control         | SSIM mean | SSIM std | MAD mean | MAD std | Hit Rate | Gain mean
Field Strength  | 0.747     | 0.031    | 0.034    | 0.008   | 1.00     | 0.9915
EchoTime        | 0.697     | 0.044    | 0.041    | 0.011   | 1.00     | 0.1294
Manufacturer    | 0.412     | 0.089    | 0.114    | 0.031   | 0.64     | 0.6800
```

**Definition of done:** All three LaTeX tables are generated and the numbers
match what you computed in Steps 12.2, 12.3, and 12.4.

---

### Step 12.6 — Compute FID and MMD (Optional but Recommended)

The base paper reports FID and MMD. If you want direct comparability, add
these two metrics.

**FID:** Use MONAI's `RadImageNet` feature extractor (same as base paper) to
compute FID between:
- Real test images at target condition
- Counterfactual images edited toward that condition

**MMD:** Already implemented in Phase 7 baseline work. Reuse.

File: `scripts/compute_fid_mmd.py`

Note: FID requires a minimum of ~100 samples to be meaningful. If your test
set is smaller per condition, report it with a caveat or skip it and note
that you follow the predictor-based validation approach from Konz and
Mazurowski 2024 instead.

---

### Step 12.7 — Produce the Tradeoff Scatter Plot for the Paper

**File:** `scripts/plot_tradeoff_figure.py`

This produces Figure 1 (or Figure 2) for your paper: the anatomy preservation
vs. target alignment tradeoff scatter plot, one per control axis.

X-axis: Global SSIM (anatomy preservation proxy)
Y-axis: Predictor target gain (target alignment)
Each point: one test example
Color: edit timestep
Marker shape: guidance scale

You already have a version of this from Phase 9. Phase 12 adds it across
100 examples instead of 5–10, making it paper-quality.

```
python scripts/plot_tradeoff_figure.py \
    --results_dir reports/phase12/ \
    --output_dir reports/paper_figures/ \
    --controls field_strength echo_time manufacturer
```

Output:
```
reports/paper_figures/
├── fig_tradeoff_field_strength.pdf
├── fig_tradeoff_echo_time.pdf
├── fig_tradeoff_manufacturer.pdf
└── fig_tradeoff_combined_3panel.pdf   <- main paper figure
```

**Definition of done:** `fig_tradeoff_combined_3panel.pdf` looks
publication-quality at 300 DPI.

---

## Execution Order (Day-by-Day)

### Days 1–2: Phase 10 Core

Day 1:
- Build `src/explainability/engine.py` (Step 10.1)
- Test on 3 examples manually (Step 10.5 validation)

Day 2:
- Build `src/explainability/panel_generator.py` (Step 10.2)
- Verify output on your Phase 9 best-case examples
- Output should be paper-quality panels

### Days 3–4: Phase 10 Batch Infrastructure

Day 3:
- Build `scripts/run_explainability_batch.py` (Step 10.3)
- Run on all Phase 9 counterfactual outputs to generate panels

Day 4:
- Build `scripts/select_paper_figures.py` (Step 10.4)
- Produce combined main figure draft
- Review: does Figure X look like something you'd submit?

### Days 5–7: Phase 12 Evaluation

Day 5:
- Build `scripts/run_phase12_evaluation.py` (Step 12.2)
- Run field_strength evaluation on 100 examples
- This will take several hours — start it early in the day

Day 6:
- Run echo_time evaluation on 100 examples
- Run manufacturer evaluation on 100 examples in parallel if possible
- Start baseline runner (Step 12.3)

Day 7:
- Run ablation grid (Step 12.4) on 25 examples
- All evaluation jobs should complete by end of day

### Days 8–9: Aggregation and Paper Figures

Day 8:
- Build and run `scripts/aggregate_phase12_results.py` (Step 12.5)
- Review all three tables — check numbers make sense
- Run `scripts/plot_tradeoff_figure.py` (Step 12.7)

Day 9:
- Run `scripts/select_paper_figures.py` to finalize paper figures
- Optional: compute FID/MMD (Step 12.6)
- Final review: do you have three tables and four figures ready?

### Day 10: Paper Writing Starts

At this point you should have:
- Table 1: Main results (three axes, mean ± std)
- Table 2: Baseline comparison
- Table 3: Ablation (timestep × guidance)
- Figure A: Combined explainability panel (best cases per axis)
- Figure B: Tradeoff scatter plot (three panels)
- All panels saved as PDF at 300 DPI

---

## What Your Paper Results Section Looks Like After This

**Section 4.1 — Quantitative Results**

Table 1 goes here. One sentence per row interpreting the key number.
Field strength: near-perfect target alignment, acceptable anatomy preservation.
EchoTime: reliable directional control, moderate SSIM cost.
Manufacturer: strong alignment possible but requires higher edit strength.

**Section 4.2 — Comparison with Non-Generative Baselines**

Table 2 goes here. Key claim: only the diffusion model achieves consistent
target acquisition alignment. Baselines achieve appearance changes but not
controlled acquisition-parameter targeting.

**Section 4.3 — Hyperparameter Ablation**

Table 3 goes here. Key claim: edit timestep 100 with guidance scale 3.0
gives the best tradeoff for field strength. Higher timesteps improve target
alignment at the cost of anatomy preservation. This directly maps to Figure B.

**Section 4.4 — Qualitative Explainability Analysis**

Figure A goes here. Walk through one field strength example, one EchoTime
example, one manufacturer example using the panel layout. Describe what
the difference map, SSIM map, histogram, and predictor table each tell you.

---

## Critical Failure Points to Watch For

**CFP-1: Predictor gives wrong output on counterfactuals**
If the predictor always outputs the source condition regardless of the
counterfactual, your gain numbers will be near zero. Check this on 5 examples
before running 100. If this happens, the issue is that the counterfactual
image is not changing the visual features the predictor uses. Fix: use higher
edit timestep or guidance scale for that axis.

**CFP-2: SSIM is computed on wrong scale**
SSIM must be computed on images normalized to [0, 1]. If you accidentally
run it on [-1, 1] tensors (the diffusion training scale), SSIM will be
meaningless. Add an explicit assert in the engine that input is in [0, 1].

**CFP-3: Test set leakage into ablation**
The ablation (Step 12.4) must use the same patient-level test split. Do not
use validation set patients for ablation. Lock the seed and patient list.

**CFP-4: Panel figures are too small to read**
Medical imaging paper figures need to be readable at single-column width
(~86mm) and double-column width (~178mm). Test your PDF at 100% zoom in
a PDF reader before considering it final.

**CFP-5: Baseline comparison uses different examples**
The 100 examples in the baseline runner must be the exact same examples as
the diffusion runner. Use identical seed and selection logic.