Below is a complete **Phase 9 report** in the same style as your previous phase reports. You can copy this into your final document. I have left image placeholders and written the exact paths where you should add images manually.

---

# Phase 9 Completion Report

## Counterfactual MRI Generation and Acquisition Parameter Editing

**Project:** Explainable Counterfactual MRI Generation and Acquisition Parameter Control for Robust Medical Imaging AI
**Current Completed Phase:** Phase 9 / Step 9–10
**Main Work Completed:** Counterfactual editing prototype, predictor validation, tuning experiments, explainability panels, and Phase 9 report generation.

---

## 1. Project Context

The main goal of this project is to build an explainable and controllable MRI counterfactual generation system.

The final system should allow a user to select an MRI slice and change acquisition-related parameters such as:

1. Scanner manufacturer
2. Magnetic field strength
3. EchoTime

The system should then generate a counterfactual MRI image that tries to change acquisition-related appearance while preserving the main anatomy.

Before Phase 9, we had already completed:

* Metadata extraction and cleaning
* ax t1 subset selection
* Image preprocessing to 128×128
* Patient-level train/val/test split
* Acquisition-parameter predictor
* Classical baselines and explainability functions
* Conditional diffusion model training
* Synthetic MRI-like sampling from random noise

In Phase 8, the diffusion model was trained and tested for conditional image generation. The selected checkpoint was:

```text
checkpoints/diffusion/ax_t1_128/latest.pt
```

This checkpoint was selected because visually it generated better MRI-like samples than the lower-validation-loss `best.pt`.

Phase 9 moved the project from **synthetic generation** to **real counterfactual editing**.

In simple terms:

```text
Phase 8:
Random noise → conditional diffusion → synthetic MRI-like image

Phase 9:
Real MRI slice → partial noising → target-conditioned denoising → counterfactual MRI
```

This is a major milestone because Phase 9 is the first time the model edits a real input image instead of only generating from random noise.

---

## 2. Objective of Phase 9

The main objective of Phase 9 was to build and test the first counterfactual editing prototype.

The target workflow was:

```text
Original MRI slice
→ Add controlled partial noise
→ Denoise using target acquisition condition
→ Generate counterfactual image
→ Validate with predictor
→ Explain with difference map, local SSIM map, and histogram
```

The system should answer these questions:

1. Can the diffusion model edit a real MRI slice?
2. Does the output preserve anatomy?
3. Does the output move toward the requested target condition?
4. Which control works best: manufacturer, field strength, or EchoTime?
5. Can we explain the changes visually and quantitatively?

---

## 3. Main Concept Used in Phase 9

The key method used in Phase 9 is **partial noising + target-conditioned denoising**.

Instead of starting from pure random noise, we start from the original MRI image.

The process is:

```text
Original image x0
→ add noise up to selected timestep t
→ get noisy image xt
→ denoise xt using target condition
→ get counterfactual image
```

This is important because starting from the original image helps preserve anatomy better than generating a completely new image from random noise.

The edit strength is controlled by:

```text
edit_timestep
```

Meaning:

```text
Lower timestep = lighter edit, better anatomy preservation, weaker target change
Higher timestep = stronger edit, stronger target change, more anatomy risk
```

The target-conditioning strength is controlled by:

```text
guidance_scale
```

Meaning:

```text
Lower guidance = safer output, weaker target push
Higher guidance = stronger target push, more risk of artifacts
```

---

## 4. Files Created in Phase 9

During Phase 9, we created and tested the following files:

```text
scripts/phase9_step1_check.py
src/inference/counterfactual_editor.py
scripts/phase9_step2_reconstruction_test.py
scripts/phase9_step3_manufacturer_edit.py
scripts/phase9_step4_predictor_check.py
src/explainability/validation.py
scripts/phase9_step5_validate_manufacturer_edits.py
scripts/phase9_step6_manufacturer_tuning_grid.py
scripts/phase9_step7_field_strength_edit.py
scripts/phase9_step8_echo_time_edit.py
scripts/phase9_step9_final_explainability_panels.py
scripts/phase9_step10_generate_report.py
```

These files together created the full Phase 9 pipeline:

```text
Compatibility checking
→ Counterfactual generation engine
→ Reconstruction sanity test
→ Manufacturer edit
→ Predictor validation
→ Manufacturer tuning
→ Field strength edit
→ EchoTime edit
→ Final explainability panels
→ Final Phase 9 report
```

---

## 5. Step-by-Step Journey of Phase 9

---

# Step 1: Repository and Checkpoint Compatibility Check

## What We Did

We first created:

```text
scripts/phase9_step1_check.py
```

This script checked whether all required files existed before implementing counterfactual editing.

It verified:

```text
configs/diffusion.yaml
checkpoints/diffusion/ax_t1_128/latest.pt
checkpoints/diffusion/ax_t1_128/best.pt
checkpoints/predictor/ax_t1_128/best.pt
data/processed/ax_t1_128/splits/test.csv
data/processed/ax_t1_128/processed_metadata.csv
label maps
diffusion model files
scheduler file
sampling file
```

It also inspected the model class signatures:

```text
AcquisitionConditionEmbedding
MRIConditionalUNet
DDPMScheduler
```

## Result

Everything was found correctly.

Important confirmed details:

```text
Test set size: 934 samples
Image shape: 128 × 128
Image range: [0, 1]
Manufacturer labels:
  GE MEDICAL SYSTEMS = 0
  SIEMENS = 1

Field strength labels:
  1.5T = 0
  3.0T = 1

EchoTime normalization:
  min = 7.88
  max = 12.272
```

The diffusion checkpoint had:

```text
unet_state_dict
conditioning_state_dict
epoch = 10
global_step = 11080
```

This confirmed that Phase 9 could safely use:

```text
checkpoints/diffusion/ax_t1_128/latest.pt
```

## Why This Step Was Important

This step prevented implementation errors. Before writing new counterfactual code, we needed to know:

* How checkpoint keys are stored
* What the scheduler function is called
* What input shape the model expects
* Which labels are used for manufacturer and field strength
* Whether the test images are accessible

This was important because a small mismatch in model constructor, checkpoint key, or scheduler function could break the entire Phase 9 pipeline.

---

# Step 2: Counterfactual Editor and Reconstruction Sanity Test

## What We Did

We created the main counterfactual generation engine:

```text
src/inference/counterfactual_editor.py
```

This file contains the reusable class:

```text
CounterfactualEditor
```

It handles:

```text
Loading diffusion checkpoint
Loading image
Converting image from [0,1] to [-1,1]
Adding partial noise
Denoising under target condition
Converting output back to [0,1]
Saving generated images
```

Then we created:

```text
scripts/phase9_step2_reconstruction_test.py
```

This tested reconstruction using the **same condition** as the original image.

Meaning:

```text
Original condition = Target condition
```

So the model was not supposed to change manufacturer, field strength, or EchoTime.

## Result

The reconstruction sanity test gave:

```text
Global SSIM = 0.8126
Mean absolute difference = 0.0249
Max absolute difference = 0.5371
```

The image was not identical, but it preserved the main anatomy.

The output was slightly smoother and had some texture differences, which is expected because DDPM reverse denoising is stochastic.

## Image to Add

Add this image manually:

```text
reports/counterfactuals/phase9_step2_reconstruction/comparison_original_reconstruction_diff.png
```

## Why This Step Was Important

Before changing any acquisition condition, we needed to verify that:

```text
Original image → partial noise → denoise with same condition
```

does not destroy the image.

If same-condition reconstruction failed badly, then actual counterfactual editing would not be reliable.

Since the reconstruction gave SSIM around `0.81`, we considered it acceptable for a first prototype.

---

# Step 3: First Manufacturer Counterfactual Edit

## What We Did

We created:

```text
scripts/phase9_step3_manufacturer_edit.py
```

This tested manufacturer transfer:

```text
GE → SIEMENS
SIEMENS → GE
```

Only manufacturer was changed.

These were kept the same:

```text
field_strength_label
echo_time_norm
```

We tested:

```text
edit_timestep = 100, 200
guidance_scale = 1.0, 3.0
```

## Result

For GE → SIEMENS:

```text
t=100, gs=1.0 | SSIM=0.8144 | MAD=0.0250
t=100, gs=3.0 | SSIM=0.8007 | MAD=0.0262
t=200, gs=1.0 | SSIM=0.7434 | MAD=0.0352
t=200, gs=3.0 | SSIM=0.6911 | MAD=0.0405
```

For SIEMENS → GE:

```text
t=100, gs=1.0 | SSIM=0.7797 | MAD=0.0246
t=100, gs=3.0 | SSIM=0.7594 | MAD=0.0261
t=200, gs=1.0 | SSIM=0.6693 | MAD=0.0341
t=200, gs=3.0 | SSIM=0.6181 | MAD=0.0371
```

## Interpretation

The light settings preserved anatomy better:

```text
t=100, gs=1.0
t=100, gs=3.0
```

The stronger settings created more visible changes, but anatomy started to shift more.

At this stage, visual editing was working, but we still needed to check whether the predictor actually moved toward the target manufacturer.

## Images to Add

Add these manually:

```text
reports/counterfactuals/phase9_step3_manufacturer_edit/grids/example_00_manufacturer_edit_grid.png
reports/counterfactuals/phase9_step3_manufacturer_edit/grids/example_01_manufacturer_edit_grid.png
```

---

# Step 4: Predictor Compatibility Check

## What We Did

Before validating generated images with the acquisition predictor, we created:

```text
scripts/phase9_step4_predictor_check.py
```

This inspected the predictor architecture, checkpoint keys, and output format.

## Result

The predictor class was confirmed as:

```text
AcquisitionPredictor
```

The checkpoint key was:

```text
model_state_dict
```

The predictor output keys were:

```text
manufacturer_logits
field_strength_logits
echo_time_norm
```

The predictor input format was confirmed as:

```text
[B, 1, 128, 128]
range = [0, 1]
```

This was very important because the diffusion model uses `[-1, 1]`, but the predictor uses `[0, 1]`.

## Why This Step Was Important

The acquisition predictor is the automatic validation module for counterfactual generation.

It tells us:

```text
Before generation:
  predictor(original image)

After generation:
  predictor(counterfactual image)

Then compare:
  Did prediction move toward target?
```

Without this step, we could only visually inspect the output, which is not enough for evaluation.

---

# Step 5: Predictor Validation for Manufacturer Edits

## What We Did

We created:

```text
src/explainability/validation.py
scripts/phase9_step5_validate_manufacturer_edits.py
```

The validation module loads the predictor and computes:

```text
manufacturer prediction before and after
field strength prediction before and after
EchoTime prediction before and after
target probability gain
```

For manufacturer editing, the main metric was:

```text
target_manufacturer_prob_gain
```

## Result

The manufacturer validation showed:

```text
Number of runs = 8
Positive target manufacturer gain rate = 1.0
Manufacturer prediction hit target rate after = 0.0
Mean target manufacturer probability gain = 0.000186
Median target manufacturer probability gain = 0.00000696
Mean global SSIM = 0.7345
Mean absolute difference = 0.0311
```

## Interpretation

All outputs moved slightly toward the target manufacturer, but none of them fully switched the predictor to the target class.

This means:

```text
Manufacturer editing was visually changing the image,
but the predictor did not consider the change strong enough to classify it as target manufacturer.
```

So we did not stop here. We decided to run a stronger manufacturer tuning grid.

## Image to Add

Add this manually:

```text
reports/counterfactuals/phase9_step5_predictor_validation/target_manufacturer_probability_gain.png
```

---

# Step 6: Manufacturer Tuning Grid

## What We Did

We created:

```text
scripts/phase9_step6_manufacturer_tuning_grid.py
```

This tested a wider range of settings:

```text
edit_timestep = 100, 200, 300, 400
guidance_scale = 1.0, 3.0, 5.0
```

Total outputs:

```text
2 examples × 4 timesteps × 3 guidance scales = 24 outputs
```

## Result

The tuning grid showed:

```text
positive_gain_rate = 1.0
hit_target_rate_after = 0.25
mean_target_gain = 0.2496
median_target_gain = 0.0019
max_target_gain = 0.9999
mean_global_ssim = 0.5770
mean_abs_diff = 0.0496
```

## Best High-Gain Manufacturer Result

```text
Source → Target: GE MEDICAL SYSTEMS → SIEMENS
edit_timestep = 400
guidance_scale = 5.0
SSIM = 0.2503
MAD = 0.1139
Target probability gain = 0.999999
Hit target = True
```

## Best Anatomy-Preserving Manufacturer Result

```text
Source → Target: SIEMENS → GE MEDICAL SYSTEMS
edit_timestep = 100
guidance_scale = 5.0
SSIM = 0.7322
MAD = 0.0281
Target probability gain = 0.002553
Hit target = False
```

## Interpretation

Manufacturer editing showed a clear tradeoff:

```text
Low edit strength:
  Better anatomy preservation
  Weak manufacturer shift

High edit strength:
  Strong manufacturer target alignment
  Poor anatomy preservation
```

So manufacturer control is possible, but not yet clean enough.

We should present manufacturer as a **tradeoff result**, not as the strongest successful control.

## Images to Add

Add these manually:

```text
reports/counterfactuals/phase9_step6_manufacturer_tuning/manufacturer_tuning_tradeoff.png
reports/counterfactuals/phase9_step6_manufacturer_tuning/grids/example_00_manufacturer_tuning_grid.png
reports/counterfactuals/phase9_step6_manufacturer_tuning/grids/example_01_manufacturer_tuning_grid.png
```

---

# Step 7: Field Strength Counterfactual Edit

## What We Did

We created:

```text
scripts/phase9_step7_field_strength_edit.py
```

This tested field strength transfer:

```text
1.5T → 3.0T
3.0T → 1.5T
```

Only field strength was changed.

These were kept the same:

```text
manufacturer_label
echo_time_norm
```

Tested settings:

```text
edit_timestep = 100, 200, 300
guidance_scale = 1.0, 3.0, 5.0
```

Total outputs:

```text
18 outputs
```

## Result

The field strength result was very strong:

```text
positive_field_gain_rate = 1.0
field_hit_target_rate_after = 1.0
mean_target_field_gain = 0.9915
median_target_field_gain = 0.9933
max_target_field_gain = 0.999996
mean_global_ssim = 0.7465
mean_abs_diff = 0.0339
```

## Best Field Strength Result

```text
Source → Target: 1.5T → 3.0T
Manufacturer: GE MEDICAL SYSTEMS
edit_timestep = 100
guidance_scale = 3.0
SSIM = 0.8091
MAD = 0.0266
Target field probability gain = 0.999997
Before predictor = 1.5T
After predictor = 3.0T
Manufacturer stayed stable = GE → GE
```

## Interpretation

Field strength editing was the strongest and cleanest control in Phase 9.

It achieved:

```text
Strong target alignment
Good anatomy preservation
Stable manufacturer prediction
```

This is the best control to show in demo/evaluation.

## Recommended Demo Setting

```text
edit_timestep = 100
guidance_scale = 3.0
```

## Images to Add

Add these manually:

```text
reports/counterfactuals/phase9_step7_field_strength_edit/field_strength_tradeoff.png
reports/counterfactuals/phase9_step7_field_strength_edit/grids/example_00_field_strength_edit_grid.png
reports/counterfactuals/phase9_step7_field_strength_edit/grids/example_01_field_strength_edit_grid.png
```

---

# Step 8: EchoTime Counterfactual Edit

## What We Did

We created:

```text
scripts/phase9_step8_echo_time_edit.py
```

This tested continuous EchoTime editing.

Only EchoTime was changed.

These were kept the same:

```text
manufacturer_label
field_strength_label
```

We tested two directions:

```text
High EchoTime → Low EchoTime
Low EchoTime → High EchoTime
```

The target values were:

```text
Low EchoTime target = 0.15
High EchoTime target = 0.85
```

Tested settings:

```text
edit_timestep = 100, 200, 300
guidance_scale = 1.0, 3.0, 5.0
```

Total outputs:

```text
18 outputs
```

## Result

EchoTime editing showed:

```text
moved_closer_rate = 1.0
mean_echo_error_improvement = 0.1294
median_echo_error_improvement = 0.1031
max_echo_error_improvement = 0.4306
mean_global_ssim = 0.6968
mean_abs_diff = 0.0321
manufacturer_stability_rate = 1.0
field_strength_stability_rate = 1.0
```

## Best EchoTime Result

```text
Manufacturer = GE MEDICAL SYSTEMS
Field strength = 1.5T
Source EchoTime norm = 0.7058
Target EchoTime norm = 0.1500
edit_timestep = 100
guidance_scale = 1.0
SSIM = 0.8127
MAD = 0.0249
Predictor before = 0.8864
Predictor after = 0.4558
Error improvement = 0.4306
Manufacturer stable = True
Field strength stable = True
```

## Interpretation

EchoTime editing was successful as **directional continuous control**.

It means the model moved the predictor closer to the target EchoTime in all tested cases.

However, it did not always reach the exact target value.

The high-to-low direction worked better than the low-to-high direction.

So EchoTime control is successful, but not perfect.

## Recommended Demo Setting

```text
edit_timestep = 100
guidance_scale = 1.0
```

## Images to Add

Add these manually:

```text
reports/counterfactuals/phase9_step8_echo_time_edit/echo_time_tradeoff.png
reports/counterfactuals/phase9_step8_echo_time_edit/grids/example_00_echo_time_edit_grid.png
reports/counterfactuals/phase9_step8_echo_time_edit/grids/example_01_echo_time_edit_grid.png
```

---

# Step 9: Final Explainability Panels

## What We Did

We created:

```text
scripts/phase9_step9_final_explainability_panels.py
```

This script selected the best final cases and generated evaluation-ready explainability panels.

The final selected cases were:

```text
Manufacturer balanced
Manufacturer high-gain
Field strength best
EchoTime best
```

Each panel included:

```text
Original image
Counterfactual image
Absolute difference map
Local SSIM map
Histogram comparison
Predictor validation text
```

## Result

Final explainability summary:

| Case                   |   SSIM |    MAD | Local SSIM Mean | Histogram L1 | Interpretation                                                       |
| ---------------------- | -----: | -----: | --------------: | -----------: | -------------------------------------------------------------------- |
| Manufacturer balanced  | 0.7322 | 0.0281 |          0.7364 |       0.1658 | Best anatomy-preserving manufacturer candidate                       |
| Manufacturer high-gain | 0.2503 | 0.1139 |          0.2752 |       0.2046 | Shows target manufacturer can be achieved, but with anatomy tradeoff |
| Field strength best    | 0.8091 | 0.0266 |          0.8149 |       0.2367 | Best field-strength edit                                             |
| EchoTime best          | 0.8127 | 0.0249 |          0.8200 |       0.4836 | Best EchoTime directional edit                                       |

## Combined Final Explainability Image to Add

Add this manually:

```text
reports/counterfactuals/phase9_step9_final_explainability/combined_final_explainability_grid.png
```

## Individual Panels to Add If Needed

```text
reports/counterfactuals/phase9_step9_final_explainability/manufacturer_balanced/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/manufacturer_high_gain/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/field_strength_best/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/echotime_best/explainability_panel.png
```

---

# Step 10: Final Phase 9 Report Generation

## What We Did

We created:

```text
scripts/phase9_step10_generate_report.py
```

This script automatically generated:

```text
reports/counterfactuals/phase9_step10_final_report/phase9_completion_report.md
```

The report summarized:

```text
Phase 9 objective
Implementation files
Manufacturer editing results
Field strength editing results
EchoTime editing results
Final explainability panel summary
Control ranking
What we can claim
What we should not claim
Recommended next phase
```

This generated report is the base for the current detailed report. The final generated Phase 9 report correctly states the main findings and avoids overclaiming clinical realism. 

---

# 6. Overall Phase 9 Results

## Control Strength Ranking

Based on Phase 9:

```text
1. Field strength — strongest and cleanest control
2. EchoTime — successful directional continuous control
3. Manufacturer — controllable, but tradeoff-heavy
```

## Best Settings Found

### Field Strength

```text
edit_timestep = 100
guidance_scale = 3.0
```

Why:

```text
Strong predictor target alignment
Good SSIM
Manufacturer remained stable
```

### EchoTime

```text
edit_timestep = 100
guidance_scale = 1.0
```

Why:

```text
Best EchoTime error improvement
Good SSIM
Manufacturer and field strength remained stable
```

### Manufacturer

For anatomy-preserving output:

```text
edit_timestep = 100
guidance_scale = 5.0
```

For target-achieving output:

```text
edit_timestep = 400
guidance_scale = 5.0
```

But we should not use manufacturer as the strongest final demo, because strong target manufacturer control causes visible anatomy distortion.

---

# 7. Main Deliverables Created in Phase 9

## Code Deliverables

```text
src/inference/counterfactual_editor.py
src/explainability/validation.py

scripts/phase9_step1_check.py
scripts/phase9_step2_reconstruction_test.py
scripts/phase9_step3_manufacturer_edit.py
scripts/phase9_step4_predictor_check.py
scripts/phase9_step5_validate_manufacturer_edits.py
scripts/phase9_step6_manufacturer_tuning_grid.py
scripts/phase9_step7_field_strength_edit.py
scripts/phase9_step8_echo_time_edit.py
scripts/phase9_step9_final_explainability_panels.py
scripts/phase9_step10_generate_report.py
```

## Report and Output Deliverables

```text
reports/counterfactuals/phase9_step2_reconstruction/
reports/counterfactuals/phase9_step3_manufacturer_edit/
reports/counterfactuals/phase9_step5_predictor_validation/
reports/counterfactuals/phase9_step6_manufacturer_tuning/
reports/counterfactuals/phase9_step7_field_strength_edit/
reports/counterfactuals/phase9_step8_echo_time_edit/
reports/counterfactuals/phase9_step9_final_explainability/
reports/counterfactuals/phase9_step10_final_report/
```

## Important Summary Files

```text
reports/counterfactuals/phase9_step6_manufacturer_tuning/manufacturer_tuning_summary.json
reports/counterfactuals/phase9_step7_field_strength_edit/field_strength_edit_summary.json
reports/counterfactuals/phase9_step8_echo_time_edit/echo_time_edit_summary.json
reports/counterfactuals/phase9_step9_final_explainability/phase9_final_explainability_summary.json
reports/counterfactuals/phase9_step10_final_report/phase9_completion_report.md
```

---

# 8. Important Images to Include in Final Document

You can add these images manually.

## Reconstruction Sanity Test

```text
reports/counterfactuals/phase9_step2_reconstruction/comparison_original_reconstruction_diff.png
```

## Manufacturer Initial Edit

```text
reports/counterfactuals/phase9_step3_manufacturer_edit/grids/example_00_manufacturer_edit_grid.png
reports/counterfactuals/phase9_step3_manufacturer_edit/grids/example_01_manufacturer_edit_grid.png
```

## Manufacturer Predictor Validation

```text
reports/counterfactuals/phase9_step5_predictor_validation/target_manufacturer_probability_gain.png
```

## Manufacturer Tuning

```text
reports/counterfactuals/phase9_step6_manufacturer_tuning/manufacturer_tuning_tradeoff.png
reports/counterfactuals/phase9_step6_manufacturer_tuning/grids/example_00_manufacturer_tuning_grid.png
reports/counterfactuals/phase9_step6_manufacturer_tuning/grids/example_01_manufacturer_tuning_grid.png
```

## Field Strength Editing

```text
reports/counterfactuals/phase9_step7_field_strength_edit/field_strength_tradeoff.png
reports/counterfactuals/phase9_step7_field_strength_edit/grids/example_00_field_strength_edit_grid.png
reports/counterfactuals/phase9_step7_field_strength_edit/grids/example_01_field_strength_edit_grid.png
```

## EchoTime Editing

```text
reports/counterfactuals/phase9_step8_echo_time_edit/echo_time_tradeoff.png
reports/counterfactuals/phase9_step8_echo_time_edit/grids/example_00_echo_time_edit_grid.png
reports/counterfactuals/phase9_step8_echo_time_edit/grids/example_01_echo_time_edit_grid.png
```

## Final Explainability Panels

```text
reports/counterfactuals/phase9_step9_final_explainability/combined_final_explainability_grid.png
```

Optional individual panels:

```text
reports/counterfactuals/phase9_step9_final_explainability/manufacturer_balanced/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/manufacturer_high_gain/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/field_strength_best/explainability_panel.png
reports/counterfactuals/phase9_step9_final_explainability/echotime_best/explainability_panel.png
```

---

# 9. What We Can Claim After Phase 9

We can confidently claim:

1. A working counterfactual editing prototype has been implemented.
2. The model can take a real MRI slice and generate target-conditioned edited versions.
3. The system supports manufacturer, field strength, and EchoTime target conditions.
4. Field strength editing is strongly validated.
5. EchoTime editing works as directional continuous control.
6. Manufacturer editing shows measurable control but has a strong anatomy-preservation tradeoff.
7. The system produces explainability outputs:

   * Difference maps
   * Local SSIM maps
   * Histogram comparison
   * Predictor before/after validation
8. The project now has a complete evaluation-ready Phase 9 report and output folder.

---

# 10. What We Should Not Claim Yet

We should not claim:

1. The model generates clinically realistic MRI images.
2. The model is ready for medical use.
3. The model perfectly preserves anatomy in all cases.
4. Manufacturer transfer is fully solved.
5. EchoTime control reaches the exact target value every time.
6. The model is final or deployment-ready.
7. The generated counterfactuals are medically validated.

The correct claim is:

```text
This is a working research prototype for explainable acquisition-parameter counterfactual MRI editing.
```

---

# 11. Justification Questions and Answers

## Q1. Why did we implement counterfactual editing after diffusion training?

Because diffusion training only proves that the model can generate MRI-like images from random noise. But the actual project goal is counterfactual generation, where a real MRI slice is edited under a new condition. Phase 9 was needed to bridge that gap.

---

## Q2. Why did we use partial noising instead of generating from pure noise?

Because pure noise generation creates a new synthetic image and does not preserve the original anatomy. Partial noising starts from the real image, so the output is more likely to retain the same patient anatomy while changing acquisition appearance.

---

## Q3. Why did we first test same-condition reconstruction?

Because before testing real edits, we needed to confirm that the pipeline can reconstruct an image without changing the target condition. If same-condition reconstruction failed, then counterfactual editing would not be trustworthy.

---

## Q4. Why was SSIM used?

SSIM measures structural similarity between the original and generated image. Since the goal is to preserve anatomy, SSIM is an important metric. Higher SSIM means the generated image is structurally closer to the original.

---

## Q5. Why did we use mean absolute difference?

Mean absolute difference measures average pixel-level intensity change. It helps quantify how much the generated image changed from the original.

---

## Q6. Why did we use local SSIM maps?

Global SSIM gives one number, but local SSIM maps show where the structure changed. This is useful because in medical imaging, a small local distortion may be important even if the global score looks acceptable.

---

## Q7. Why did we use the predictor for validation?

The predictor checks whether the generated counterfactual moved toward the requested target acquisition condition. Without the predictor, we would only have visual results and no automatic validation.

---

## Q8. Why did manufacturer editing need tuning?

The first manufacturer edits preserved anatomy but did not strongly change predictor classification. Therefore, we increased edit timestep and guidance scale to test whether stronger edits could produce stronger target alignment.

---

## Q9. What did manufacturer tuning show?

It showed a tradeoff. Strong edits can flip the predictor to the target manufacturer, but they also damage anatomy. Lighter edits preserve anatomy but do not fully change manufacturer prediction.

---

## Q10. Why is field strength the strongest result?

Field strength editing achieved target predictor alignment in all tested cases while keeping SSIM reasonably high. It also kept manufacturer stable. This makes it the cleanest successful control.

---

## Q11. Why is EchoTime considered successful even though it does not exactly reach the target?

EchoTime is a continuous variable, so exact target matching is harder than binary classification. The important first result is that the predictor moved closer to the target in all tested cases. That means the model learned directional EchoTime control.

---

## Q12. Why did EchoTime high-to-low work better than low-to-high?

The model and dataset may represent some EchoTime ranges more strongly than others. Also, continuous regression is harder than binary classification. More data, stronger training, or better conditioning may improve low-to-high control.

---

## Q13. Why did we keep other conditions fixed during each edit?

To isolate the effect of one control at a time. If we changed manufacturer, field strength, and EchoTime together, we would not know which condition caused the observed change.

---

## Q14. Why did we create final explainability panels?

Because the project is not just image generation. It is explainable counterfactual MRI generation. The final panels show the original, counterfactual, difference map, local SSIM map, histogram shift, and predictor validation in one place.

---

## Q15. Which result should be shown as the main demo?

The best main demo should be field strength editing:

```text
1.5T → 3.0T
edit_timestep = 100
guidance_scale = 3.0
SSIM = 0.8091
target field probability gain = 0.999997
```

This is the cleanest and strongest result.

---

## Q16. Which result should be shown as a secondary demo?

EchoTime editing should be shown as the secondary demo:

```text
EchoTime norm 0.7058 → 0.1500
edit_timestep = 100
guidance_scale = 1.0
SSIM = 0.8127
EchoTime error improvement = 0.4306
```

---

## Q17. How should manufacturer editing be presented?

Manufacturer editing should be presented as a tradeoff experiment. We should show both:

```text
Manufacturer balanced:
  Better anatomy preservation but weak target shift

Manufacturer high-gain:
  Strong target shift but poor anatomy preservation
```

This is honest and scientifically defensible.

---

## Q18. Is Phase 9 complete?

Yes. Phase 9 is complete because the system now performs real image counterfactual editing, validates outputs with the predictor, and generates explainability panels.

---

# 12. Current Project Status After Phase 9

Completed till now:

```text
Step 0: MVP scope
Step 1: Repository setup
Step 2: Environment setup
Step 3: Metadata extraction
Step 4: Metadata exploration and subset selection
Step 5: Preprocessing and patient-level splits
Step 6: Acquisition predictor
Step 7: Baselines and explainability functions
Step 8: Conditional diffusion training
Step 9: Counterfactual editing prototype
```

Current strongest achievement:

```text
A working explainable counterfactual MRI editing prototype has been implemented.
```

---

# 13. What To Do Next

The recommended next phase is:

```text
Phase 10A: Expanded Evaluation
Phase 10B: Streamlit Dashboard
```

## Phase 10A: Expanded Evaluation

Right now, Phase 9 was tested on selected examples. For stronger evaluation, we should run the best settings on more test images.

Recommended expanded evaluation:

```text
Field strength:
  50–100 test examples
  edit_timestep = 100
  guidance_scale = 3.0

EchoTime:
  50–100 test examples
  edit_timestep = 100
  guidance_scale = 1.0

Manufacturer:
  smaller tradeoff evaluation
  balanced setting and high-gain setting
```

Metrics to report:

```text
Mean SSIM
Mean absolute difference
Target predictor gain
Target hit rate
Control stability
Histogram shift
Failure examples
Best/worst examples
```

## Phase 10B: Dashboard Integration

After expanded evaluation, we should build a Streamlit dashboard.

The dashboard should allow:

```text
Select test image
View original metadata
Select target manufacturer
Select target field strength
Select target EchoTime
Choose edit timestep
Choose guidance scale
Generate counterfactual
Show output
Show difference map
Show local SSIM map
Show histogram comparison
Show predictor before/after validation
```

Recommended dashboard defaults:

```text
Field strength demo:
  edit_timestep = 100
  guidance_scale = 3.0

EchoTime demo:
  edit_timestep = 100
  guidance_scale = 1.0

Manufacturer demo:
  show as tradeoff mode
```

---

# 14. Final Phase 9 Conclusion

Phase 9 was a major milestone.

Until Phase 8, the model could generate MRI-like samples from random noise. In Phase 9, the system became a real counterfactual editing prototype.

The final result is:

```text
Real MRI slice
→ target acquisition condition
→ generated counterfactual MRI
→ explainability output
→ predictor validation
```

The strongest result is field strength editing. EchoTime editing also works directionally. Manufacturer editing is possible but currently has a strong tradeoff between target alignment and anatomy preservation.

So the correct final conclusion is:

```text
Phase 9 successfully implemented the first explainable counterfactual MRI editing prototype. The model can edit acquisition-related attributes, especially field strength and EchoTime, while generating validation-ready outputs through difference maps, local SSIM maps, histograms, and predictor-based target alignment.
```
