# Progress Report: Step 5, Step 6, and Step 7 Completion

**Project:** Explainable Counterfactual MRI Generation and Acquisition Parameter Control for Robust Medical Imaging AI
**Current Stage:** Non-diffusion prototype foundation completed
**Main Work Completed Today:** Preprocessing, patient-level splits, acquisition-parameter predictor, baseline transformations, and explainability metrics.

---

## 1. Project Context

The project goal is to build an explainable and controllable MRI transformation system where a user can select an MRI slice and modify scanner/acquisition-related parameters such as:

```text
Scanner manufacturer
Magnetic field strength
One continuous timing parameter: EchoTime
```

The proposal states that the final system should generate counterfactual MRI images while preserving anatomy and should display difference maps, local SSIM maps, histogram shifts, and predictor-based validation so that the generated changes are interpretable rather than black-box outputs. 

Before today, Steps 0–4 had already been completed. The earlier report established that the project selected:

```text
First subset: ax t1
Discrete control 1: Manufacturer
Discrete control 2: MagneticFieldStrength
Continuous control: EchoTime
```

The `ax t1` subset was selected because it contains 6355 slices, 103 patients, both GE and Siemens scanners, both 1.5T and 3.0T field strengths, and usable EchoTime variation. 

Today’s work started from that point and completed the next three implementation stages:

```text
Step 5: Preprocessing and slice curation
Step 6: Acquisition-parameter predictor
Step 7: Baseline transformations and explainability functions
```

These steps follow the implementation plan, which explicitly recommends preprocessing, patient-level splits, acquisition-parameter predictor training, baseline transformations, and explainability modules before moving to conditional diffusion. 

---

# 2. Step 5: Preprocessing and Slice Curation

## 2.1 What We Did

We created and ran a preprocessing script:

```text
data/preprocess.py
```

The command used was:

```bash
python data/preprocess.py \
  --metadata_csv data/metadata/ax_t1_metadata.csv \
  --output_dir data/processed/ax_t1_128 \
  --image_size 128 \
  --save_format npy \
  --min_foreground 0.01 \
  --seed 42
```

The script performed the following operations:

```text
1. Read ax_t1_metadata.csv
2. Loaded DICOM pixel arrays from slice_path
3. Converted images to float32
4. Applied DICOM rescale slope/intercept where available
5. Clipped intensities using 1st and 99th percentiles
6. Normalized each slice to [0, 1]
7. Resized each slice to 128×128
8. Checked foreground content
9. Saved processed images as .npy files
10. Created processed_metadata.csv
11. Created patient-level train/val/test splits
12. Saved label maps and preprocessing summaries
```

This directly matches the implementation plan’s Step 5 requirements: load DICOM pixel arrays, convert to float32, clip using 1st and 99th percentiles, normalize to `[0, 1]`, resize to 128 or 256 for early experiments, remove low-content slices, save processed metadata, and create patient-level splits. 

---

## 2.2 Preprocessing Results

The preprocessing output was:

```json
{
  "input_rows_after_filtering": 6355,
  "processed_slices": 6355,
  "rejected_slices": 0,
  "num_patients": 103,
  "manufacturer_counts": {
    "GE MEDICAL SYSTEMS": 4837,
    "SIEMENS": 1518
  },
  "field_strength_counts": {
    "1.5T": 3399,
    "3.0T": 2956
  },
  "echo_time_min": 7.88,
  "echo_time_max": 12.272,
  "echo_time_mean": 9.918987254130606,
  "splits": {
    "total_patients": 103,
    "train_patients": 72,
    "val_patients": 15,
    "test_patients": 16,
    "train_images": 4434,
    "val_images": 987,
    "test_images": 934
  }
}
```

## 2.3 Interpretation

These results are good.

| Check                   |      Result | Interpretation                                    |
| ----------------------- | ----------: | ------------------------------------------------- |
| Input rows              |        6355 | Matches selected `ax t1` subset                   |
| Processed slices        |        6355 | All selected slices were successfully processed   |
| Rejected slices         |           0 | No read failure or low-content rejection occurred |
| Patients                |         103 | Enough patients for patient-level splitting       |
| GE slices               |        4837 | Majority manufacturer class                       |
| Siemens slices          |        1518 | Minority but still sufficient                     |
| 1.5T slices             |        3399 | Good count                                        |
| 3.0T slices             |        2956 | Good count                                        |
| EchoTime range          | 7.88–12.272 | Clean continuous range for first baseline         |
| Train/val/test patients |    72/15/16 | Correct patient-level split behavior              |

The fact that `rejected_slices = 0` is acceptable here because `ax t1` was already a clean curated subset. It means the DICOM paths were valid, the images were readable, and the foreground threshold did not eliminate any selected slices.

---

## 2.4 Why We Used `ax_t1_metadata.csv` Instead of Full Metadata

We used:

```text
data/metadata/ax_t1_metadata.csv
```

instead of the full metadata CSV because previous analysis showed that mixing all series would introduce confounding. MRI series differ by sequence type, contrast phase, orientation, protocol, and scanner settings. If all series were mixed at once, the model might learn sequence differences instead of acquisition-parameter differences.

The earlier report already justified `ax t1` as the first preprocessing subset because it supports all three MVP controls: manufacturer, field strength, and EchoTime. It also noted that `ax dyn pre`, while closer to the final dynamic/pre-contrast goal, currently contains only Siemens in the downloaded subset and therefore cannot support GE ↔ Siemens manufacturer transfer. 

So this was a staged implementation decision, not a change in the project goal.

---

## 2.5 Why We Used 128×128 Images

We used:

```text
image_size = 128
```

instead of 256 because this is the first experimental baseline. The implementation plan allows 128×128 or 256×256 for early experiments. 

The reason for 128×128 is practical:

```text
Faster preprocessing
Faster predictor training
Lower GPU memory usage
Faster debugging
Suitable for first diffusion experiments
```

Later, after the pipeline is stable, we can move to 256×256 for better visual quality.

---

## 2.6 Why We Saved Images as `.npy`

We saved processed images as:

```text
.npy
```

instead of PNG because `.npy` preserves floating-point normalized intensity values. This is important for medical imaging because intensity changes are part of the acquisition effect we want to model. PNG is useful for visualization, but it can quantize intensities to 8-bit values.

---

## 2.7 Why Patient-Level Splitting Was Required

The implementation plan explicitly says to use patient-level train/validation/test splits instead of random slice-level splits because random slice-level splitting causes data leakage. 

This is important because MRI slices from the same patient are highly similar. If slices from the same patient appeared in both training and validation/test sets, the model could appear to perform well simply because it had already seen almost identical anatomy.

Our split result was:

```text
Train: 72 patients, 4434 images
Val:   15 patients, 987 images
Test:  16 patients, 934 images
```

This is a correct and defensible split.

---

# 3. Step 6: Acquisition-Parameter Predictor

## 3.1 What We Did

We created the acquisition-parameter predictor pipeline with these files:

```text
src/data/dataset.py
src/models/predictor.py
src/training/train_predictor.py
scripts/evaluate_predictor.py
configs/predictor.yaml
```

The model was trained using:

```bash
python src/training/train_predictor.py --config configs/predictor.yaml
```

The predictor was designed as a multi-task model:

```text
Input:
128×128 normalized MRI slice

Shared backbone:
Small CNN

Output heads:
1. Manufacturer classification
2. Field strength classification
3. EchoTime regression
```

The implementation plan states that the predictor should take an MRI slice as input, use a ResNet18 or small CNN backbone, and predict manufacturer, field strength, and TE/TR through classification and regression heads. 

---

## 3.2 Why We Built the Predictor Before Diffusion

The acquisition-parameter predictor is not just an extra model. It becomes the validation module for the final counterfactual system.

The implementation plan says this model should be built before diffusion because later, when a diffusion model generates a counterfactual image, the predictor checks whether the generated image moved toward the requested target acquisition condition. 

Example:

```text
Original image:
SIEMENS, 1.5T, TE = x

Target condition:
GE, 3.0T, TE = y

Generated image:
Predictor should move toward GE, 3.0T, TE = y
```

This directly supports the proposal’s validation goal, where predictor-based target alignment is displayed along with difference maps, SSIM maps, and histograms. 

---

## 3.3 Why We Used a Small CNN First

We used a small CNN instead of starting with ResNet18 because the goal at this stage was to validate the pipeline, not to maximize model complexity.

A small CNN is better for the first baseline because:

```text
It trains quickly.
It is easier to debug.
It uses less GPU memory.
It proves whether acquisition labels are learnable from images.
It is enough for a first verification model.
```

Later, if needed, this can be upgraded to ResNet18.

---

## 3.4 Training Setup

The training used GPU successfully:

```text
Device: cuda
```

The model had:

```text
455,397 trainable parameters
```

Class weighting was used because manufacturer distribution was imbalanced:

```text
Manufacturer class weights: [0.6261, 2.4826]
Field-strength class weights: [0.9056, 1.1163]
```

This was important because the processed subset had more GE slices than Siemens slices:

```text
GE: 4837
Siemens: 1518
```

Without class weighting, the model could become biased toward GE.

---

## 3.5 Training Results

The final training epoch produced:

```text
train_loss = 0.0058
val_loss = 0.0823
val_mfg_acc = 1.0000
val_field_acc = 0.9757
val_echo_norm_mae = 0.1591
```

The best checkpoint was saved at:

```text
checkpoints/predictor/ax_t1_128/best.pt
```

## 3.6 Training Interpretation

The training result was good because:

```text
Manufacturer validation accuracy reached 100%.
Field-strength validation accuracy reached 97.57%.
Validation loss reached the best value at the final epoch.
The model successfully saved best.pt and latest.pt checkpoints.
```

The validation loss curve was somewhat spiky, but this was not a blocker because the best checkpoint had strong validation performance. Spikes can happen because:

```text
The validation set is patient-level and smaller.
BatchNorm can fluctuate with small dataset distributions.
Field strength prediction may vary across patient groups.
Learning rate may be slightly high.
```

The correct decision was to judge the saved best checkpoint using the held-out test set.

---

## 3.7 Test Evaluation Results

We evaluated the predictor using:

```bash
python scripts/evaluate_predictor.py \
  --checkpoint checkpoints/predictor/ax_t1_128/best.pt \
  --test_csv data/processed/ax_t1_128/splits/test.csv \
  --output_dir reports/predictor/ax_t1_128/test_eval
```

The result was:

```json
{
  "num_samples": 934,
  "manufacturer_accuracy": 1.0,
  "manufacturer_balanced_accuracy": 1.0,
  "field_strength_accuracy": 0.9957173447537473,
  "field_strength_balanced_accuracy": 0.9958610188261351,
  "echo_time_norm_mae": 0.05458231262336196,
  "echo_time_norm_rmse": 0.08683645031037222,
  "echo_time_norm_mean_baseline_mae": 0.1253872545679325,
  "echo_beats_mean_baseline": true
}
```

## 3.8 Test Result Interpretation

| Metric                           | Result | Interpretation                             |
| -------------------------------- | -----: | ------------------------------------------ |
| Test samples                     |    934 | Good held-out test size                    |
| Manufacturer accuracy            | 1.0000 | Perfect manufacturer prediction            |
| Manufacturer balanced accuracy   | 1.0000 | Not biased toward one class                |
| Field strength accuracy          | 0.9957 | Almost perfect field strength prediction   |
| Field strength balanced accuracy | 0.9959 | Strong across both field classes           |
| EchoTime normalized MAE          | 0.0546 | Good regression error                      |
| EchoTime mean-baseline MAE       | 0.1254 | Model is better than naive mean prediction |
| Beats baseline                   |   true | EchoTime head is learning useful signal    |

The field-strength confusion matrix showed:

```text
True class 0: 429 correct, 1 wrong
True class 1: 501 correct, 3 wrong
```

This means the model is not simply guessing the majority class. It is learning both field-strength classes.

For EchoTime, the normalized MAE was:

```text
0.05458
```

The EchoTime range was:

```text
12.272 - 7.88 = 4.392 ms
```

So the approximate real EchoTime error is:

```text
0.05458 × 4.392 ≈ 0.24 ms
```

That is a strong first baseline.

---

## 3.9 Step 6 Conclusion

Step 6 is complete.

We now have:

```text
A trained acquisition-parameter predictor
A saved best checkpoint
Manufacturer accuracy report
Field-strength confusion matrix
EchoTime regression plot
Predictions CSV
Metrics JSON
```

This satisfies the implementation plan’s predictor milestone: checkpoint, accuracy/MAE report, confusion matrix, and validation loss. 

---

# 4. Step 7: Baseline Transformations and Explainability

## 4.1 What We Did

We implemented simple baseline transformations and explainability functions using these files:

```text
src/inference/baselines.py
src/explainability/difference_map.py
src/explainability/ssim_map.py
src/explainability/histogram.py
scripts/run_baseline_explainability_demo.py
```

We first tested the pipeline on 5 examples, then reran it on 50 examples for a more reliable report.

The final command was:

```bash
python scripts/run_baseline_explainability_demo.py \
  --metadata_csv data/processed/ax_t1_128/processed_metadata.csv \
  --output_dir reports/baselines/ax_t1_128_full \
  --num_examples 50 \
  --seed 42
```

The implementation plan says that before diffusion, simple baselines should be created, including identity, random intensity augmentation, histogram matching, Gaussian noise, and contrast shift. It also says these baselines should be compared using SSIM, histogram distance, predictor target confidence, and visual quality. 

---

## 4.2 Baselines Implemented

We implemented:

```text
1. Identity baseline
2. Contrast shift baseline
3. Gamma correction baseline
4. Gaussian noise baseline
5. Histogram matching baseline
```

### Identity Baseline

This returns the original image unchanged.

Purpose:

```text
Serves as the no-change reference.
SSIM should be 1.0.
Difference should be 0.
Histogram distance should be 0.
```

### Contrast Shift Baseline

This applies a simple linear intensity change:

```text
output = alpha × image + beta
```

Purpose:

```text
Tests whether simple scanner-style intensity changes can produce measurable differences while preserving anatomy.
```

### Gamma Baseline

This applies a nonlinear intensity transformation.

Purpose:

```text
Simulates moderate brightness/contrast style changes.
Useful as a classical non-generative style baseline.
```

### Gaussian Noise Baseline

This adds random noise.

Purpose:

```text
Tests whether random perturbation damages structure.
A good generative model should not behave like random noise.
```

### Histogram Matching Baseline

This matches the source image histogram to a reference image from another manufacturer.

Purpose:

```text
Provides a strong classical scanner-style baseline.
Useful comparison for later diffusion model.
```

---

## 4.3 Explainability Functions Implemented

We implemented:

```text
compute_absolute_difference(original, generated)
compute_local_ssim_map(original, generated)
compute_histogram_comparison(original, generated)
```

These match the proposal and implementation plan requirements. The proposal states that explainability outputs should include difference maps, local structural similarity maps, histogram shifts, and predictor-confidence changes. 

The implementation plan similarly states that the explainability module should compute absolute difference, local SSIM maps, histogram comparison, and predictor shift, and that these outputs turn the project from simple image generation into an explainable counterfactual MRI platform. 

---

## 4.4 50-Example Baseline Results

The final 50-example baseline summary was:

```json
{
  "num_examples": 50,
  "num_baseline_records": 250,
  "baselines": [
    "contrast_shift",
    "gamma",
    "gaussian_noise",
    "histogram_matching",
    "identity"
  ],
  "mean_metrics_by_baseline": {
    "mean_absolute_difference": {
      "contrast_shift": 0.013540799133479595,
      "gamma": 0.015570441782474518,
      "gaussian_noise": 0.020025226548314096,
      "histogram_matching": 0.06567134084179997,
      "identity": 0.0
    },
    "global_ssim": {
      "contrast_shift": 0.9928092945580343,
      "gamma": 0.959317494045019,
      "gaussian_noise": 0.7395710025026943,
      "histogram_matching": 0.7435789160801685,
      "identity": 1.0
    },
    "hist_l1_distance": {
      "contrast_shift": 0.14395751953125,
      "gamma": 0.39521484375,
      "gaussian_noise": 0.361845703125,
      "histogram_matching": 0.5821728515625,
      "identity": 0.0
    },
    "hist_intersection": {
      "contrast_shift": 0.928021240234375,
      "gamma": 0.802392578125,
      "gaussian_noise": 0.8190771484375,
      "histogram_matching": 0.70891357421875,
      "identity": 1.0
    }
  }
}
```

---

## 4.5 Baseline Result Interpretation

| Baseline           | Mean Difference | Global SSIM | Histogram L1 | Histogram Intersection | Interpretation                                           |
| ------------------ | --------------: | ----------: | -----------: | ---------------------: | -------------------------------------------------------- |
| identity           |          0.0000 |      1.0000 |       0.0000 |                 1.0000 | Perfect no-change reference                              |
| contrast_shift     |          0.0135 |      0.9928 |       0.1440 |                 0.9280 | Very small change, structure preserved                   |
| gamma              |          0.0156 |      0.9593 |       0.3952 |                 0.8024 | Moderate style shift, structure still preserved          |
| gaussian_noise     |          0.0200 |      0.7396 |       0.3618 |                 0.8191 | Noise damages structure                                  |
| histogram_matching |          0.0657 |      0.7436 |       0.5822 |                 0.7089 | Strongest appearance shift, lower structure preservation |

These results are correct and expected.

The pattern is meaningful:

```text
Identity gives no change.
Contrast shift gives a tiny change with very high SSIM.
Gamma gives moderate intensity/style change with good SSIM.
Gaussian noise reduces structural similarity.
Histogram matching creates the strongest histogram/style shift but sacrifices structure.
```

This gives a clear comparison framework for diffusion.

---

## 4.6 Why Baselines Are Necessary Before Diffusion

Baselines are necessary because diffusion is expensive and complex. Without baselines, we would not know whether diffusion is actually useful or merely producing changes that simple transformations can already produce.

The implementation plan explicitly says that if the diffusion model performs worse than histogram matching or simple contrast changes, then something is wrong. 

So these baselines will later answer:

```text
Does diffusion preserve anatomy better than histogram matching?
Does diffusion create more meaningful acquisition-style change than contrast shift?
Does diffusion avoid noise-like structural damage?
Does diffusion move predictor outputs toward target conditions?
```

---

# 5. What We Have Completed Today

Today’s completed work can be summarized as:

```text
Step 5 completed:
- Preprocessed 6355 ax_t1 DICOM slices
- Saved normalized 128×128 .npy files
- Created processed_metadata.csv
- Created patient-level train/val/test splits
- Confirmed 103 patients and usable label distributions

Step 6 completed:
- Built PyTorch dataset loader
- Built small CNN acquisition predictor
- Trained predictor using GPU
- Saved best checkpoint
- Evaluated on held-out test set
- Achieved strong manufacturer, field strength, and EchoTime results

Step 7 completed:
- Implemented identity, contrast, gamma, noise, and histogram matching baselines
- Implemented difference maps
- Implemented local SSIM maps
- Implemented histogram comparison
- Evaluated baselines on 50 examples
- Produced baseline metrics and visualization outputs
```

---

# 6. Deliverables Created Today

## Data and Preprocessing Deliverables

```text
data/processed/ax_t1_128/images/
data/processed/ax_t1_128/processed_metadata.csv
data/processed/ax_t1_128/rejected_slices.csv
data/processed/ax_t1_128/preprocessing_summary.json
data/processed/ax_t1_128/splits/train.csv
data/processed/ax_t1_128/splits/val.csv
data/processed/ax_t1_128/splits/test.csv
data/processed/ax_t1_128/label_maps/manufacturer_to_id.json
data/processed/ax_t1_128/label_maps/field_strength_to_id.json
data/processed/ax_t1_128/label_maps/echo_time_stats.json
```

## Predictor Deliverables

```text
src/data/dataset.py
src/models/predictor.py
src/training/train_predictor.py
scripts/evaluate_predictor.py
configs/predictor.yaml

checkpoints/predictor/ax_t1_128/best.pt
checkpoints/predictor/ax_t1_128/latest.pt
reports/predictor/ax_t1_128/history.csv
reports/predictor/ax_t1_128/loss_curve.png
reports/predictor/ax_t1_128/training_summary.json
reports/predictor/ax_t1_128/test_eval/metrics.json
reports/predictor/ax_t1_128/test_eval/predictions.csv
reports/predictor/ax_t1_128/test_eval/field_strength_confusion_matrix.png
reports/predictor/ax_t1_128/test_eval/manufacturer_confusion_matrix.png
reports/predictor/ax_t1_128/test_eval/echo_time_norm_predictions_vs_true.png
```

## Baseline and Explainability Deliverables

```text
src/inference/baselines.py
src/explainability/difference_map.py
src/explainability/ssim_map.py
src/explainability/histogram.py
scripts/run_baseline_explainability_demo.py

reports/baselines/ax_t1_128_full/baseline_metrics.csv
reports/baselines/ax_t1_128_full/baseline_summary.json
reports/baselines/ax_t1_128_full/figures/
```

---

# 7. Evaluation and Justification Questions

## Q1. Why did we preprocess before training any model?

Because models need consistent, clean, normalized image tensors. Raw DICOM images can have different dimensions, intensity scales, metadata formats, and pixel representations. Preprocessing ensures every slice has the same size, intensity range, and valid label connection.

Also, the implementation plan puts preprocessing before predictor training and diffusion training because data quality directly affects model validity. 

---

## Q2. Why did we use only `ax t1` for this stage?

Because `ax t1` is the strongest first subset for the baseline. It has:

```text
6355 slices
103 patients
GE and Siemens
1.5T and 3.0T
multiple EchoTime values
```

This supports all three MVP controls. The earlier report states that `ax dyn pre` is closer to the final dynamic MRI goal but currently contains only Siemens, so it cannot support manufacturer transfer in the first baseline. 

---

## Q3. Is using `ax t1` a change from the proposal?

No. It is a staged implementation decision. The final project still targets explainable acquisition-parameter counterfactual generation. `ax t1` was chosen first because it is cleaner and better balanced for proving the preprocessing and predictor pipeline.

Later, dynamic/pre-contrast series can still be curated separately for diffusion experiments.

---

## Q4. Why did we resize to 128×128?

Because 128×128 is faster and safer for early experiments. It reduces compute cost and GPU memory usage while preserving enough structure for a first model. The implementation plan explicitly allows 128×128 for early experiments. 

---

## Q5. Why normalize images to `[0, 1]`?

Neural networks train more stably when inputs are in a consistent numerical range. MRI intensities are not standardized across scanners, so percentile clipping and normalization reduce extreme outliers while keeping relative intensity structure.

---

## Q6. Why not use z-score normalization?

Because scanner/acquisition effects may appear as intensity and contrast differences. Strong per-slice z-score normalization could remove some of the style differences we want the model to learn. `[0, 1]` percentile normalization is safer for the first acquisition-control baseline.

---

## Q7. Why save as `.npy` instead of `.png`?

`.npy` preserves floating-point normalized pixel values. PNG is useful for visualization, but it compresses/quantizes intensities into image format. For training, `.npy` is more faithful.

---

## Q8. Why use patient-level splits?

Because random slice-level splitting can cause leakage. Slices from the same patient are highly similar. If one patient appears in both train and test, the model may memorize patient anatomy instead of learning generalizable acquisition features. The implementation plan specifically requires patient-level splits. 

---

## Q9. Why did we build the predictor before diffusion?

Because the predictor becomes the automatic validator for generated counterfactuals. The final system must show whether a generated image moved toward the target manufacturer, field strength, and EchoTime. The implementation plan explicitly states that the predictor should be built first for this reason. 

---

## Q10. Why use three predictor heads?

Because the MVP condition vector has three components:

```text
manufacturer_label
field_strength_label
echo_time_norm
```

Manufacturer and field strength are classification tasks. EchoTime is a continuous regression task. A multi-task model allows one shared image encoder to learn common acquisition-related features and separate heads to predict each control.

---

## Q11. Why did we use class weights?

Because the manufacturer distribution is imbalanced:

```text
GE: 4837
Siemens: 1518
```

Without class weights, the model could achieve misleading performance by favoring GE. Class weights penalize mistakes on the minority class more strongly and improve balanced learning.

---

## Q12. Are the predictor results good?

Yes. The held-out test result is very strong:

```text
Manufacturer accuracy: 100%
Field strength accuracy: 99.57%
EchoTime normalized MAE: 0.0546
EchoTime beats mean baseline: true
```

This means the image contains enough visual signal to predict acquisition parameters, which supports the project hypothesis.

---

## Q13. Why is EchoTime prediction important?

EchoTime is the continuous control variable. If the predictor can estimate EchoTime from the image, then later we can use the same predictor to check whether generated counterfactuals move toward a target EchoTime.

---

## Q14. What does EchoTime MAE mean in real units?

The normalized MAE was:

```text
0.05458
```

The EchoTime range was:

```text
12.272 - 7.88 = 4.392 ms
```

So approximate real MAE is:

```text
0.05458 × 4.392 ≈ 0.24 ms
```

This is strong for a first baseline.

---

## Q15. Why did we build simple baselines before diffusion?

Because baselines are necessary comparison points. Diffusion is complex and expensive. If diffusion cannot outperform or meaningfully improve over simple contrast shift, gamma correction, noise, or histogram matching, then the diffusion setup needs revision.

The implementation plan explicitly requires simple baselines before diffusion. 

---

## Q16. Why include identity baseline?

Identity gives the no-change reference:

```text
Difference = 0
SSIM = 1
Histogram distance = 0
```

Every generated or transformed image can be compared against this baseline.

---

## Q17. Why include contrast shift and gamma?

Because scanner/protocol differences often appear as intensity and contrast changes. These are simple non-generative transformations that test whether basic intensity manipulation already explains some acquisition differences.

---

## Q18. Why include Gaussian noise?

Because it shows what a bad or meaningless perturbation looks like. Noise changes the image but damages structure. A good counterfactual model should not behave like Gaussian noise.

---

## Q19. Why include histogram matching?

Histogram matching is a strong classical style-transfer baseline. It can create large intensity distribution changes, but it may reduce local structural similarity. This makes it useful for comparing whether diffusion can achieve style change while preserving anatomy better.

---

## Q20. Are the baseline results good?

Yes. The results follow the expected pattern:

```text
Identity: perfect no-change baseline.
Contrast shift: very small change, high SSIM.
Gamma: moderate intensity shift, still high SSIM.
Gaussian noise: lower SSIM due to structural disturbance.
Histogram matching: strongest histogram shift, lower SSIM.
```

This confirms that the baseline and explainability code is working.

---

## Q21. Why is SSIM important?

SSIM measures structural similarity. In this project, anatomy preservation is critical. A generated image may look different, but it should not change breast structure. SSIM and local SSIM maps help detect whether structure is preserved.

The proposal’s evaluation strategy includes anatomy preservation through global and local structural similarity. 

---

## Q22. Why is histogram distance important?

Histogram distance measures intensity distribution shift. Since scanner/acquisition differences often affect image intensity and contrast, histogram comparison helps show whether the output changed in a meaningful acquisition-style way.

---

## Q23. Why are difference maps important?

Difference maps show where the image changed. This is important for explainability because the user should be able to inspect whether changes are broad scanner-style changes or suspicious localized anatomical changes.

---

## Q24. What is the biggest achievement today?

The biggest achievement is that the project now has a complete non-diffusion foundation:

```text
Clean processed data
Leakage-safe splits
Strong predictor
Baseline transformations
Explainability metrics
Baseline evaluation report
```

This means the project can now move to diffusion with a reliable evaluation framework already in place.

---

# 8. Current Overall Project Status

The current project path is:

```text
Raw DICOM data
→ DICOM metadata extraction
→ Metadata sanity check
→ Metadata exploration
→ Invalid row removal
→ ax_t1 subset selection
→ Preprocessing and patient-level splits
→ Acquisition-parameter predictor
→ Baseline transformations
→ Explainability functions
→ Conditional diffusion model
→ Counterfactual editing
→ Dashboard integration
→ Final evaluation
```

Completed so far:

```text
Step 0: MVP scope frozen
Step 1: Repository/environment setup
Step 2: Dataset/environment preparation
Step 3: DICOM metadata extraction
Step 4: Metadata exploration and control selection
Step 5: Preprocessing and slice curation
Step 6: Acquisition-parameter predictor
Step 7: Baselines and explainability functions
```

Still remaining:

```text
Step 8: Conditional diffusion model
Step 9: Counterfactual editing
Step 10: Predictor-shift validation integration
Step 11: Streamlit dashboard
Step 12: Quantitative evaluation experiments
Step 13: Final report and demo assets
```

---

# 9. What To Tell the Professor

You can say:

```text
We completed the next implementation stage after metadata exploration. We preprocessed the selected ax_t1 MRI subset into normalized 128×128 slices and created patient-level train/validation/test splits to avoid data leakage. The processed subset contains 6355 slices from 103 patients, with both GE and Siemens scanners, both 1.5T and 3.0T field strengths, and usable EchoTime variation.

We then trained a multi-task acquisition-parameter predictor using a small CNN. On the held-out test set of 934 slices, it achieved 100% manufacturer accuracy, 99.57% field-strength accuracy, and EchoTime normalized MAE of 0.0546, which is better than the mean baseline. This predictor will later be used as the validation module for generated counterfactual MRIs.

Finally, we implemented baseline transformations and explainability functions. We tested identity, contrast shift, gamma correction, Gaussian noise, and histogram matching on 50 examples. The baselines behaved as expected: contrast shift preserved structure with SSIM 0.9928, gamma produced moderate intensity shift with SSIM 0.9593, Gaussian noise reduced SSIM, and histogram matching created the strongest histogram shift but reduced structural similarity. These baselines will serve as comparison points for the upcoming conditional diffusion model.
```

---

# 10. Next Steps

## Immediate Next Step: Step 8 Conditional Diffusion

The next stage is to implement the conditional diffusion model.

The implementation plan says Step 8 should use a conditional U-Net-based diffusion model with:

```text
Image size: 128×128 first
Channels: 1
Diffusion steps: 1000 during training
Sampling steps: 50 or 100 during inference
Condition dropout: 10–20%
Conditioning:
  Manufacturer → embedding
  Field strength → embedding
  EchoTime → scalar MLP embedding
```

The definition of done for this stage is that the model can sample synthetic MRI-like images conditioned on manufacturer, field strength, and EchoTime. 

## Step 8 Files To Build Next

```text
src/models/conditioning.py
src/models/diffusion_unet.py
src/training/train_diffusion.py
src/inference/sample_diffusion.py
configs/diffusion.yaml
```

## Step 8 First Goal

The first diffusion goal should be simple:

```text
Input:
Target condition = GE, 1.5T, EchoTime value

Output:
Synthetic 128×128 MRI-like slice
```

Do not start counterfactual editing immediately. First prove that conditional generation from noise works.

## After That: Step 9 Counterfactual Editing

Once conditional diffusion sampling works, then implement:

```text
Original image
→ partially noise original image
→ denoise under target condition
→ generate counterfactual image
```

This is the true core of the project, but it should come only after basic diffusion training and sampling are stable.
