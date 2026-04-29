I read the proposal. Your implementation target is a **research prototype**, not a clinical product: a 2D MRI counterfactual generation system where users choose acquisition parameters, generate a modified MRI slice, and inspect explanation panels such as difference maps, SSIM maps, histograms, and predictor-based validation. 

The key is: **do not start directly with diffusion training.** Start with the data pipeline, metadata extraction, and acquisition-parameter predictor first. Your own proposal also places these before full conditional diffusion training in the methodology and work plan. 

## Implementation Plan

## Phase 0: Freeze the MVP scope

Before coding, lock the first version to something feasible.

**MVP controls:**

1. Scanner manufacturer
2. Field strength
3. One continuous timing parameter, preferably **TE** or **TR**

**MVP output:**

1. Original MRI slice
2. Counterfactual MRI slice
3. Difference heatmap
4. Local SSIM map
5. Histogram comparison
6. Acquisition-parameter predictor output before and after generation

This matches the proposal’s objective of implementing user-facing parameter control and an explainability/validation layer. 

---

# Step 1: Create the project repository

Start with a clean, modular structure.

```text
counterfactual-mri/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│   ├── 01_metadata_exploration.ipynb
│   ├── 02_preprocessing_checks.ipynb
│   └── 03_predictor_baseline.ipynb
│
├── src/
│   ├── data/
│   │   ├── dicom_parser.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── predictor.py
│   │   ├── diffusion_unet.py
│   │   └── conditioning.py
│   │
│   ├── training/
│   │   ├── train_predictor.py
│   │   └── train_diffusion.py
│   │
│   ├── inference/
│   │   ├── counterfactual_editor.py
│   │   └── sampler.py
│   │
│   ├── explainability/
│   │   ├── difference_map.py
│   │   ├── ssim_map.py
│   │   ├── histogram.py
│   │   └── validation.py
│   │
│   └── utils/
│       ├── config.py
│       └── logging.py
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── data.yaml
│   ├── predictor.yaml
│   └── diffusion.yaml
│
├── experiments/
├── checkpoints/
├── reports/
├── requirements.txt
└── README.md
```

Use GitHub from day one. Every team member should work through branches and pull requests.

---

# Step 2: Set up the environment

Use Python with PyTorch. Your practical stack should be:

```text
pydicom
numpy
pandas
opencv-python
scikit-image
matplotlib
SimpleITK
torch
torchvision
monai
diffusers
tqdm
pyyaml
streamlit
plotly
scikit-learn
```

A good first setup command:

```bash
conda create -n cf-mri python=3.10
conda activate cf-mri
pip install torch torchvision torchaudio
pip install pydicom numpy pandas opencv-python scikit-image matplotlib SimpleITK monai diffusers streamlit plotly scikit-learn tqdm pyyaml
```

---

# Step 3: Dataset acquisition and DICOM metadata extraction

Your proposal uses the Duke-Breast-Cancer-MRI dataset and its DICOM metadata for acquisition parameters. 

First deliverable: **metadata table**.

Extract these DICOM fields:

```text
PatientID
StudyInstanceUID
SeriesInstanceUID
SOPInstanceUID
Manufacturer
MagneticFieldStrength
EchoTime
RepetitionTime
FlipAngle
SliceThickness
PixelSpacing
Rows
Columns
InstanceNumber
ImagePositionPatient
SeriesDescription
ProtocolName
```

Create:

```text
data/metadata/dicom_metadata.csv
```

Each row should represent one DICOM slice.

**Definition of done:**

You should be able to run:

```bash
python src/data/dicom_parser.py \
  --input_dir data/raw \
  --output_csv data/metadata/dicom_metadata.csv
```

And get a table like:

| patient_id | manufacturer | field_strength |  TE |  TR | flip_angle | slice_path |
| ---------- | -----------: | -------------: | --: | --: | ---------: | ---------- |
| P001       |      SIEMENS |            1.5 | 4.7 | 7.8 |         10 | path.dcm   |

This is the first real implementation milestone.

---

# Step 4: Metadata exploration and control selection

Before preprocessing images, check whether your selected controls are usable.

In `notebooks/01_metadata_exploration.ipynb`, analyze:

1. Number of patients
2. Number of slices
3. Manufacturer distribution
4. Field strength distribution
5. TE/TR distribution
6. Missing values
7. Series/protocol names
8. Whether TE or TR has enough variation

Do not choose a continuous parameter blindly. Pick the one with:

1. Low missingness
2. Meaningful variation
3. Enough samples across manufacturers/field strengths

For the MVP, your target condition vector can be:

```text
condition = [
  manufacturer_class,
  field_strength_class,
  normalized_TE_or_TR
]
```

---

# Step 5: Preprocessing and slice curation

Your proposal says the system should use pre-contrast volumes, remove non-informative edge slices, resize images, apply percentile-based normalization, and focus first on 2D slices for feasibility. 

Implement:

```bash
python src/data/preprocess.py \
  --metadata_csv data/metadata/dicom_metadata.csv \
  --output_dir data/processed \
  --image_size 256
```

Recommended preprocessing:

1. Load DICOM pixel array
2. Convert to float32
3. Clip using 1st and 99th percentile
4. Normalize to `[0, 1]`
5. Resize to `256 x 256` or `128 x 128` for early experiments
6. Remove edge slices with very low foreground content
7. Save as `.npy` or `.png`
8. Save processed metadata as `processed_metadata.csv`

Use **patient-level splits**, not random slice-level splits.

```text
train patients: 70%
val patients: 15%
test patients: 15%
```

This prevents data leakage.

**Definition of done:**

You should have:

```text
data/processed/images/
data/processed/processed_metadata.csv
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

---

# Step 6: Build the acquisition-parameter predictor first

This is critical. The proposal uses the predictor as a verification module to check whether the generated counterfactual moved toward the requested target acquisition condition. 

Model:

```text
Input: MRI slice
Backbone: ResNet18 / small CNN
Heads:
  - Manufacturer classification
  - Field strength classification
  - TE/TR regression
```

Metrics:

```text
Manufacturer accuracy
Field strength accuracy
TE/TR MAE
Confusion matrix
Validation loss
```

Suggested model file:

```text
src/models/predictor.py
```

Training script:

```bash
python src/training/train_predictor.py \
  --config configs/predictor.yaml
```

**Why this comes before diffusion:**

Because later, when your diffusion model generates a counterfactual image, this predictor becomes your automatic validator:

```text
Original image → predictor says SIEMENS, 1.5T
Generated image → predictor should move toward GE, 3.0T
```

This directly supports the proposal’s validation goal. 

---

# Step 7: Build simple baselines before diffusion

Before training diffusion, create simple baseline transformations. This gives you comparison points.

Implement:

1. Identity baseline
2. Random intensity augmentation
3. Histogram matching between manufacturers
4. Gaussian noise / contrast shift baseline

Why this matters:

If your diffusion model performs worse than histogram matching or simple contrast changes, something is wrong.

Create:

```text
src/inference/baselines.py
```

Metrics to compare:

```text
SSIM
Histogram distance
Predictor target confidence
Visual quality
```

---

# Step 8: Implement the conditional diffusion model

Now start the main model.

Your proposal specifies a conditional U-Net-based diffusion model with classifier-free guidance. 

For feasibility, start with:

```text
Image size: 128 x 128 first
Channels: 1
Diffusion steps: 1000 during training
Sampling steps: 50 or 100 during inference
Condition dropout: 10–20%
Batch size: as large as GPU allows
```

Conditioning design:

```text
Manufacturer → embedding
Field strength → embedding
TE/TR → normalized scalar → MLP embedding
Final condition vector → injected into U-Net
```

Training command:

```bash
python src/training/train_diffusion.py \
  --config configs/diffusion.yaml
```

Save:

```text
checkpoints/diffusion/latest.pt
checkpoints/diffusion/best.pt
experiments/diffusion_logs/
```

**Definition of done:**

You can sample synthetic MRI-like images conditioned on:

```text
manufacturer = GE
field_strength = 3.0T
TE/TR = target value
```

At this stage, outputs do not need to be perfect. They need to prove that the pipeline works.

---

# Step 9: Implement counterfactual editing

This is the core of the project.

The proposal describes the counterfactual process as: partially noise the original slice, then denoise it under a new target condition so that appearance changes while anatomy remains stable. 

Implement:

```text
src/inference/counterfactual_editor.py
```

Inputs:

```text
original_image
source_condition
target_condition
noise_strength / edit_timestep
guidance_scale
num_reverse_steps
```

Outputs:

```text
generated_counterfactual
metadata
run_id
```

Start with this experiment grid:

| Parameter      | Values               |
| -------------- | -------------------- |
| edit timestep  | 100, 200, 300, 400   |
| guidance scale | 1, 3, 5              |
| reverse steps  | 50                   |
| image size     | 128 first, 256 later |

Your proposal mentions the base paper selected guidance scale 3 with 50 reverse steps as a good balance, so that is a sensible starting point. 

---

# Step 10: Build the explainability module

Your proposal explicitly requires difference maps, local SSIM maps, histogram shifts, and predictor outputs. 

Implement four functions:

```text
compute_absolute_difference(original, generated)
compute_local_ssim_map(original, generated)
compute_histogram_comparison(original, generated)
compute_predictor_shift(original, generated, target_condition)
```

The dashboard should show:

```text
Original image
Generated image
Absolute difference heatmap
Local SSIM map
Original vs generated histogram
Predictor result:
  before: SIEMENS, 1.5T, TE = x
  after: GE, 3.0T, TE = y
  target: GE, 3.0T, TE = z
```

This is what turns the project from “image generation” into an **explainable counterfactual MRI platform**.

---

# Step 11: Create the Streamlit dashboard

Your proposal’s final system is an interactive dashboard for generation, inspection, and validation. 

Create:

```text
app/streamlit_app.py
```

Dashboard layout:

```text
Sidebar:
  - Select patient
  - Select series
  - Select slice
  - Select target manufacturer
  - Select target field strength
  - Select target TE/TR
  - Guidance scale slider
  - Edit strength slider
  - Generate button

Main panel:
  - Original image
  - Counterfactual image
  - Difference map
  - Local SSIM map
  - Histogram comparison
  - Predictor validation table
```

Run:

```bash
streamlit run app/streamlit_app.py
```

Important practical decision: if live generation is slow, add a **demo mode** that loads pre-generated examples. That will protect your final presentation.

---

# Step 12: Evaluation experiments

Your proposal defines three evaluation axes: anatomy preservation, conditioning success, and practical usefulness through robustness or augmentation testing. 

Minimum evaluation table:

| Evaluation goal              | Metric                               |
| ---------------------------- | ------------------------------------ |
| Anatomy preservation         | SSIM, local SSIM, visual review      |
| Target alignment             | Predictor accuracy/confidence shift  |
| Continuous parameter control | MAE toward target TE/TR              |
| Image distribution quality   | FID or MMD, optional                 |
| Practical usefulness         | Downstream robustness test, optional |

Start with these core experiments:

1. Manufacturer transfer

   ```text
   GE → SIEMENS
   SIEMENS → GE
   ```

2. Field strength transfer

   ```text
   1.5T → 3.0T
   3.0T → 1.5T
   ```

3. Continuous parameter sweep

   ```text
   TE/TR low → medium → high
   ```

4. Guidance scale ablation

   ```text
   guidance = 1, 3, 5
   ```

5. Edit strength ablation

   ```text
   low noise, medium noise, high noise
   ```

---

# Step 13: Assign work according to your team roles

Your proposal already gives a sensible role allocation. Use it directly. 

| Member         | Implementation responsibility                                |
| -------------- | ------------------------------------------------------------ |
| Naman Kundra   | DICOM parsing, metadata extraction, preprocessing, splits    |
| Kshitiz Jain   | Conditional diffusion model, U-Net, classifier-free guidance |
| Vidyt Bhudolia | Counterfactual editing engine, sampling, tuning              |
| Kavya Aggarwal | SSIM maps, difference maps, histogram analysis, evaluation   |
| Snehil Jhawar  | Streamlit dashboard, integration, documentation, demo flow   |

To avoid blocking each other, start these tracks in parallel:

```text
Naman → data pipeline
Kavya → explainability functions using sample images
Snehil → Streamlit UI skeleton with placeholder images
Kshitiz/Vidyt → diffusion code skeleton and toy MNIST/synthetic test
```

Do not wait for the full MRI dataset before building the UI and explanation modules.

---

# First 7-Day Execution Plan

## Day 1

Create GitHub repo, environment, folders, and configs.

Deliverables:

```text
README.md
requirements.txt
project folder structure
configs/data.yaml
configs/predictor.yaml
configs/diffusion.yaml
```

## Day 2

Implement DICOM metadata extraction.

Deliverable:

```text
data/metadata/dicom_metadata.csv
```

## Day 3

Explore metadata and choose final MVP parameters.

Deliverables:

```text
manufacturer distribution plot
field strength distribution plot
TE/TR missingness report
selected control parameter decision
```

## Day 4

Implement preprocessing.

Deliverables:

```text
processed 128x128 or 256x256 slices
processed_metadata.csv
patient-level train/val/test split
```

## Day 5

Train first acquisition-parameter predictor.

Deliverables:

```text
predictor checkpoint
accuracy/MAE report
confusion matrix
```

## Day 6

Build explanation functions and test them on original-vs-augmented images.

Deliverables:

```text
difference map function
SSIM map function
histogram comparison function
sample visualization grid
```

## Day 7

Build Streamlit dashboard skeleton.

Deliverables:

```text
slice selector
target parameter controls
placeholder output panels
predictor output table placeholder
```

At the end of week 1, you should have a working non-diffusion prototype.

---

# Recommended Implementation Order

Follow this exact order:

```text
1. Repo + environment
2. DICOM metadata extraction
3. Metadata exploration
4. Preprocessing
5. Patient-level splits
6. Acquisition-parameter predictor
7. Baseline transformations
8. Explainability module
9. Streamlit dashboard skeleton
10. Conditional diffusion training
11. Counterfactual editing
12. Dashboard integration
13. Quantitative evaluation
14. Final report and demo assets
```

This order minimizes risk. Even if diffusion takes time, you will still have a demonstrable system with data, metadata, predictor, explainability, and dashboard components.

---

# Minimum Viable Demo Target

Your first complete demo should do this:

```text
Input:
  Select one MRI slice.

User controls:
  Target manufacturer
  Target field strength
  Target TE/TR

Output:
  Original image
  Generated counterfactual
  Difference map
  SSIM map
  Histogram shift
  Predictor before/after validation
```

That directly satisfies the proposal’s expected outcomes: cleaned MRI slice dataset, acquisition-parameter predictor, conditional diffusion generation pipeline, explainability module, dashboard, and final demo assets. 
