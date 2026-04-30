Below is a **full progress and decision report** you can keep in `reports/current_progress_report.md` and also use while explaining work to your professor/evaluator.

---

# Progress Report: Dataset Preparation, Metadata Extraction, and Control Selection

## Project Title

**Explainable Counterfactual MRI Generation and Acquisition Parameter Control for Robust Medical Imaging AI**

## 1. Project Context

The goal of this capstone project is to build an **explainable and controllable MRI transformation system**. The system should allow a user to select an MRI slice and modify selected acquisition parameters such as scanner manufacturer, field strength, and one continuous timing parameter. The generated counterfactual MRI should preserve anatomy while changing scanner/acquisition-related appearance. This aligns with the proposal’s goal of building a research prototype for domain-shift simulation, robustness testing, and educational visualization in medical imaging. 

The final system is expected to include:

* A cleaned MRI slice dataset with structured acquisition metadata.
* An acquisition-parameter predictor.
* A conditional diffusion-based counterfactual generation pipeline.
* Explainability outputs such as difference maps, local SSIM maps, histogram shifts, and predictor-based validation.
* An interactive dashboard for generation and inspection. 

At this stage, we have focused on the **data foundation**, because the later predictor and diffusion model depend entirely on correct metadata extraction and careful control selection.

---

# 2. Work Completed Till Now

## 2.1 Repository and Environment Setup

We created the project repository structure according to the implementation plan. The planned project structure separates raw data, processed data, metadata, source code, scripts, configs, reports, experiments, and checkpoints. This modular structure is important because the project has multiple stages: DICOM parsing, preprocessing, predictor training, diffusion training, explainability, and dashboard integration. 

The environment was set up with the required Python stack for medical image processing and deep learning, including:

* `pydicom`
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-image`
* `SimpleITK`
* `torch`
* `torchvision`
* `monai`
* `diffusers`
* `streamlit`
* `plotly`
* `scikit-learn`

This stack supports DICOM parsing, metadata handling, preprocessing, visualization, model training, and future dashboard development.

---

## 2.2 Dataset Download

We downloaded an expanded subset of the **Duke-Breast-Cancer-MRI** dataset into:

```text
data/raw/
```

Initially, only a small subset was downloaded. That subset had:

```text
2732 slices
4 patients
20 series
```

This was useful for testing the scripts, but not enough for training or reliable patient-level splitting.

After expanding the dataset, the metadata extraction showed:

```text
101,765 slices
123 patients
123 studies
811 series
```

This confirms that the larger dataset is now being correctly read and parsed. 

---

# 3. DICOM Metadata Extraction

## 3.1 Purpose

The purpose of metadata extraction is to convert raw DICOM files into a structured CSV table where each row represents one MRI slice. This is a critical first implementation milestone because the model cannot learn acquisition-parameter control unless each image is linked with its acquisition metadata.

The implementation plan specifically required creating:

```text
data/metadata/dicom_metadata.csv
```

with one row per DICOM slice and fields such as `PatientID`, `Manufacturer`, `MagneticFieldStrength`, `EchoTime`, `RepetitionTime`, `FlipAngle`, `SliceThickness`, image dimensions, `SeriesDescription`, and `slice_path`. 

## 3.2 Fields Extracted

The parser extracts these key DICOM fields:

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
slice_path
```

## 3.3 Result

After running the parser on the expanded dataset, the output was:

```text
101,765 rows × 18 columns
```

This means the script successfully parsed **101,765 DICOM slice records**. 

## 3.4 Why This Step Matters

This table is the backbone of the whole project.

It will later be used for:

* Selecting usable MRI series.
* Filtering invalid/non-image rows.
* Choosing acquisition control parameters.
* Creating train/validation/test splits.
* Training the acquisition-parameter predictor.
* Conditioning the diffusion model.
* Validating generated counterfactuals.

Without this metadata table, the project would only have images but no way to control or verify acquisition parameters.

---

# 4. Metadata Exploration

## 4.1 Purpose

After extraction, we performed metadata exploration to check whether the selected MVP controls are actually usable.

The implementation plan clearly says that before preprocessing images, we must analyze:

* Number of patients.
* Number of slices.
* Manufacturer distribution.
* Field strength distribution.
* TE/TR distribution.
* Missing values.
* Series/protocol names.
* Whether TE or TR has enough variation.

It also says not to choose TE or TR blindly; the selected parameter should have low missingness, meaningful variation, and enough samples across manufacturer and field-strength groups. 

## 4.2 Dataset Scale After Expansion

The expanded dataset summary is:

```text
Number of slices: 101,765
Number of patients: 123
Number of studies: 123
Number of series: 811
```

This is a major improvement over the initial 4-patient subset and is sufficient to begin preprocessing and baseline predictor work. 

---

# 5. Manufacturer Distribution

## 5.1 Observed Distribution

The expanded dataset contains:

```text
GE MEDICAL SYSTEMS: 73,622 slices
SIEMENS: 28,012 slices
QIICR: 131 slices
```

## 5.2 Interpretation

This is good because the project’s MVP requires **scanner manufacturer control**. The two main scanner manufacturers are present:

```text
GE MEDICAL SYSTEMS
SIEMENS
```

However, the distribution is imbalanced:

```text
GE > SIEMENS
```

This means that when training the predictor or diffusion model, we may need to use class balancing, weighted sampling, or carefully selected subsets so the model does not become biased toward GE.

## 5.3 Why QIICR Was Removed

The `QIICR` rows are not normal MRI acquisition images. They correspond to segmentation/annotation-related series such as:

```text
AIMI breast-tumor AI segmentation
Segmentation
AIMI breast-tumor radiologist corrected segmentation
```

These rows also contain missing acquisition metadata such as field strength, TE, and TR. Therefore, they are not suitable for acquisition-parameter prediction or counterfactual MRI generation.

Decision:

```text
Remove QIICR rows from cleaned metadata.
```

Reason:

```text
They are segmentation/label objects, not normal scanner-acquired MRI slices.
```

---

# 6. Field Strength Distribution

## 6.1 Observed Distribution

The field strength distribution is:

```text
1.5T: 48,837 slices
3.0T: 52,797 slices
NaN: 131 slices
```

## 6.2 Interpretation

This is very good. The two main field strength values are almost balanced:

```text
1.5T ≈ 3.0T
```

This makes **field strength** a strong control variable for the MVP.

The 131 missing values correspond to the QIICR segmentation-related rows. After removing QIICR rows, the field strength metadata becomes usable.

Decision:

```text
Use MagneticFieldStrength as one of the main discrete control variables.
```

Reason:

```text
Both 1.5T and 3.0T are present in large numbers and are approximately balanced.
```

---

# 7. EchoTime vs RepetitionTime Selection

The MVP requires one continuous timing parameter. The two main candidates are:

```text
EchoTime
RepetitionTime
```

## 7.1 EchoTime Summary

From metadata exploration:

```text
Valid EchoTime values: 101,634
Mean: 2.72518
Standard deviation: 2.70003
Minimum: 1.34
Maximum: 106.632
Unique values: 90
```

## 7.2 RepetitionTime Summary

From metadata exploration:

```text
Valid RepetitionTime values: 101,634
Mean: 46.9371
Standard deviation: 173.773
Minimum: 3.54
Maximum: 5000
Unique values: 129
```

## 7.3 Why RepetitionTime Was Not Selected First

Although RepetitionTime has more unique values, its distribution is highly skewed. Most dynamic MRI series have TR values around:

```text
3.5 to 7
```

But some sequence types have very large TR values:

```text
450 to 786
5000
```

For example:

```text
ax t1 series: TR around 450–786
AX IDEAL Breast: TR = 5000
```

This shows that TR is strongly affected by **sequence type**. If we use TR too early without careful sequence filtering, the model may learn differences between MRI sequence types instead of learning controlled acquisition-parameter effects.

## 7.4 Why EchoTime Was Selected

EchoTime also has some outliers, but compared with RepetitionTime, it is less extreme and easier to use as the first continuous control. It has:

* Very low missingness.
* 90 unique values.
* Meaningful variation.
* Availability across GE and Siemens.
* Availability across 1.5T and 3.0T.

Decision:

```text
Selected continuous parameter: EchoTime
```

Reason:

```text
EchoTime has low missingness, meaningful variation, and is less distorted by sequence-type outliers than RepetitionTime. Therefore, EchoTime is safer for the first MVP condition vector.
```

The first MVP condition vector becomes:

```python
condition = [
    manufacturer_class,
    field_strength_class,
    normalized_echo_time
]
```

This directly matches the implementation plan’s recommended target condition vector: manufacturer class, field strength class, and normalized TE/TR. 

---

# 8. Series-Level Analysis

## 8.1 Why Series-Level Analysis Was Needed

MRI datasets contain many different types of series. These series may differ by:

* Orientation.
* Sequence type.
* Contrast phase.
* Scanner protocol.
* Manufacturer.
* Field strength.
* Timing parameters.

If all series are mixed blindly, the model may learn the wrong relationship. For example, it might learn the difference between pre-contrast and post-contrast images instead of learning the difference between GE and Siemens scanners.

This is called a **confounding problem**.

Therefore, we ran a `series_checker.py` script to inspect each `SeriesDescription` separately.

## 8.2 Important Series Observed

The top series included:

```text
ax t1
ax dyn 1st pass
ax dyn 3rd pass
ax dyn pre
ax dyn 2nd pass
Ph2/ax 3d dyn MP
Ph3/ax 3d dyn MP
ax 3d dyn MP
Ph1/ax 3d dyn MP
Ph4/ax 3d dyn MP
Ph1/ax 3d dyn
ax 3d dyn
Ph2/ax 3d dyn
Ph3/ax 3d dyn
```

The expanded summary showed `ax t1` as the largest single useful series with 6355 slices. 

---

# 9. Why `ax t1` Was Selected for the First Preprocessing Subset

## 9.1 Observed `ax t1` Properties

From the series checker:

```text
Series: ax t1
Slices: 6355
Patients: 103
Manufacturers: GE MEDICAL SYSTEMS and SIEMENS
Field strengths: 1.5T and 3.0T
EchoTime: multiple values
RepetitionTime: multiple values
```

## 9.2 Why This Is a Good First Subset

`ax t1` is currently the best single-series candidate for a first preprocessing and predictor baseline because it contains:

* A large number of slices.
* A large number of patients.
* Both scanner manufacturers.
* Both field strengths.
* Multiple EchoTime values.

This means it supports all three MVP controls:

```text
Manufacturer
Field strength
EchoTime
```

## 9.3 Why Not `ax dyn pre` First?

The project proposal is ultimately motivated by pre-contrast dynamic MRI and diffusion-based counterfactual generation. However, in the expanded dataset, `ax dyn pre` contains:

```text
5624 slices
34 patients
SIEMENS only
1.5T and 3.0T
multiple EchoTime values
```

This is useful for Siemens-only dynamic analysis, but it cannot support GE ↔ Siemens manufacturer transfer because it contains only Siemens.

Therefore:

```text
ax dyn pre is not selected for the first manufacturer-control predictor baseline.
```

## 9.4 Final Subset Decision

Decision:

```text
Selected first preprocessing subset: ax t1
```

Reason:

```text
It is the strongest single-series subset for the first baseline because it contains both GE and Siemens, both 1.5T and 3.0T, enough patients, and usable EchoTime variation.
```

Important limitation:

```text
ax t1 is selected for the first preprocessing and acquisition-parameter predictor baseline. It may not be the final subset for the diffusion-based counterfactual generation stage. For the final diffusion stage, a dynamic/pre-contrast series family may still be curated separately.
```

This is a practical staged decision, not a change in the final project goal.

---

# 10. Clean Metadata Creation

## 10.1 Purpose

Before preprocessing images, we created a cleaned metadata subset to remove invalid rows and keep only meaningful MRI acquisition data.

The cleaning logic removes:

```text
QIICR rows
segmentation rows
rows with missing field strength
rows with missing EchoTime
rows with missing RepetitionTime
rows with missing required metadata
```

## 10.2 Why This Cleaning Is Important

The predictor and diffusion model should be trained only on real MRI slices with valid acquisition metadata. If segmentation masks or invalid rows are included, the model may learn irrelevant image patterns or fail during preprocessing.

## 10.3 Created Subset

The selected subset is:

```text
ax_t1_metadata.csv
```

This subset will be used for Step 5 preprocessing.

---

# 11. Current Status of Implementation

## Completed

```text
Step 0: MVP scope frozen
Step 1: Repository structure created
Step 2: Environment setup completed
Step 3: DICOM metadata extraction completed
Step 4: Metadata exploration completed
Step 4.1: Series-level suitability analysis completed
Step 4.2: Clean metadata/subset creation completed
```

## Current Main Decisions

```text
Discrete control 1: Manufacturer
Discrete control 2: MagneticFieldStrength
Continuous control: EchoTime
First preprocessing subset: ax t1
Invalid rows removed: QIICR and segmentation-related rows
```

## Next Step

The next implementation stage is:

```text
Step 5: Preprocessing and slice curation
```

This step should:

* Load DICOM pixel arrays.
* Convert images to `float32`.
* Clip intensities using 1st and 99th percentiles.
* Normalize images to `[0, 1]`.
* Resize to 128×128 first for speed.
* Remove low-content/non-informative edge slices.
* Save processed images.
* Save processed metadata.
* Create patient-level train/validation/test splits.

The implementation plan specifically recommends patient-level splitting rather than random slice-level splitting to prevent data leakage. 

---

# 12. Why We Are Not Training the Model Yet

We are not training the acquisition predictor or diffusion model immediately because the project must first ensure that:

```text
1. The dataset is correctly parsed.
2. The metadata is reliable.
3. The selected controls are available.
4. Invalid rows are removed.
5. A consistent image subset is selected.
6. Patient-level splits are created.
```

Training before this would be risky because the model could learn artifacts from mixed series, segmentation masks, or data leakage.

This is why the implementation plan puts the acquisition-parameter predictor after metadata extraction, metadata exploration, preprocessing, and patient-level splits. 

---

# 13. Evaluation-Relevant Justifications

## Question: Why did we extract metadata first?

Because the entire project depends on acquisition parameters. The model must know which scanner manufacturer, field strength, and timing parameter correspond to each MRI slice. Without metadata, there is no way to train a controllable model.

## Question: Why did we not directly train diffusion?

Because diffusion training is expensive and sensitive to data quality. We first need a clean dataset and a predictor baseline. The predictor later becomes the validation module that checks whether generated counterfactuals moved toward the target condition. 

## Question: Why did we choose EchoTime?

EchoTime was chosen because it has very low missingness, 90 unique values, meaningful variation, and is less affected by extreme sequence-type outliers than RepetitionTime.

## Question: Why not RepetitionTime?

RepetitionTime has useful information, but its distribution is highly skewed. Some MRI sequence types have TR values around 3–7, while others have TR values in the hundreds or even 5000. This indicates that TR is more strongly tied to sequence type, so using it too early could introduce confounding.

## Question: Why did we remove QIICR?

QIICR rows are segmentation/annotation-related, not normal scanner-acquired MRI slices. They also have missing acquisition parameters, so they are not suitable for this acquisition-control task.

## Question: Why did we select `ax t1`?

Because it is the strongest single-series subset for the first baseline. It has 6355 slices, 103 patients, both GE and Siemens, both 1.5T and 3.0T, and multiple EchoTime values. This supports all MVP controls.

## Question: Why not `ax dyn pre`?

`ax dyn pre` is useful and closer to the final dynamic MRI direction, but in the current downloaded subset it contains only Siemens data. Since manufacturer transfer is part of the MVP, it is not enough for the first manufacturer-control predictor baseline.

## Question: Is selecting `ax t1` a change from the proposal?

No. It is a staged implementation decision. The final project still targets controllable MRI counterfactual generation. `ax t1` is selected first because it is the cleanest and most balanced subset for building the first working preprocessing and predictor pipeline. Later, dynamic/pre-contrast series can be curated for diffusion experiments.

## Question: Why use patient-level splits?

Because slice-level random splitting can cause data leakage. Slices from the same patient are highly similar. If slices from one patient appear in both training and validation, the model may appear to perform well without actually generalizing.

## Question: What will be shown in the next evaluation?

The next evaluation milestone can show:

```text
1. Extracted metadata CSV.
2. Metadata distribution plots.
3. Cleaned metadata decision report.
4. Selected ax_t1 subset.
5. Preprocessed MRI slices.
6. Patient-level train/val/test splits.
7. Initial predictor training plan.
```

---

# 14. Deliverables Created So Far

The following implementation artifacts have been created or are expected to exist now:

```text
src/data/dicom_parser.py
scripts/dicom_metadata_sanity_check.py
scripts/metadata_exploration.py
scripts/series_checker.py
scripts/create_clean_metadata.py
data/metadata/dicom_metadata.csv
data/metadata/clean_metadata.csv
data/metadata/ax_t1_metadata.csv
reports/metadata_summary.md
reports/figures/manufacturer_distribution.png
reports/figures/field_strength_distribution.png
reports/figures/echo_time_distribution.png
reports/figures/repetition_time_distribution.png
```

Raw DICOM data should **not** be committed to GitHub.

Full metadata CSV files should also generally not be committed if they are large. Instead, commit:

```text
code
scripts
reports
figures
small sample metadata CSV
README updates
.gitignore
```

---

# 15. Suggested GitHub Commit Message

Use this commit message:

```bash
git add scripts src reports README.md .gitignore
git commit -m "Complete metadata extraction and select first MRI preprocessing subset"
```

If you also want a more detailed commit message:

```bash
git commit -m "Analyze expanded Duke MRI metadata and select ax_t1 subset for baseline preprocessing"
```

---

# 16. Short Professor Update

You can say this:

> We completed the data foundation stage of the project. We expanded the Duke MRI dataset and successfully extracted DICOM metadata for 101,765 MRI slices from 123 patients and 811 series. We analyzed scanner manufacturer, field strength, TE/TR distributions, missing values, and series descriptions. Based on this analysis, we selected manufacturer and field strength as discrete controls and EchoTime as the first continuous timing control. We removed segmentation-related QIICR rows and selected the `ax t1` series as the first preprocessing subset because it contains both GE and Siemens scanners, both 1.5T and 3.0T field strengths, 103 patients, and usable EchoTime variation. The next step is preprocessing this subset into normalized 2D slices and creating patient-level train/validation/test splits.

---

# 17. Current Conclusion

The project is currently on track.

The most important achievement is that the team has moved from a vague dataset idea to a structured, evidence-backed dataset selection process. The decisions are justified by metadata statistics rather than assumptions.

The current selected path is:

```text
Raw DICOM data
→ DICOM metadata extraction
→ metadata sanity check
→ metadata exploration
→ invalid row removal
→ ax_t1 subset selection
→ preprocessing and patient-level splits
→ acquisition-parameter predictor
→ baseline transformations
→ diffusion model
→ counterfactual editing
→ explainability/dashboard
```

This is a strong and defensible foundation for the next implementation stage.
