# Reversible geometry and verified native annotations, stage 3

This stage provides a real train/validation geometry view, request-support
diagnostics and verified native-grid public MANUAL tissue annotations. It
does not retrain or silently replace the current editor and is not clinical
validation of generated outputs. All historical data/reports/models are preserved.

## Canonical view

`orientation.py` derives each transform from DICOM ImageOrientationPatient,
using the [DICOM patient-coordinate definition](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).
The first IOP triplet follows increasing **column** index, the second increasing
**row** index. Canonical columns point towards LPS +X (patient left); rows
towards +Y (posterior). Vendor labels do not choose the transform.

Only exact transpose/flips are allowed. Missing, non-orthonormal, non-axial or
oblique geometry is rejected. The transform is invertible for pixels, masks and
landmarks, including rectangular grids. Physical maps record the declared
whole-image, centre-aligned resize, without pretending slice thickness is a
verified inter-slice distance. `audit_view.py` reproduces the native-DICOM
preprocessing on its selected sources before trusting that assumption.

The index is lazy: original arrays remain in place, and the manifest records
source paths, hashes and transformations. Call `prepare_view.load_canonical`.
There is deliberately no legacy `image_path` column; an old loader must not
silently read native arrays and label them canonical. Use the saved inverse to
return any future generated image or annotations to the source pixel grid.

The current checkpoints learned the historical frame. They have **not** been
validated on the canonical view, and the API is not changed. Canonicalisation
does not register anatomy across patients or remove protocol/cohort confounding.

## Condition support

`support.py` checks target label-cell patient coverage and fixed-source-EchoTime
range/exact support from training metadata. It distinguishes missing/sparse cells,
outside-range requests, interpolation-only values and observed conditions.
An unseen exact value inside a continuous range is not proof of invalidity;
likewise being inside range is not proof of valid causal support. These are
engineering diagnostics, never clinical approval. The strict research helper
requires an explicit override for unverified conditions. EchoTime is not edited.

## Native DCE annotations

`probe_public_seg.py` inspects one size-capped public SEG reference sample.
It never downloads a collection or applies its mask to a mismatched image.
Matching patient IDs or FrameOfReferenceUID alone are insufficient.

`native_annotations.py` uses official catalogue entries and **already local**
train/validation DCE images/SEG objects. Each accepted frame must have one
exact referenced source SOP, the correct patient, MANUAL algorithm type,
matching dimensions, orientation, spacing and plane position. Missing or
mismatched frames are rejected. Source intensities are rescaled but not resized
or normalised; masks remain binary on the exact native grid.

The dataset is separate from ax-t1. FGT and fatty-tissue labels retain their
original semantics; do not automatically call them tumour/whole-breast masks.
They annotate real DCE source images, not our generated outputs. The small
subject count and scanner coverage are profiled, not hidden. Additional local
annotation candidates outside recorded original/judge splits are identified
without decoding them; they are not yet an approved fresh test cohort.

## Commands

From `counterfactual-mri-generation/`, system Python:

```powershell
python -m unittest scripts.v7_geometry.test_geometry scripts.v7_geometry.test_annotations -v
python -m scripts.v7_geometry.prepare_view --run-id <new_view_id>
python -m scripts.v7_geometry.audit_view --view data/processed/ax_t1_128_canonical/<new_view_id> --output reports/v7_geometry/<new_audit_id>
python -m scripts.v7_geometry.probe_public_seg --output reports/v7_geometry/<new_small_probe_id>
python -m scripts.v7_geometry.native_annotations --run-id <new_native_annotation_id>
```

All output folders must be new. `geometry_view_v1.json` predeclares train/val
only. No test image or existing checkpoint is read for these experiments.
The optional public probe caps a single SEG object at 1 MB and its bucket
listing at 64 KB. No MRI collection download or external contact is invoked.
`verify_stage.py --help` saves a new complete provenance/test/profile receipt.

## Next efficacy work

Measure residual acquisition/protocol leakage, validate model training in the
chosen coordinate frame, and predeclare matched baselines and preservation
outcomes before a larger run. A DCE model/data branch would require its own
splits, conditioning and assessor validation; do not reuse ax-t1 performance
claims or silently merge the native tissue data into the current paper.
