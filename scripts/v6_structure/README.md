# Source-structure conditioning, stage 2

This is a tested research prototype and bounded optimisation/inference check.
It is not a trained, validated anatomy-preserving editor. The frozen epoch-110
U-Net and acquisition conditioner remain unchanged; only a small new adapter
is updated. Original checkpoints, reports and paper figures are preserved.

## Architecture and objectives

`model.py` extracts three coarse 32x32 contrast-normalised gradient maps from
each source. Raw source intensities and full-resolution source pixels do not
enter the adapter. A lightweight convolutional branch produces zero-initialised
spatial residuals at the U-Net's 128/64/32 scales. Zero initialisation recovers
the base model exactly. Disabling the branch after training also recovers it.
Both classifier-free guidance arms use the same source guide.

The frozen base is forced into eval mode, including dropout. Training uses
only original training patients, real-image noise prediction, geometry-preserving
photometric perturbation before guide extraction, independent class/structure
dropout, and a low-timestep structure-proxy consistency term. The clean source
is a denoising target, not an unrestricted conditioning shortcut. This is still
self-conditioned reconstruction training: no paired target-scanner ground truth
or expert anatomy/lesion supervision is available. It cannot prove copying or
scanner leakage has been eliminated.

## Important negative finding

The representation audit detects strong manufacturer information in these maps.
The source orientation and acquisition protocol confounds remain. Do not call
the maps scanner-invariant anatomy, and do not launch full efficacy training
based on the smoke results. The engineering run verifies wiring and optimisation
only. A weak leakage probe would not have proven invariance either.

## Commands

From `counterfactual-mri-generation/`, using system Python:

```powershell
python -m unittest scripts.v6_structure.test_model -v
python -m scripts.v6_structure.audit_representation --output reports/v6_structure/<new_audit>
python -m scripts.v6_structure.train_smoke --run-id <new_engineering_run>
python -m scripts.v6_structure.evaluate_smoke --checkpoint checkpoints/diffusion/ax_t1_128_v6_structure/<new_engineering_run>/adapter_smoke.pt --output reports/v6_structure/<new_inference_run>
python -m scripts.v6_structure.render_smoke --pilot reports/v6_structure/<new_inference_run>
python -m scripts.v6_structure.annotation_resources --output reports/v6_structure/<new_resource_check>
```

`configs/structure_adapter_v1.json` is a new versioned protocol. The smoke
runner is capped at 128 optimisation steps; default is 32. It deliberately
cannot be presented as full training. All output IDs/folders must be new.
It invokes no cloud service, data downloader, account registration or messaging.
The checkpoint stores only adapter weights plus the exact base reference/hash.

## Checks and controls

- Zero adapter matches the original noise predictor exactly.
- No-guide inference matches the original sampler bitwise after adaptation.
- Wrong-source guides come from different validation patients; both correct
  and wrong guides use identical target conditions and noise seeds.
- Gradients reach the adapter encoder, never the frozen base/conditioner.
- Raw-image guide inputs and malformed presence gates are rejected.
- Feature sensitivity to intensity and spatial changes is checked separately.
- One source per patient in the leakage audit; fixed probe, no tuning or test images.
- Validation figures retain every request, with each held-out judge reported
  separately and patient-bootstrap intervals. Small/degenerate CIs remain limited.
- Human preservation outcomes remain empty until actual labels/reviews arrive.

Use `verify_stage.py --help` to verify an entire stage and save a fresh source
snapshot/test receipt. The software verification is distinct from anatomy validation.

## Annotation resources and next decisions

The resource catalogue lists Duke's expert tissue masks and MAMA-MIA expert
tumour masks, their primary references, sequence mismatch and access routes.
These are real labels for DCE-related scans, not verified labels for current
non-fat-suppressed ax-t1 arrays or generated outputs. Existing local `seg_masks`
were made from bounding boxes; do not treat them as tissue segmentation truth.

`excluded_patients.csv` records the original and held-out-judge patients to
exclude from any claim of a new untouched Duke evaluation cohort. Downloading
a different sequence also requires a new data/model protocol and independent
geometry/series matching, not resizing existing labels. No bulk data was fetched.

Institutional reviewer routes and a brief are prepared but nobody was contacted
or recruited. Blinded PNGs and blank mask/landmark/reader templates exist for
both the earlier pilot and the new smoke outputs. `scripts/v5_anatomy/score_review.py`
can ingest their completed responses. Unknown labels remain unknown.

Before full model development: establish independent anatomy evaluation; address
orientation/protocol shortcuts in a new data view; predeclare efficacy training
and controlled comparisons; and reserve a genuinely untouched final cohort.
No public-data mask can retrospectively certify preservation in a generated image.
