# Held-out judge: local data preparation

Run these modules from `counterfactual-mri-generation/` with system Python.
The original data, reports, checkpoints and papers are never overwritten.

1. `python -m scripts.v4_heldout.audit --run-id <new-id>` reads the packaged IDC
   catalogue and local metadata. It creates a new immutable preflight report,
   a 64-patient plan (48 train / 16 val), original-cohort exclusion list,
   clinical-label manifests and mask inventory. No network download occurs.
2. `python -m scripts.v4_heldout.download --plan-run <id>` verifies the plan and
   prints its local dry run. After explicit user approval, add
   `--execute --max-gb <approved-decimal-GB>`. It downloads only planned series,
   checks DICOM headers and writes separate receipts. It refuses an existing raw
   output directory. If interrupted, preserve the partial files/receipts and
   inspect them before creating an explicit recovery procedure; do not silently
   restart into the same folder. The byte guard checks between series, so an
   unexpected size can exceed the cap by the current series.
3. `python -m scripts.v4_heldout.preprocess --plan-run <id>` requires a completed
   download receipt, reuses original preprocessing functions, label maps and echo
   statistics, preserves the planned split, and creates a float32 training/val
   pack plus a visual QA sheet. No original test images enter the pack. Inspect
   `preprocessing_qc.png` before training. Outputs use fresh run-specific folders.

The data-preparation commands do not launch Modal or train judges. The old judge script hardcodes
the original volume filenames and must not be pointed at this cohort unchanged.
A future versioned runner needs explicit input/output namespaces and a separate
approved cloud allowance, after the current B1 app finishes. Three robust seeds
must select on the new validation patients before testing on the original cohort.

`train_judge.py` now prepares that training core with explicit pack/output paths,
three prespecified seeds, robust augmentation, clean/noisy validation selection,
data-hash and patient-exclusion guards. It is not launched. A synthetic CPU
forward/backward smoke is recorded in
`reports/v4_heldout_judge/runner_smoke_20261016_01/`; the real-pack loader will also
be checked after preprocessing. No original test images are loaded by this core.

The completed QA/header audit (`orientation_20261016_01`) finds a vendor-linked
orientation cue in every retained original/new slice. `train_judge.py` therefore
adds an independent vertical flip with probability 0.5 for the PRIMARY new judges,
alongside the existing horizontal/intensity/noise/shift augmentation. Use
`--legacy-horizontal-only` only for the original-recipe comparison. Validation
selection stays the same; vertical-flip and 180-degree stress predictions are
reported after selecting the checkpoint. This is a prepared control, not evidence
that orientation invariance or useful discrimination has been achieved. No
generator inputs or saved images are reoriented.

`audit_masks.py` fetches three small official TCIA metadata files with bounded
response size. It maps the official 2D breast/FGT SEG manifest to the cached IDC
entries, audits sequence compatibility and patient overlap, and writes a new
`reports/v4_downstream/<run-id>/`. It never downloads masks. The completed
`mask_feasibility_20261016_01` audit establishes that patient overlap alone is
insufficient: its filepath mapping has no exact `ax t1` series.

Current authorised plan: `preflight_20261016_01`; user approved approximately
1.24 GB, with `--max-gb 1.5`, and local preprocessing. B2 remains incomplete until
judge checkpoints, accuracy and re-grades exist. B5's label manifests establish
availability only, not utility or label independence from vendor.

The initial AWS transfer stopped on a recorded connection error with an
incomplete series. `resume_download.py --plan-run preflight_20261016_01
--attempt-id recovery_01` explicitly continues the same plan via IDC's GCS mirror,
skips complete series, syncs missing objects, verifies every header, and checks
the hashes of all files present before recovery. It preserves the failed receipt
and writes a new receipt folder. Preprocessing must use
`--download-attempt recovery_01` to read that completed recovery receipt.

Keep all 64 new patients out of DDPM training and generator/operating-point
selection. They are reserved for the independent judge's train/validation split.
The prepared pack is not an expanded diffusion-training dataset.

After preprocessing, run `python -m scripts.v4_heldout.finalize --plan-run
preflight_20261016_01 --run-id <new-snapshot-id>` to exercise the real
training loader's patient-exclusion/range/hash checks, recheck original input
hashes and freeze a source snapshot. Inspect the QA image separately. This step
does not train a classifier or start Modal.

`implementation_20261016_01` preserves the trainer before the orientation audit;
`implementation_20261016_02` is the current snapshot/protocol after adding the
orientation control. The latter's training_protocol.json is the prepared recipe.

## Cloud execution preparation, later on 2026-10-16

`orientation_probe.py` ran quick local inference on 100 original VALIDATION slices
with the existing seven judges. Its report is `orientation_probe_20261016_01/`.
No test data or paid cloud app was used. All three existing robust judges changed
manufacturer decisions on every selected image after a vertical flip, despite
unchanged scanner labels. This supports the new orientation control; its success
still requires training and evaluation.

`prepare_run.py --run-id b2_20261016_01` freezes the new training pack and 13
existing diffusion/baseline sets, with explicit original-test index maps.
`test_b2_pipeline.py` checks integrity, separation, CPU integration, partial-model
preservation, uncertainty and budget/active-app guards. Synthetic software-test
metrics/checkpoints in tmp/ and test logs MUST NOT be cited as research results.
An initial Windows namespace/Modal decorator import issue was fixed by keeping
pure budget constants in budget.py. Avoid importing Modal decorators as normal
package modules on Windows; the file-based `modal run .../modal_app.py --help`
path loads without creating an app.

After checks, `seal_run.py --run-id b2_20261016_01` seals code/inputs locally.
`launch.py --run-id b2_20261016_01` is a read-only preflight. Only after separate
human approval use `--execute --wait --approved-budget-usd 5`. This waits LOCALLY
until every existing Modal app ends before dispatching one detached B2 app.
The computer must remain online until dispatch. No queue is started by preparing
files. `modal_app.py` uses one A10 sequentially with full-timeout reservations
and conservative budget receipts; shared volume filenames remain unchanged.

The controller trains three primary and three legacy-control seeds, freezes all
six checkpoints, then calls evaluate_heldout.py on all original 934 test slices
and saved old/B1-B3 edits. Descriptive judge-validity flags use new validation
only, never filter test results, and never trigger checkpoint retuning. A
deadline saves a partial checkpoint and stops before testing if incomplete.
Uncertainty includes patient bootstrap and Wilson context at binary boundaries.
Missing B3 completion is explicit rather than claimed as complete.

Use `python -m scripts.v4_heldout.fetch --run-id b2_20261016_01` to download
committed completed stages and export new checkpoints without replacing files.
Earlier snapshots remain historical; b2_20261016_01's seal is the current cloud
implementation (the trainer additionally preserves partial state on timeout).
The separate US$5 allowance is proposed, not approved. B1/B3's US$15 stays
allocated to its existing controller.
