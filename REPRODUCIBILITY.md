# Reproducing the frozen evaluation

Prepared 16 November 2026. This guide distinguishes provenance verification,
rebuilding the manuscript and expensive regeneration. Existing evidence is
immutable. No clinical preservation validation or fresh training is implied.

## Runtime

Use Windows and Python 3.12 for the recorded local path. The actual package
snapshot and `pip check` output are in
`reports/v11_submission_20261116_02/environment{.json,-observed.txt}` and
`pip-check-observed.txt`. The historical system environment has a NumPy/SciPy
constraint conflict and unrelated installed-package conflicts. It is not a
clean-install lock, even though local verification passes.

`environment/requirements-evaluation-candidate.txt` pins a proposed compatible
scientific/API stack (NumPy 1.26.4 with SciPy 1.14.1). It is separate from the
historical environment. `requirements-torch-cu128.txt` pins the actual tensor
pair. A clean installation, `pip check`, inference parity and numerical metrics
must pass before this candidate can be called a validated release environment.
The present work does not silently alter the global environment or download
large tensor wheels. A resolver check, when present under
`reports/v11_environment_resolution_20261116_01/`, is not installation validation.
That check succeeded; `requirements-resolved.txt` there freezes all selected
non-tensor transitive packages and wheel hashes. Use that file in place of the
candidate input list when installing the exact resolved Windows/Python 3.12
stack. The tensor pair remains a separate install with its own dependency check.

Provisioning commands for a new machine, after arranging authorised data/model
access and the large tensor download:

```powershell
py -3.12 -m venv .venv-publication
.\.venv-publication\Scripts\python.exe -m pip install -r environment/requirements-torch-cu128.txt
.\.venv-publication\Scripts\python.exe -m pip install -r environment/requirements-evaluation-candidate.txt
.\.venv-publication\Scripts\python.exe -m pip check
.\.venv-publication\Scripts\python.exe -m unittest api.test_contract api.test_dashboard_fixes -v
```

Activate that environment before launching `start_demo.bat`. CUDA is unnecessary
for exact saved replay. The current API imports torch even in replay mode;
the complete saved demo is not a dependency-free static site. Modal is needed
only for cloud regeneration, not the local API/PDF checks above. Do not start
cloud jobs as part of installation.

## Frozen inputs and identities

All paths below are relative to this directory. SHA-256 values are recorded in
`reports/v11_submission_20261116_02/inputs.json`, rather than a handwritten digest
that can drift.

| Role | Input |
|---|---|
| Original train/validation/test membership | `data/processed/ax_t1_128/splits/{train,val,test}.csv` |
| Original pixel arrays | Paths in those CSVs; stored native orientation, normalised 128x128 |
| Development generator | `checkpoints/diffusion/ax_t1_128/latest.pt` (epoch 10, raw weights) |
| Continued generator | `checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt` (sample EMA U-Net **and** EMA conditioner) |
| Original predictor | `checkpoints/predictor/ax_t1_128/best.pt` |
| Primary judges | `reports/v4_heldout_judge/b2_local_20261016_01/training/orientation_robust_seed{0,1,2}/judge_heldout_resnet18_seed{0,1,2}.pt` |
| Judge checkpoint registry | `reports/v4_heldout_judge/b2_local_20261016_01/frozen_checkpoints.json` |
| Frozen experiment settings | `reports/v4_diffusion/b1_20261016_01/protocol.json`, `protocol_addendum_v2.json` and `remote/full/selected_operating_points.json` |

The original cohort has 72/15/16 patients and 4,434/987/934 slices; counts and
patient disjointness are freshly checked in `dataset_summary.json`. The separate
judge cohort is not an external institution. No source DCE tissue mask may be
substituted for axial-T1 ground truth on the basis of patient identity.

Evaluation uses `select_diverse_examples(test_df, 100, seed=42)` from
`scripts/run_explainability_batch.py`, up to ten slices per patient. The saved
`indices` arrays are authoritative for order; per-example base seed is 42+i.
Additional generation offsets are 10000 and 20000. Field strength uses t=100,w=5;
the epoch-110 primary manufacturer setting is t=300,w=3. Its t=300,w=5 result is
a fixed comparator, not the primary selection. Epoch-10 manufacturer uses w=5.
EchoTime stays at the source value. Do not feed the later canonical data view
into a native-frame checkpoint and call it the same experiment.

## Evidence to manuscript

| Evidence | Source or producer |
|---|---|
| Epoch-10 original predictor and patient CIs | `reports/v2_quickwins/bootstrap/bootstrap_ci.csv`; `scripts/v2_quickwins/bootstrap_ci.py` |
| Historical training-cohort robust judges | `reports/v2_quickwins/regrade/regrade_summary.csv`; `regrade_with_judge.py` |
| Held-out judge results | B2 `regrade_old/` and `regrade_b3/{per_slice,summary}.csv`; `scripts/v4_heldout/evaluate_heldout.py:regrade` |
| Epoch-110 original/legacy outcomes, soundness and FID | `reports/v4_diffusion/b1_20261016_01/remote/analysis/20261016T153744Z/`; `scripts/v4_convergence/analyze.py` |
| Paired slice and patient tests | `reports/v4_stats_heldout_20261016_01/`; `scripts/v4_review/heldout_b3_stats.py` |
| Protocol, EchoTime and foreground texture | `reports/v4_header_confounds_20261017_01/`; `scripts/v4_header_confounds/audit_headers.py` |
| Matched qualitative examples and probabilities | `reports/v10_qualitative_judges_20261116/`; `../research-paper-v10-judges/scripts/prepare_evidence.py` |
| Remaining historical table values | `../research-paper-v11-evaluation/numbers.md` and `numbers_v4.md` map them to report files |
| v11 corrections | `../research-paper-v11-evaluation/changes.json` and `scripts/prepare_revision.py` |

The v11 handoff verifies corrected headline rows and recomputes held-out hit
means from saved per-slice grading. It also checks all tables, algorithms and
figure files against v10. This is not a fresh derivation of every historical CI,
FID, downstream result or training run. Judge-seed ranges are not CIs.

Run read-only verification into **new** attempt folders:

```powershell
python scripts/v11_submission/prepare_handoff.py --output reports/v11_submission_new_attempt
python scripts/v11_submission/verify_release.py --output reports/v11_release_new_attempt
```

The release verifier also renders the manuscript into a new `qa/` directory;
for a second manuscript verification, use a new copied paper revision or archive
the first QA directory under a new name first. It never overwrites old reports.

## Rebuild the IEEE master

The delivered figures are byte-identical copies of v10, whose source/evidence
provenance is retained. From a **new copy** of `../research-paper-v11-evaluation/`
(leave historical PDFs intact):

```powershell
tectonic --keep-logs paper_v11_ieee.tex
```

Render and inspect every page after changes; inspect the log for overfull boxes,
missing glyphs and unresolved references. The v11 preparation script is exclusive
creation and intentionally refuses to overwrite its delivered source/assets.

## Regenerate research only as a separately scoped run

Training and judge regeneration are not necessary for the current handoff.
The exact historical code, protocol and input hashes are preserved in the B1/B2
run folders; current source files are development copies. Historical cloud run
recipes are in `scripts/v4_convergence/README.md` and `scripts/v4_heldout/README.md`.
Their dated pending-budget statements do not describe the completed runs.

For a new local header audit (read-only data access, new reports):

```powershell
python scripts/v4_header_confounds/audit_headers.py --run-id v11_headers_new_attempt
python scripts/v4_review/heldout_b3_stats.py --run-id v11_stats_new_attempt
```

For a newly authorised full B1/B3 run, the versioned runner is:

```powershell
python -m scripts.v4_convergence.prepare --run-id new_authorised_run
python -m scripts.v4_convergence.test_pipeline --run-id new_authorised_run
# Paid regeneration is a separate decision, not a handoff step:
# python -m modal run scripts/v4_convergence/modal_app.py --run-id new_authorised_run --stage full --budget-usd <approved-budget>
```

Never silently regenerate into an existing run or use the deprecated ablation
synthesiser. Dataset retrieval requires provider-authorised access; no data,
checkpoint or public redistribution permission is inferred here.
