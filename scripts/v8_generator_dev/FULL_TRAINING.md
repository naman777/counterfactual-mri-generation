# Full-training runner and development validation

The first engineering smoke is preserved. These new modules implement the next
milestone without changing its files, historical models or publication evidence.

## Frozen comparison

`configs/generator_full_v2.json` derives from the original development config.
`prepare_full.py` resolves/copies it into a new protocol, checks the original
source hashes, preserves the validation-source selection, and snapshots code,
judges and support. `full_preparation_20261116_02/` is the current local protocol;
the earlier `_01` full preparation is superseded by integration fixes and
explicit validation/worker caps. Original source preparation remains valid.

R0 and R1 use identical common initial weights, patient/cell sampling, draws,
timesteps and noise. Both stop at a common epoch when both meet the plateau
criterion, or at the maximum epoch. Primary comparisons use this common epoch;
earliest minimum patient-macro EMA noise loss selects secondary checkpoints.
Noise validation reports the actual slice-weighted and patient-macro means.
Saved raw weights, EMA, optimizer/scaler, dropout RNG and sample cursor support
exact recovery in the same runtime. Resumption rejects a changed protocol,
data identity, variant, batch size, torch version, device type or thread count.
The controller resumes from completed paired blocks; emergency resource-stop
checkpoints are retained for investigation, not silently treated as completed
paired runs. Time/storage caps can yield an explicitly incomplete experiment.

Verified train/val tensors are cached in host RAM for full training. The cache
does not move the dataset to GPU, rebalance validation, or change historical
arrays. The frozen source ledger is checked before and after execution.

## Development editing validation

`evaluate_full.py` retains source, same-noise null and target images for both
axes, fixed source EchoTime and all declared settings/seeds. Arrays stay in the
canonical frame. Before grading, each source's recorded inverse orientation is
applied to source/null/target so frozen native-frame judges see their expected
coordinate convention. All three judges remain descriptive development
instruments; they do not supply generator gradients or a fresh final test.

Outputs include per-case arrays, per-image probabilities/hits/paired deltas,
source correctness, non-edited-axis stability, correspondence/proxy measures,
direction/support strata and equal-patient bootstrap intervals. Repeated
generation seeds are averaged within patient before resampling. Selection uses
only its declared seed and supported/interpolation-range sources, requires
coverage in both directions, positive target-minus-null improvement and the
declared correspondence/stability screens. It can return no setting.

**Current support constraint:** the frozen one-slice-per-patient selection has
insufficient eligible field-strength 0-to-1 patients for the selection rule.
Field-strength selection must remain infeasible. Fixed-setting/outside-range
results can be reported as flagged diagnostics; this is not corrected by lower
loss or stronger classifier recognition. Manufacturer directions meet the
support-count rule but remain protocol-confounded and lack anatomy validation.
Numbers and exact counts are in the preparation's `selection_support.csv`.

SSIM and threshold-mask diagnostics do not establish anatomy or pathology
preservation. Those outcomes remain unknown without independent compatible
labels/review. Confirmation seeds check stability on the same patients, not
generalisation to an untouched cohort.

Boundary bootstrap intervals can be degenerate when every observed outcome is
the same. These small development samples do not turn such intervals into
certainty; engineering-run rates are not efficacy estimates.

## Local verification commands

Run from the code root with system Python. Every ID/folder must be new.

```powershell
python -m unittest scripts.v8_generator_dev.test_pipeline scripts.v8_generator_dev.test_full -v
python -m scripts.v8_generator_dev.prepare_full --run-id <new_protocol_id>
python -m scripts.v8_generator_dev.full_train --prepared reports/v8_generator_dev/<new_protocol_id> --run-id <first_id> --engineering-steps 2
python -m scripts.v8_generator_dev.full_train --prepared reports/v8_generator_dev/<new_protocol_id> --run-id <resume_id> --engineering-steps 2 --resume-pair reports/v8_generator_dev/<first_id>/complete.json
python -m scripts.v8_generator_dev.full_train --prepared reports/v8_generator_dev/<new_protocol_id> --run-id <direct_id> --engineering-steps 4
python -m scripts.v8_generator_dev.evaluate_full --prepared reports/v8_generator_dev/<new_protocol_id> --checkpoint checkpoints/diffusion/ax_t1_128_v8_generator_dev/<resume_id>/additive_block_0001.pt --run-id <evaluation_id> --engineering
```

Repeat the bounded evaluation for the scale-shift checkpoint. Then run
`verify_full_v2.py --prepared ... --first ... --resumed ... --direct ...
--evaluation ... --evaluation ... --run-id <new_verification_id>`.
The corrected verifier preserves literal `null` CSV controls and matches
CUDA/thread/TF32 settings to grading. It saves test output and compares interrupted/direct raw and
EMA weights, optimizer/scaler, RNG/cursors and draw/loss sequences. It also
recomputes all saved grading probabilities/hits/paired deltas and source frames.
An engineering segment cannot be promoted into full efficacy training.

The original verifier and failed check are preserved. Its default CSV reader
treated `null` as missing; `grading_io.py` and regression tests correct only the
reader. Raw arrays, probabilities and training states were not edited.

## Reviewable cloud execution

`bundle_full.py` creates a verified local ZIP of frozen code/metadata, existing
train/val arrays and judges; it does not upload anything. The archive manifest,
size/hash and calculation-based execution proposal are saved alongside it.
Raw DICOM and test indexes are excluded. `modal_full.py` is a prepared private
one-app/one-A10/one-container workflow with no retries and explicit time guards.
The current protocol has no approved monetary budget, so full execution is
blocked until a new resource-authorised protocol is frozen after user approval.

After explicit approval only, set `CAPSTONE_V8_BUNDLE` to the verified absolute
archive path and invoke the Modal entrypoint with a new ID and approved
`--budget-usd`. It derives a new authorised config/protocol on the volume,
validates the cloud runtime with tests and a capped engineering segment, then
trains both fresh variants and evaluates their common-epoch checkpoints.
No other Modal app may run in parallel. The source code/plan is reviewable;
cloud runtime and remote throughput are still unverified until that smoke.

The proposal uses [Modal's official pricing](https://modal.com/pricing), with
CPU/memory and a conservative margin/reserve. It is an estimate-based execution
guard, not a proved billing API cap or a promise that training finishes before
the time ceiling. No free credits are assumed. Inspect `complete.json` versus
`failed.json`/`resource_stop.json` before treating any cloud stage as complete.

Progress log: `improvements/GENERATOR_DEV_PROGRESS.md` at the capstone root.

## Numerically interrupted run and precision candidate (17 November 2026)

`full_20261116_04` failed on a nonfinite gradient after the common epoch-70
checkpoints. Do not use the resource-only recovery helper for this failure or
invent a full-training completion receipt. `failed_pair.py` verifies the actual
failure and selects only the latest fully matched block. The validation-only
Modal entrypoint keeps original settings/judges, hashes, incomplete status and
the remainder of the single approved allowance. The original uploaded code is
unchanged. `analyse_interrupted.py` explicitly retains interrupted status.

For a future separately prepared training run, the copied config
`configs/generator_full_v2_precision.json` enables eight bounded AMP gradient
retries. `precision_step.py` backs the scale off, restores dropout RNG and
replays the same batch without advancing the optimiser, EMA or sample cursor
until one finite-gradient update succeeds. Zero remains the old-config default.
Persistent nonfinite gradients and nonfinite loss still stop the run. This is
tested with induced CPU-AMP overflow, including the actual training engine; it
is not a reproduction of the historical cloud failure or proof that all
numerical problems are solved. No replacement training has been launched.

Run the new checks with:
```powershell
python -m unittest scripts.v8_generator_dev.test_failed_pair scripts.v8_generator_dev.test_precision_step -v
```

`audit_failed_pair.py` checks downloaded raw/EMA/optimiser/scaler finiteness and
saves the full test log. `fetch_failed_validation.py` verifies checkpoint hashes
and retrieves complete validation arrays/receipts into new folders. Every
analysis/fetch completion marker describes only that stage, not completed or
converged training, independent anatomical validity, or untouched-test efficacy.
