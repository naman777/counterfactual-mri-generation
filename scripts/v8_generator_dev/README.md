# Canonical-data generator development: first milestone

This implements the first step of `improvements/generator-development-plan-2026-11-16.md`:
a strict canonical loader, supported-cell/patient-aware sampler, matched additive
and adaptive-normalisation U-Nets, frozen support/profile preparation and capped
local optimisation/EMA/checkpoint/CFG verification. It does not replace the
native-frame editor, API, existing checkpoint or paper evidence.

## Data and architecture

`data.py` validates each canonical row against its original train/val membership,
labels, fixed EchoTime and source path, checks source hashes on loading, recomputes
the orientation from recorded DICOM IOP, and verifies exact inverse transforms.
No test index/pixels are needed. The training sampler spreads mass over supported
acquisition cells, then patients, then slices. Sparse cells remain represented at
their original patient-uniform mass. Validation is never rebalanced.

`model.py` retains the historical U-Net unchanged for R0. R1 subclasses its
topology but replaces every residual block's additive condition injection with
scale/shift modulation after the second GroupNorm and before SiLU. Both variants
start with identical common convolution/normalisation/embedding weights. R1's
shift projections start with R0's projection weights and its scale projections
start at zero. The operation and parameter count differ; this is not a bitwise
equivalent initial model or legacy checkpoint format.

The source-structure adapter is deliberately not included in this experiment:
its acquisition leakage must be re-audited in the canonical frame before efficacy
training. Existing geometry correction is not protocol harmonisation. Existing
DCE masks are not current axial-T1 annotations.

## Run from the code root using system Python

```powershell
python -m unittest scripts.v8_generator_dev.test_pipeline -v
python -m scripts.v8_generator_dev.prepare --run-id <new_preparation_id>
python -m scripts.v8_generator_dev.train_smoke --prepared reports/v8_generator_dev/<new_preparation_id> --run-id <new_smoke_id> --device cuda
python -m scripts.v8_generator_dev.verify_stage --prepared reports/v8_generator_dev/<new_preparation_id> --smoke reports/v8_generator_dev/<new_smoke_id> --run-id <new_verification_id>
```

All report and checkpoint folders must be new. Preparation snapshots code/config
and verifies every included train/val source. The smoke checks the frozen hashes
before and after running, uses identical sampled sources/timesteps/noise for both
variants, updates EMA only after applied optimizer steps and reloads the saved EMA
checkpoint for exact prediction/seeded inference checks. Raw and EMA states are
stored separately with an explicit coordinate frame and dedicated loader.

The smoke uses fresh weights, capped steps/batch/inference timesteps and one fixed
validation slice per patient. It measures software behaviour and resources; its
images, noise losses and throughput do not establish useful editing or select a
winner. There is no judge-guided training, cloud call, download or test evaluation.
The saved optimiser state is diagnostic; exact resumed training is not implemented
or claimed by this capped runner.

`verify_stage.py` records actual unit-test output, rechecks frozen input and
checkpoint hashes, compares saved source arrays to canonical validation images,
checks raw/EMA separation and fixed EchoTime, and verifies nonzero target/null
condition responses. Nonzero differences establish wiring only. Every report
folder includes a README mapping output files to its producer.

Full R0/R1 efficacy training requires its own versioned stopping, selection,
resource and preservation/editing-evaluation protocol. Subsequent attention,
capacity, source adapter and native-derived higher-resolution experiments remain
separate candidates, with independent preservation evidence and untouched final
patients still required. No gain, runtime or cloud cost is promised in advance.

Progress log: `improvements/GENERATOR_DEV_PROGRESS.md` at the capstone root.
