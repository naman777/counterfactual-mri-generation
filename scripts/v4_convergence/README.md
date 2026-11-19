# B1 convergence and B3 evaluation

Run all commands from `counterfactual-mri-generation/` with system Python.
Existing experiments, original checkpoints and old report directories are inputs only.

## Prepared experiment

`reports/v4_diffusion/b1_20261016_01/` contains an immutable data manifest, copied
configuration, original model/judge inputs and preregistered `protocol.json`.
`configs/diffusion_v4.yaml` is the same initial config for inspection.

The generator architecture, linear schedule and complete ancestral editing chain
are unchanged. Training continues the epoch-10 raw model and AdamW optimiser,
with batch 32, lr 1e-4 and EMA 0.999. EMA inference weights and raw resume weights
are both saved; optimizer/scaler and CUDA/CPU torch RNG states support continuation.
The fixed validation-loss subset contains 256 slices and reuses the same noise
and timesteps. The append-only `protocol_addendum_v2.json` requires a plateau in
that loss as well as the classifier gate before an early stop. Reaching the
epoch limit is reported as a limit, rather than proof of convergence.
Epochs still mean complete passes over the same training split. Compare gradient
updates as well as epochs because the original batch size was four.

The synthetic gate starts from pure Gaussian noise, uses training-cell median
EchoTime, balances the four label combinations and grades both w=3/w=5 using all
three robust judges. It never reads a test image or test label. The gate criteria
are manufacturer >=90% and field strength >=80%, following task 01. Checkpoint
selection maximises the weaker normalised axis, then mean hit. The original
epoch-10 checkpoint is assessed under the same gate. At most 200 total epochs
are allowed, with a prespecified gate-plateau check after epoch 100; reaching
epoch 200 alone is not proof of convergence. A plateau in this noisy fixed gate
is also not proof of optimal generative modelling.

B3 evaluates a 4x3 timestep/guidance grid for each axis on 100 validation slices.
The selection constraints, tie-breaks and infeasible fallback are in the protocol.
Test edits are generated only after the validation choices are saved. An infeasible
operating point stays explicitly flagged. Fixed old operating points are retained
as comparators, so retraining and operating-point changes can be distinguished.

The histogram baseline uses training references, rather than historical test
references. The old baseline data remain unchanged; report the protocol change.
Sampling uses separate RNG streams per slice and the same full-precision DDPM
formula as the original editor. Local checks compare the implementations.

## Commands

```powershell
python -m scripts.v4_convergence.prepare --run-id <new_id>
python -m scripts.v4_convergence.test_pipeline --run-id <id>
python -m modal run scripts/v4_convergence/modal_app.py --run-id <id> --stage smoke
# Full paid run only after its compute budget has been authorised:
python -m modal run --detach scripts/v4_convergence/modal_app.py --run-id <id> --stage full --budget-usd 25
python -m scripts.v4_convergence.fetch --run-id <id>
python -m scripts.v4_convergence.analyze --run-id <id>
```

Use the file-path form for `modal run` on this Windows installation. Modal's
module-path discovery crosses namespace-package roots on different drives.
For downloads use the standalone `fetch.py` module, which avoids importing
Modal app decorators and copies completed training checkpoints into the new
checkpoint directory. It was added as a local download utility after launch;
the cloud scientific source snapshot is unchanged.
`--detach` retains the cloud controller if the client disconnects; it does not
schedule a future chat turn or notify the user from this chat.

The cloud controller chains training, gates, validation, test jobs and final
analysis inside one
app. The GPU pool is limited to eight containers; the current controller submits
one GPU task at a time to simplify reproducibility and cost accounting. Never
start a second Modal app while this one is active. Completed stage receipts can
be reused after interruption. A partially written job without its receipt must
be inspected and recovered under a new attempt directory, not silently replaced.

The compute ledger uses listed A10/CPU/memory rates with a 20% allowance and a
startup allowance. It reserves a whole worker timeout before starting another
task. This is a conservative scheduling guard, not an account-level billing
limit. Storage, egress and unrelated workspace usage are not metered by it.
Check the actual Modal billing dashboard for charges and available credits.

The smoke run is isolated from scientific training: it takes only twelve
training steps plus eight full-chain samples. Its checkpoint is marked `smoke`
and is never used to initialise the full run or choose a research checkpoint.

## Outputs

- `prepare.py`: float32 pack (including EchoTime and patient identities), hashes,
  copied input checkpoints, config, frozen protocol and report README.
- `train.py`: per-epoch training/validation JSON and immutable checkpoint per block.
- `evaluate.py`: synthetic gate outputs and probabilities; per-job image arrays
  and judge metrics for validation, main/fixed/stronger/joint edits, seeds,
  same-noise soundness controls and five baselines.
- `modal_app.py`: remote orchestration, conservative cost ledger, completed-stage
  receipts, checkpoint and operating-point selections, append-only downloads.
- `analyze.py`: B1 gate curves; B3 slice/patient bootstrap intervals, direction
  breakdown, paired tests, seed variability, same-noise SSIM gaps, HF energy,
  CQS/CADI, target-reference FID/MMD in predictor and robust-judge features.
- `test_pipeline.py`: numerical agreement, source-data integrity, selection and
  immutability checks. `local_preflight.json` records the first check result;
  rerun checks use a separately named receipt to preserve that first artifact.

No scientific completion claim is made by the smoke test or by preparing files.
`remote/full/complete.json` marks sampling complete;
`remote/full/research_complete.json` is written only after the automatic analysis,
including FID/MMD, completes. Its referenced analysis folder contains `status.json`.
