# MR-one: evaluation of counterfactual breast MRI edits

Current scientific scope: an evaluation of judge dependence, training sensitivity,
orientation shortcuts and failures of anatomy preservation. This is a local
research prototype; reliable anatomy-preserving manufacturer conversion and
clinical utility have not been established.

Start with the [current IEEE master](../research-paper-v11-evaluation/paper_v11_ieee.pdf)
and its [revision notes](../research-paper-v11-evaluation/README.md).
The [reproducibility guide](REPRODUCIBILITY.md) identifies the actual data splits,
frozen models, result sources, commands and environment limits. The
[supervisor review pack](../research-paper-v11-evaluation/SUPERVISOR_REVIEW.md)
lists the remaining author decisions. Nothing has been submitted or released.

## Run the local demo

From this directory, with the research inputs and dependencies already present:

```powershell
.\start_demo.bat
```

This defaults to saved replay. Open http://127.0.0.1:3000; use the paper link
for the v11 evaluation master. `start_demo.bat -Live` enables uncached generation
and requires the frozen diffusion checkpoints and a suitable runtime.
The frontend lockfile is in `web-app/package-lock.json`; use `npm ci` in
`web-app/` when provisioning a new installation. A fresh installation and a
timed team rehearsal still require verification.

## What is frozen

- Original cohort patient split and native stored pixel frame; the generator
  trains on 72 patients, rather than all 103 original patients.
- Epoch-10 raw checkpoint and epoch-110 EMA checkpoint.
- Three primary orientation-robust ResNet-18 judges trained on additional
  patients from the same collection. These are not external-site validation.
- Field strength and manufacturer are the only edited axes; source EchoTime
  remains fixed. Target hits are judge-specific outcomes, not anatomy approval.
- Existing figures and aggregate tables, including manufacturer failures.

The `scripts/v5_anatomy/`, `v6_structure/` and `v7_geometry/` branches are
separate development work. Their engineering checks and native DCE labels do
not validate anatomy preservation for the axial-T1 editor. Further model
development is deferred for this submission scope.

## Where to work

Run Python commands from this directory. Preserve existing `reports/` and
checkpoints; use a new output directory for every attempt. The current roadmap
starts with the November 2026 submission section in `../improvements/ROADMAP.md`.
Older dated entries are historical records, not instructions to repeat completed
experiments. The code folder already has Git metadata; the parent workspace
does not. No release tag or clean repository state is asserted.
