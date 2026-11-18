# Anatomy-preservation development, milestone 1

This is a research-only benchmark and inference experiment. It does not train
a new model, change the application, or establish an anatomy-preserving editor.
The frozen epoch-110 EMA DDPM remains the reference. The existing papers,
reports and checkpoints are preserved.

## Run

Use system Python from `counterfactual-mri-generation/`:

```powershell
python -m unittest scripts.v5_anatomy.test_pipeline -v
python -m scripts.v5_anatomy.run_pilot --output reports/v5_anatomy/<new_run>
python -m scripts.v5_anatomy.render_pilot --pilot reports/v5_anatomy/<new_run>
```

The runner refuses an existing output directory and a non-validation split.
It verifies patient separation, frozen model hashes and source array ranges;
rechecks all input hashes at completion; and stores its protocol before sampling.
Default: one seeded random slice per original validation patient, both paper
axes, one generation seed per patient. This is an exploratory pilot and cannot
quantify generation-seed variance or generalisation. `--patients 2` allows a
smaller smoke run, recorded as an explicit cohort-size override. No cloud API,
download or network service is invoked. The local GPU is sufficient for this
small inference experiment; larger studies need a separately scoped run.

## Comparators and hypothesis

- Identity, to expose the trivial preservation-without-edit solution.
- Same-noise source-conditioned null edit.
- Unchanged SDEdit reference at the existing primary operating points.
- Coarse-source anchoring at predeclared strengths 0.25 and 0.5.

Each target edit shares the source, target labels, timestep, guidance and
per-patient RNG seed. The null edit differs only in its source condition.
EchoTime is always fixed. Both targets are evaluated separately, not jointly.
The manufacturer setting is t=300,w=3; field strength is t=100,w=5.

For each reverse step, estimate x0 using the predicted noise. Correct only
its Gaussian-smoothed component towards the similarly smoothed source:

`delta_x0 = strength * (1-t/(t_edit+1)) * (LP(source)-LP(x0_estimate))`.

Add `beta_t*sqrt(alpha_bar_previous)/(1-alpha_bar_t) * delta_x0` to the
original reverse mean, then use the original ancestral noise draw. This is
a modification of the clean-image term in the DDPM posterior; it is not a
learned anatomy encoder or a mathematically guaranteed anatomical constraint.
It can suppress genuine contrast change. Low-pass intensity is itself
scanner-sensitive and must not be called scanner-invariant anatomy.
Zero strength delegates to the original sampler. Tests plus a real first-case
check per axis verify identical reference outputs.

## Outcomes and limitations

`metrics.py` measures SSIM/MAD and a foreground proxy at three fixed intensity
thresholds after mild smoothing. Report worst threshold Dice, symmetric HD95,
centroid shift, area change and boundary F1. These measurements are made on the
native source grid without registration; distances are in pixels. Empty masks
are invalid, not perfect. Source foreground fixes the HF diagnostic region.
Threshold masks are contrast-sensitive and neither tissue nor tumour labels.

`configs/anatomy_pilot_v1.json` contains predeclared **engineering** screen
tolerances, not clinical or reader-calibrated thresholds. Reports distinguish:

1. Judge-specific target hits/gains, including source correctness and other-axis stability.
2. Proxy-screen passes and target-hit-plus-proxy passes.
3. Human-preservation outcomes: missing until actual reviews are supplied.

Each frozen held-out orientation-robust judge is reported separately beside the
original predictor. These assessors are descriptive development tools here;
their outputs do not guide gradients or choose a winner. They cannot be called
a fresh final test after this experiment. One slice per patient allows a paired
patient bootstrap; small-sample and degenerate-interval limitations remain.

Target-cell patient count, exact EchoTime support and training-range overlap
are reported from training metadata only. They are support diagnostics, not
proof of causal identification. Unsupported combinations are permitted only
for this flagged offline experiment. Field strength is protocol-confounded;
orientation remains unchanged in this milestone. No output is deployment-approved.

## Independent review and annotations

`review/` provides shuffled source/edit pairs with algorithm/judge labels hidden,
a blank response CSV and an annotation template. Keep `private_mapping.csv`
away from readers. This prepares review; it is not a completed reader study.

Provide binary `.npy` source/output masks in exactly the original grid and/or
named `[x,y]` landmark maps, independently checked on each image. Do not use
unregistered masks from another sequence, resize annotations, or treat tumour
boxes as tissue ground truth. Lesion judgements require explicit assessability.

```powershell
python -m scripts.v5_anatomy.score_review --pilot reports/v5_anatomy/<run> --review-csv <completed.csv> --annotations <completed.json> --output reports/v5_anatomy/<new_review_run>
```

Reader responses and annotation metrics remain separate. Missing data are not
filled from proxies; partial coverage and per-reader outcomes are retained,
and consensus is never invented. Qualified blinded review, reliable pathology
labels and a genuinely untouched final patient cohort remain required before
making stronger preservation/reliability claims.

## Next milestones

Use the pilot to understand failure modes, then predeclare a larger validation
comparison. Add a source-structure conditioning branch only after establishing
an independent benchmark and a training scheme that avoids source-copy shortcuts.
Canonical DICOM orientation and protocol-support improvements require a new
preprocessing/data version, not changes to the historical arrays. A matched
learned baseline and later multi-slice/higher-resolution work remain separate tasks.
