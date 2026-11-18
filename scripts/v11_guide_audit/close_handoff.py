"""Record completed local diagnosis and distinguish prototype from efficacy readiness."""
import json
from scripts.v8_generator_dev.io import ROOT


def close():
    base=ROOT/'reports/v11_guide_audit'; report=base/'report_20261117_02'
    done=json.loads((report/'complete.json').read_text())
    checked=json.loads((base/'verification_20261117_01/complete.json').read_text())
    engineering=json.loads((base/'engineering_20261117_01/complete.json').read_text())
    probe=json.loads((base/'forward_probe_20261117_01/complete.json').read_text())
    assert done['complete'] and not done['next_training_execution_ready']
    assert checked['arrays_independently_checked']==2295 and engineering['actual_updates']==12 and probe['rows']==1080
    directory=ROOT.parent/'improvements'
    title='## Local guide-control diagnosis and prototype complete'
    status=f'''{title}

Start with `reports/v11_guide_audit/report_20261117_02/RESULTS.md` and
`NEXT_EXPERIMENT.md`; the root `reports/v11_guide_audit/README.md` maps all outputs.
The original replicated-study advancement failure below remains unchanged.

Exploratory analysis retains all 51 profiles / 2,295 cases / three named judges,
all training/generation seeds, direction/support/warning strata. The 27 paired
contrasts yield 8,505 interval rows, with descriptive crossed CIs and a declared
15-member editing-interaction family. Target/null decomposition and every
patient are retained; leave-one-patient-out point editing differences stay
negative for robust seed0/seed1. No favourable-case omission or gate retuning.

Finer guidance makes target versus original-label-null images closer in the
strict source interior: fine-minus-coarse MAD -.0066 [95% descriptive crossed
CI -.0090,-.0043]. Independent Torch convolution reproduces every new image
distance/HF diagnostic (maximum error {checked['maximum_new_metric_error']:.3g}).
This measures conditional image change, not clinical preservation.

A local forward-only probe evaluates the same noisy source under source,
target-manufacturer and null conditions, with field strength/EchoTime fixed.
All 15 patients, three training seeds, two guided modes, six controls, two
timesteps and one fixed noise seed are retained (1,080 observations). At t=300,
the fine-minus-coarse label-response/parent-RMS ratio difference is -.0757
[95% descriptive crossed CI -.0961,-.0544]. Response alignment is also lower
and the unconditional CFG-arm shift larger. These surrogate findings support
investigating guide/condition competition; they do not prove copying or the
cause of every judge loss. They are not actual reverse-trajectory states or
new classifier hit rates. All 180 absent-parent observations match exactly.

The guide-state dropout candidate is implemented in
`scripts/v11_guide_audit/guide_dropout.py`. Twelve disposable batch-five
updates across six fresh copies exercise full/coarse/second-block/zero/absent
states, with matching actual draws, changed finite adapters and unchanged
parent/conditioner bytes. Seven tests pass across the analysis/prototype suites.
No efficacy checkpoint is saved. Full-exposure strict-resume/config/bundle and
intended-runtime batch-32 checks remain pending. No trained improvement,
clinical preservation, independent cohort or default promotion is claimed.

All charts and six all-patient panels were visually inspected. All work here
is local; no new cloud job/download or paid allowance was used. The next
controlled-run design/resource forecast is a proposal, not execution-ready or
authorised; prior budgets stay closed. Next: prepare the distinct full-exposure
dropout trainer/config, strict resume tests and exact bounded execution package.

'''
    path=directory/'STATUS.md';text=path.read_text(encoding='utf-8');assert title not in text
    marker='## Replicated-guide study complete; advancement rule not met'
    assert marker in text;path.write_text(text.replace(marker,status+marker,1),encoding='utf-8')
    active='''## Latest local continuation: guide-control diagnosis complete

Evidence: `reports/v11_guide_audit/report_20261117_02/RESULTS.md` and
`NEXT_EXPERIMENT.md`; index `reports/v11_guide_audit/README.md`.

- All saved controls/strata analysed, independent new-distance verification
  passes, all primary point estimates reproduce the previous study. Every
  patient, named judge and replica stays; the original advancement failure is
  not reclassified by exploratory CIs.
- Local one-step response probe completes 1,080 observations in about 42 s,
  with exact parent equality for 180 absent states. Parent/source hashes and
  full design are checked. Only one noise seed and initial q(source,t) states
  are probed; raw epsilon vectors are not retained/replayed for all forwards.
- Guide-state dropout prototype: five states, independently seeded RNG,
  seven analysis/prototype tests and 12 actual disposable updates across six
  copies. Matching draws and frozen parent/conditioner pass. This is capped
  engineering, not a trained efficacy result or full-exposure resume proof.
- Code: `scripts/v11_guide_audit/`. No historical source/report/checkpoint is
  changed. Final overview is report _02 (signed wording clarified; _01 kept).
  All three charts and six patient panels are visually inspected.

Next local task: distinct full-exposure guide-dropout trainer/config/checkpoint
identity with strict gate-RNG/policy resume checks and a frozen bounded bundle.
Use cached old controls as declared baselines; keep EchoTime source, test cohort
untouched, warning/support rules and all named judges. The next experiment
proposal has ten simultaneous gates, a proposed new US$12/one-A10/6.5-hour cap,
and no paid retry. It is not execution-ready or authorised. Before a later paid
launch, recheck pricing, finish local readiness and obtain new-run approval.
No app is running and previous allowance remainders stay closed.

'''
    path=directory/'GENERATOR_DEV_ACTIVE.md';text=path.read_text(encoding='utf-8')
    marker='## Completed replicated study (17 November 2026)'
    assert marker in text; text=text.replace(marker,active+marker,1)
    old='''Next local task: diagnose paired correct/zero/absent/wrong/split-guide failures,
retaining all named judges and per-training-seed results, before designing a
separate prospective refinement. Mask-aware sampling was not tested in this
allocation. Do not retune the completed gates or infer another paid allowance.'''
    text=text.replace(old,'''The paired-control diagnosis and refinement proposal are now complete in v11;
use the latest local continuation above. Mask-aware sampling remains separate.
Do not retune completed gates or infer another paid allowance.''')
    path.write_text(text,encoding='utf-8')
    path=directory/'ROADMAP.md';text=path.read_text(encoding='utf-8')
    old='- [ ] Diagnose the completed paired guide-control failures locally and prepare'
    assert text.count(old)==1;text=text.replace(old,'- [x] Diagnose the completed paired guide-control failures locally and prepare')
    text+='''
  [`reports/v11_guide_audit/paired_diagnosis_20261117_01/`,
  `verification_20261117_01/`, `forward_probe_20261117_01/`,
  `probe_verification_20261117_01/`, `engineering_20261117_01/`,
  `visuals_20261117_01/`, `report_20261117_02/`]. Local diagnosis/prototype only;
  no trained efficacy improvement, default promotion or new paid app.
- [ ] Prepare the distinct full-exposure guide-dropout trainer/config and strict
  policy/RNG resume tests; freeze an exact bounded package before new paid work.
  Intended-runtime full-batch checks require new-run authorisation; prior caps
  and unused balances do not authorise another launch.
'''
    path.write_text(text,encoding='utf-8')
    progress='''
## 2026-11-17 - local guide-control diagnosis and dropout prototype complete

- Entrypoint: `reports/v11_guide_audit/report_20261117_02/RESULTS.md`,
  `NEXT_EXPERIMENT.md`; full index in its parent README. All work uses fresh
  script/report paths and saved development outputs. No cloud app or download.
- Prepared the exploratory plan after the primary result was known; retained
  all 51 profiles / 2,295 cases / 6,885 case-judge rows. Twenty-seven paired
  contrasts, 8,505 intervals, every stratum/replica and all patient contributions
  remain. Original primary point estimates match. Target/null decomposition
  shows different loss/gain patients across judges; leave-one-out does not make
  robust seed0/seed1 point differences positive. No gate retuning or exclusions.
- New target-null image-change/HF diagnostics independently checked for every
  array using Torch convolution instead of the SciPy producer; max error below
  1e-15. Four statistical/pairing/missingness/image checks pass.
- A bounded local forward-only probe completes 1,080 observations at two
  initial-source timesteps, all sources/training seeds/controls, one fixed noise
  seed. Finer guidance weakens/rotates label response and shifts the null CFG
  arm more. All 180 disabled-adapter predictions exactly equal the parent;
  source/parent byte and completeness audit passes. These are not reverse
  trajectories, clinical evidence or proof of source copying; raw vectors are
  not retained/replayed for every forward.
- Implemented independent source-guide state dropout with distinct zero-input
  and absent-adapter states. Three gate/RNG/analytical-response tests pass.
  Twelve disposable real updates, six copies, batch five, forced all-state
  coverage: finite changed adapters, matching draw/gate streams and frozen
  parent/conditioner pass. No efficacy checkpoint or full-runtime readiness.
- Reviewed all three diagnostic charts and six all-patient panels. Initial
  report _01 is preserved; report _02 clarifies signed-difference wording only.
- Prepared a one-factor six-copy training proposal, cached baselines and ten
  prospective simultaneous gates. Runtime/cost extrapolation is explicit and
  not a guarantee/approval. Full trainer/resume/bundle readiness remains next;
  future paid work needs new authorisation. Clinical/default/publication scope
  and the previous failed advancement decision remain unchanged.
'''
    with (directory/'GENERATOR_DEV_PROGRESS.md').open('a',encoding='utf-8') as file:file.write(progress)
    decision='''
## 2026-11-17 - test guide-state dropout after local control/response diagnosis

Use `reports/v11_guide_audit/report_20261117_02/RESULTS.md` as the local
diagnosis entrypoint. The finer model's smaller target-null image change and
weaker/rotated one-step label response support testing guide/condition
competition. They do not prove source copying or explain every judge outcome.
Keep all controls, patients, named judges, directions/support/warning strata,
descriptive uncertainty and the declared interaction correction. Do not use
these post-hoc intervals to change the original advancement failure.

Prepare one-factor guide-state dropout; no concurrent loss/backbone/mask/CFG
change. The local candidate routes full/coarse/second-block/zero/absent states
correctly and passes disposable matched engineering updates, but has no full
efficacy/resume/batch-32 evidence. Zero-input bias and disabled-adapter parent
equality are different controls. A coarse model's second block is still coarse.

The next proposal uses matched fresh coarse/fine copies and cached old
baselines, with ten simultaneous gates including old-fine editing/source-MAD
safeguards. Margins remain research choices, not clinical equivalence. Passing
would only establish a development candidate, not superiority to every old
editor, independent confirmation or default/clinical promotion.

No new paid app, data download, test use or publication edit. The new proposed
US$12/one-A10/6.5-hour limit is not authorisation and old budgets stay closed.
Finish distinct full-exposure protocol/checkpoint/RNG resume identity and exact
local package before requesting a new paid-run decision. Recheck pricing then.
'''
    with (directory/'DECISIONS.md').open('a',encoding='utf-8') as file:file.write(decision)


if __name__=='__main__':close()
