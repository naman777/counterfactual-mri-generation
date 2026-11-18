"""Close the completed bounded study in the editable project decision/progress files."""
import json
from scripts.v8_generator_dev.io import ROOT


def close():
    base = ROOT / 'reports/v10_detail'
    final = json.loads((base / 'final_review_20261117_03/complete.json').read_text())
    gate = json.loads((base / 'final_20261117_02_analysis/advancement.json').read_text())
    checked = json.loads((base / 'final_20261117_02_verified/complete.json').read_text())
    cost = json.loads((base / 'budget_20261117_01/budget.json').read_text())
    assert final['complete'] and not final['research_advancement'] and final['app_stopped']
    assert checked['cases'] == 2295 and checked['grades'] == 20655 and cost['app_stopped']
    assert len(gate['gates']) == 6 and not gate['research_advancement']
    root = ROOT.parent / 'improvements'
    refs = ('Evidence entrypoint: `reports/v10_detail/final_review_20261117_03/RESULTS.md`; '
            'figures: `final_visuals_20261117_02/`. Original intermediate reports remain preserved.')
    g = gate['gates']
    primary = '\n'.join(f'- `{row["judge"]}` editing change: {100*row["estimate"]:+.2f} percentage points '
                        f'[{100*row["ci_low"]:.2f}, {100*row["ci_high"]:.2f}].'
                        for row in g[3:])
    status = f'''## Replicated-guide study complete; advancement rule not met

{refs}

All nine adapters finish the same 20-epoch exposure (2,780 applied updates each;
25,020 actual rows independently checked). All 51 profiles / 2,295 cases / 20,655
named-judge/control grades are retrieved and independently verified. These are
repeats from 15 reused development patients, not independent new patients.
Source/guide/request/grade identities and actual-source fixed EchoTime also pass.
Maximum local target-probability error is {checked['max_probability_error']:.9g};
all absent-guide copies are equal. Full sampling/noise arrays were not replayed.

Finer guidance improves measured source correspondence against its matched
coarse control, but fails the frozen editing noninferiority requirements for
heldout_orientation_robust_seed0 and seed1. Four of six simultaneous gates pass;
all three original support/correspondence safeguards pass. Headline differences
below are fine minus coarse, with simultaneous crossed patient/training-replica
intervals:

- Target strict-interior MAD: {g[0]['estimate']:.4f} [{g[0]['ci_low']:.4f}, {g[0]['ci_high']:.4f}].
- Target strict-interior HF residual MAD: {g[1]['estimate']:.4f} [{g[1]['ci_low']:.4f}, {g[1]['ci_high']:.4f}].
{primary}

Failed noninferiority does not establish a statistically significant editing
decrease; the full intervals and per-replica results remain. Field strength is
exploratory: one forward / seven reverse eligible patients versus minimum two
per direction. Both source-warning patients (Breast_MRI_024 and Breast_MRI_255),
all adverse controls and unsupported/out-of-range cases are retained. Source
correspondence proxies do not establish clinical anatomy/lesion preservation.
No fresh independent cohort, model/default promotion or publication change.

The app is stopped and its CLI exited zero. Conservative estimate, including
the frozen rate margin and reserve: US${cost['conservative_estimate_usd']:.4f} of the separate US$12 cap;
actual provider invoice unknown. `budget_20261117_01/` records the provider
app-stop evidence. Previous budgets stay closed; no unused balance authorises
another paid run. Next: local paired-control failure diagnosis before a new
prospective refinement proposal. More paid work needs its own authorisation.

'''
    path = root / 'STATUS.md'; text = path.read_text(encoding='utf-8')
    start = text.index('## Replicated-guide study')
    stop = text.index('## Mask, shortcut and actual-parent local preparation complete', start)
    path.write_text(text[:start] + status + text[stop:], encoding='utf-8')
    active = f'''## Completed replicated study (17 November 2026)

{refs}

- Run `guide_replicates_20261117_01`, app `ap-x6oIjE8IEpnFott8WemeN1` is
  complete/stopped. CLI 11413 and monitor 14167 exited zero. No cloud task or
  automatic paid restart is active. The separate US$12 allocation is closed;
  conservative estimate US${cost['conservative_estimate_usd']:.4f}, invoice unknown. No remainder transfer.
- `results_20261117_01/`: full original retrieval. All nine full-exposure
  final checkpoints are local; `training_snapshot_20261117_01/` checks final
  finite states, `training_verification_20261117_01/` checks all 25,020 rows.
  The latter consumed the original terminal receipt in `results_20261117_01/root/`.
- `final_20261117_02_verified/`: all 2,295 saved cases / 20,655 grades pass;
  `metadata_verification_20261117_01/`: all request/source/guide/grade identities,
  actual-source fixed EchoTime and array labels pass. Raw sampling/noise tensors
  were not independently regenerated; clinical preservation stays unknown.
- Final verifier is local-only `verify_saved_v3`, called by `postprocess_v2`.
  It computes each image diagnostic once and compares with every judge row.
  All original assertions/settings remain; the interrupted redundant local
  v2 check and regression evidence are in `regrading_optimisation_20261117_01/`.
  No cloud sampling/training restarted. Frozen analysis code is unchanged.
- `final_20261117_02_analysis/`: four of six simultaneous gates pass; seed0
  and seed1 editing noninferiority fails. All original safeguards pass. Overall
  research advancement is false. Keep named judges, all controls/strata and
  uncertainty. No default, clinical or independent-test claim.
- Preferred figures `final_visuals_20261117_02/` and overview
  `final_review_20261117_03/`. Older overview _01/_02 used an integer warning
  series as row labels; the report-only formatter is corrected and tested in
  `final_quality_review_20261117_01/`. Actual warnings are 024/255. Earlier plots
  with overlapping axis labels are preserved. All corrected charts and all
  fifteen-patient panels for each training seed have been visually inspected.

Next local task: diagnose paired correct/zero/absent/wrong/split-guide failures,
retaining all named judges and per-training-seed results, before designing a
separate prospective refinement. Mask-aware sampling was not tested in this
allocation. Do not retune the completed gates or infer another paid allowance.
Historical sections below describe prior states, not pending execution commands.

'''
    path = root / 'GENERATOR_DEV_ACTIVE.md'; text = path.read_text(encoding='utf-8')
    start = text.index('## Active approved replicated study')
    stop = text.index('## Latest continuation: mask/shortcut/integration checks complete', start)
    path.write_text(text[:start] + active + text[stop:], encoding='utf-8')
    path = root / 'ROADMAP.md'; text = path.read_text(encoding='utf-8')
    old = '- [ ] Pass the pinned-A10 nine-copy full-batch gate; execute the approved matched'
    assert text.count(old) == 1
    text = text.replace(old, '- [x] Pass the pinned-A10 nine-copy full-batch gate; execute the approved matched')
    text += '''
  Completed: `reports/v10_detail/results_20261117_01/`,
  `final_20261117_02_{verified,analysis,report}/`,
  `metadata_verification_20261117_01/`, `training_verification_20261117_01/`,
  `final_review_20261117_03/`, `final_visuals_20261117_02/`, `budget_20261117_01/`.
  Execution/verification is complete; scientific advancement is false (two
  judge noninferiority requirements fail). No default/clinical promotion.
- [ ] Diagnose the completed paired guide-control failures locally and prepare
  a separate prospective refinement proposal before any new paid run.
'''
    path.write_text(text, encoding='utf-8')
    progress = f'''
## 2026-11-17 - replicated-guide study complete, negative advancement decision

- {refs}
- The single approved cloud job completes all nine 20-epoch adapters and all
  51 profiles; CLI exits zero, app stopped. Full retrieval preserves original
  files/hashes. All 25,020 applied journal rows and nine finite final states
  pass. All 2,295 saved cases / 20,655 named grades pass independent source,
  image-metric, probability/decision and absent-copy checks; maximum probability
  error {checked['max_probability_error']:.9g}. Supplementary full metadata audit passes.
- Local-only verifier v3 caches deterministic image diagnostics across judges
  while still checking every row/assertion. Original v2 process was stopped
  read-only; new prefix _02 passes in full. Regression evidence and partial old
  stdout are preserved; no cloud work restarted and no scientific rule changed.
- Four of six simultaneous gates pass. Source-interior MAD/HF correspondence
  improves; editing noninferiority is not established for robust seed0/seed1.
  All original support/correspondence safeguards pass. Keep the whole mixed
  result, field-strength support limitation, 024/255 warnings and all adverse
  controls. No anatomy/lesion validation, independent cohort or promotion.
- Figure QA corrected overlapping vertical labels in a new _02 plot folder.
  Comparing all-patient panels against the overview exposed an integer-warning
  indexing error in the report formatter only; tested Boolean masking produces
  the corrected overview _03. Initial outputs remain preserved. All corrected
  charts and all-patient panels were inspected; no numeric scientific result changed.
- Closed allocation: {cost['worker_seconds']:.2f} worker seconds; conservative provider-app-wall
  estimate with margin/reserve US${cost['conservative_estimate_usd']:.4f} / US$12, invoice unknown.
  No further paid action follows from unused credits. Status, roadmap and
  decision log updated; next task is local paired-control failure diagnosis.
'''
    path = root / 'GENERATOR_DEV_PROGRESS.md'
    with path.open('a', encoding='utf-8') as f: f.write(progress)
    decision = f'''
## 2026-11-17 - retain the replicated finer-guide failure to advance

Use `reports/v10_detail/final_review_20261117_03/RESULTS.md` as the final study
entrypoint. The matched three-training-seed experiment completes in full and
passes saved-output/judge/metadata/exposure checks. Four of six simultaneous
gates pass. Better source-interior correspondence does not waive failed editing
noninferiority for heldout_orientation_robust_seed0 and seed1; overall research
advancement is false. Failed noninferiority is not proof of a significant
editing decrease. Keep all intervals, replica directions, original safeguards,
historical-reference caveats and control/support/warning strata.

No default/model promotion, clinical preservation, independent confirmation,
new data admission or change to the existing paper results. Field strength
remains exploratory with unchanged support constraints. Next is local paired
control diagnosis, then a separately frozen refinement proposal; mask-aware
sampling was not tested here. Do not retune this completed study's gates.

The versioned local verifier v3 only removes repeated image-metric calculations
across judges and retains all assertions, including undefined diagnostics;
postprocess_v2 uses the unchanged frozen analysis. Full new outputs and code
hashes are retained. Report-only warning-index/plot-label fixes are documented
in `final_quality_review_20261117_01/`; the actual flagged patients are 024/255.

This one-A10 app is stopped, CLI zero. The separate new US$12 allocation is
closed at US${cost['conservative_estimate_usd']:.4f} conservative estimate (margin/reserve included; invoice
unknown). Old allowances and unused balances cannot authorise a restart or
another paid app. No additional paid work was launched.
'''
    path = root / 'DECISIONS.md'
    with path.open('a', encoding='utf-8') as f: f.write(decision)


if __name__ == '__main__':
    close()
