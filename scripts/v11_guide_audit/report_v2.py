"""Evidence-based local diagnosis and a bounded, unapproved refinement proposal."""
import argparse
import json
import numpy as np
import pandas as pd
from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path,verify_inputs
from scripts.v11_guide_audit.guide_dropout import STATE_NAMES,STATE_PROBABILITIES


def fmt(row,scale=1):return f'{scale*row.estimate:+.4f} [{scale*row.ci_low:.4f}, {scale*row.ci_high:.4f}]'


def write(run_id):
    root=ROOT/'reports/v11_guide_audit'
    analysis=root/'paired_diagnosis_20261117_01';probe=root/'forward_probe_20261117_01'
    checked=root/'verification_20261117_01';probe_checked=root/'probe_verification_20261117_01'
    engineering=root/'engineering_20261117_01';visuals=root/'visuals_20261117_01'
    for home in (analysis,probe,checked,probe_checked,engineering,visuals):
        assert json.loads((home/'complete.json').read_text())['complete']
        verify_inputs(json.loads((home/'inputs.json').read_text()))
    f=pd.read_csv(analysis/'contrast_intervals.csv');all_m=f[(f.axis=='manufacturer')&(f.stratum=='all')]
    primary=all_m[all_m.contrast.eq('fine_minus_coarse__correct')]
    response=pd.read_csv(probe/'paired_intervals.csv')
    cases=pd.read_csv(analysis/'primary_cases.csv');cases=cases[cases.axis=='manufacturer']
    influence=pd.read_csv(analysis/'leave_one_patient_out.csv');influence=influence[influence.axis=='manufacturer']
    output=root/run_id;output.mkdir(parents=True,exist_ok=False)
    patients=cases.groupby(['judge','patient_id','direction'])[['target_hit_difference','null_hit_difference','paired_hit_delta_difference']].mean().reset_index()
    csv_write(output/'patient_contributions.csv',patients)
    lines=['# Generator guide-control diagnosis','',
        '**Finding:** the completed data support a guide-dependent source-correspondence/editing trade-off, but do not prove source copying. '
        'Finer guidance reduces the target-versus-null image change and weakens/rotates the one-step manufacturer-label response. '
        'A guide-state dropout candidate is implemented and has passed capped local integration checks; it has not undergone efficacy training.','',
        'The previous advancement failure remains unchanged. This is exploratory analysis after the primary outcome was known, '
        'using the same 15 development source patients, three training replicas and three saved generation seeds. '
        'All 51 profiles / 2,295 cases / 6,885 case-by-judge observations are retained. No test pixels, new reverse edits, '
        'paid cloud app, publication change or default promotion. Clinical anatomy/lesion preservation remains unknown.','',
        '## Where the primary editing change comes from','',
        'Editing is target hit minus original-label-null hit. These are fine-minus-coarse paired differences with descriptive '
        '95% crossed patient/training-seed intervals; they do not replace the original simultaneous advancement intervals.','',
        '| Named robust judge | Target-hit change, pp | Null-hit change, pp | Net editing change, pp |','|---|---:|---:|---:|']
    for judge,group in primary.groupby('judge'):
        values=[group[group.metric==m].iloc[0] for m in ('target_hit','null_hit','paired_hit_delta')]
        lines.append('| '+judge+' | '+' | '.join(fmt(v,100) for v in values)+' |')
    distance=primary[(primary.judge=='heldout_orientation_robust_seed0')&primary.metric.eq('null_to_target_interior_mad')].iloc[0]
    lines += ['', f'Strict-source-interior target-versus-null absolute image change (fine minus coarse): {fmt(distance)} '
        'normalised intensity units. This is an image-change measure, not a clinical preservation score. '
        'The new distance/HF diagnostics were independently recomputed for every saved array using a separate Torch convolution; '
        'all primary point estimates reproduce the earlier study.', '',
        'Losses for judge seed0 are mainly target-hit losses with a smaller contribution from null hits. Seed1 has a different '
        'patient/direction pattern; seed2 has gains. patient_contributions.csv contains every patient for all three named judges. '
        'Neither warning source (024/255) is removed. Do not replace the full cohort by a favourable subset.','',
        '| Named robust judge | Range of leave-one-patient-out net point differences, pp (not a CI) |','|---|---:|']
    for judge,part in influence.groupby('judge'):
        lines.append(f'| {judge} | [{100*part.editing_difference.min():.4f}, {100*part.editing_difference.max():.4f}] |')
    lines += ['', 'The two negative judge point differences remain negative after leaving out any single patient. '
        'This sensitivity check does not authorise omission or change the frozen gates.','',
        '## What the guide controls can establish','',
        'A zero input still activates learned adapter biases. An absent adapter exactly restores the frozen parent. '
        'The coarse-control second block contains duplicated coarse information; its fine_only control is not a fine guide. '
        'Wrong-patient and split/zero guide states were not trained as such interventions, so degradation can reflect distribution shift. '
        'Correct-minus-control differences alone do not prove copying or scanner-invariant anatomy.','',
        'All 15 declared editing interactions are shown below. Interaction = (fine minus coarse with correct guide) minus '
        '(fine minus coarse with the named control). Intervals use the declared 15-member simultaneous family. '
        'The other many diagnostic intervals are descriptive 95% only.','',
        '| Control interaction | Named robust judge | Editing interaction, pp [family interval] |','|---|---|---:|']
    interactions=all_m[all_m.interaction & all_m.metric.eq('paired_hit_delta')]
    for r in interactions.itertuples():
        lines.append(f'| {r.contrast} | {r.judge} | {100*r.estimate:+.4f} [{100*r.family_ci_low:.4f}, {100*r.family_ci_high:.4f}] |')
    lines += ['', 'The correct-versus-wrong interaction is strongest for robust judge seed0; the pattern is not uniform across judges. '
        'Wrong guides also change anatomy-related image content, so higher judge editing under wrong guidance is not a desirable editor. '
        'Every direction/support/warning stratum and per-training-seed interval remains in the full analysis tables. '
        'Field strength still has insufficient directional support and is exploratory.','',
        '## Local one-step label-response probe','',
        'The same noisy source is evaluated under source manufacturer, target manufacturer and null condition; field strength '
        'and EchoTime stay at the source. No reverse trajectory is sampled. All 15 sources, three training seeds, both guided '
        'models and all six controls are included at t=100 and t=300, with one already fixed noise seed. '
        'The following are correct-guide fine-minus-coarse differences with descriptive 95% crossed intervals. '
        'These are epsilon-vector diagnostics, not classifier hit rates.','',
        '| Timestep | Diagnostic | Difference [95% CI] |','|---:|---|---:|']
    for r in response[response.guide_control.eq('correct')].itertuples():lines.append(f'| {r.timestep} | {r.metric} | {fmt(r)} |')
    lines += ['', 'At both inspected timesteps, finer guidance reduces the label-response RMS relative to the parent, '
        'reduces alignment with the parent label-response direction, and creates a larger shift in the unconditional CFG arm. '
        'This supports investigating conditional-signal competition. It does not establish the cause of every judge loss: '
        'only one noise seed was probed, q(source,t) differs from actual later reverse states, and no clinical assessment exists. '
        'The 180 absent-parent observations are exactly equal; the parent and all source hashes were checked. '
        'Raw epsilon vectors were not saved or independently replayed for every forward.','',
        '## Implemented candidate and next decision','',
        'The proposed single change is independently sampled guide-state dropout during adapter training. The prototype '
        'supports full, coarse-only, second-block-only, zero-input and absent-adapter states; it keeps source labels, '
        'the parent, conditioner and loss definition fixed. The 12 disposable actual updates exercise every state and '
        'retain matched source/noise/label-drop draws across modes. All six fresh copies update finite adapter weights '
        'while parent/conditioner bytes stay unchanged. Seven analysis/prototype tests pass across the two suites.','',
        '**These are engineering checks, not a trained improvement.** No efficacy checkpoint is saved. A full versioned '
        'trainer/config with strict new-policy resume identity, a frozen execution package and intended-runtime full-batch '
        'checks are still required before full training. NEXT_EXPERIMENT.md records the prospective design, resources and limits.','',
        '## Evidence','']
    for label,home in [('Analysis',analysis),('Independent image/statistical checks',checked),('Local response probe',probe),
                       ('Probe identity audit',probe_checked),('Gated engineering',engineering),('All-patient/all-control figures',visuals)]:
        lines.append(f'- [{label}]({(home/"README.md").as_posix()})')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    old_root=ROOT/'reports/v10_detail/results_20261117_01/root/complete.json'
    prior=json.loads(old_root.read_text());plan_path=ROOT/'reports/v10_detail/preparation_20261117_02/plan.json'
    old_plan=json.loads(plan_path.read_text())
    phases=prior['phases'];train=next(x['seconds'] for x in phases if x['phase']=='train')*6/9
    evaluations=sum(x['seconds'] for x in phases if x['phase'].startswith('eval_') and '__learned_zero__' not in x['phase'])
    projected=1.25*(train+evaluations+32.1+180)
    proposal=dict(format='guide_state_dropout_proposal_v1',execution_ready=False,paid_authorisation=False,
        policy=dict(zip(STATE_NAMES,STATE_PROBABILITIES)),training_copies=6,epochs=20,updates_per_copy=2780,
        total_updates=16680,new_profiles=42,new_cases=1890,new_named_grades=17010,
        old_baselines_reused=True,new_data=False,echo_time_edited=False,
        one_app=True,one_gpu='A10',max_containers=1,retries=0,proposed_new_cap_usd=12,
        proposed_worker_cap_seconds=23400,forecast_seconds=projected,forecast_training_seconds=train,
        forecast_evaluation_seconds=evaluations,forecast_multiplier=1.25,forecast_staging_seconds=180,
        conservative_rate=old_plan['price_with_margin'],reserve_usd=old_plan['reserve_usd'],
        forecast_usd=projected*old_plan['price_with_margin']+old_plan['reserve_usd'],
        forecast_within_proposed_time_cap=projected<=23400,
        prior_allowances_closed=True,automatic_paid_restart=False,
        statistical_family_size=10,interval_quantiles=[.05/20,1-.05/20],clinical_preservation='unknown')
    json_write(output/'proposal.json',proposal)
    proposal_text=f'''# Proposed next controlled refinement (not execution-ready or authorised)

Test one change: source-guide state dropout during fresh adapter training.
Use probabilities full .40, coarse-only .20, second-block-only .20, zero input
.10 and absent adapter .10. Gate RNG seed = training seed + 5,000,000 + step,
independent of source/noise/timestep/label-drop and photometric streams. Use the
same gates for matched coarse and fine modes. Do not train with a wrong patient
guide; retain it as a stress test. Zero input and absent adapter stay distinct.

Train six fresh adapters (two modes x the existing three matched training seeds)
from the same frozen epoch-130 parent, each for the common final 20 epochs.
Keep all other losses, conditioning, sampling, data and operating points fixed;
EchoTime remains source. Save raw final weights; no validation checkpoint choice.
Reuse the immutable old coarse/fine/learned-zero/parent comparisons as baselines.
No test-patient pixels or new download is needed.

Evaluate all six manufacturer guide controls and the fixed exploratory
field-strength point for each copy: 42 profiles / 1,890 new cases / 17,010 named
grades. Retain all 15 development patients, three generation seeds, every judge,
warning/support/direction stratum, null and identity. Clinical anatomy, physical
scanner realism and independent confirmation remain unknown.

Prospective advancement family: ten simultaneous gates, 10,000 crossed bootstrap
resamples, quantiles .0025/.9975. Retain the six original fine-versus-coarse
requirements under the new policy. Add three named-judge editing safeguards for
new fine versus cached old fine (lower interval > -.05, every replica point >=
-.05), and target-interior-MAD noninferiority to old fine (upper interval < .005).
The .005 and five-percentage-point tolerances are research choices, not clinical
equivalence/safety. Retain original source/support/other-axis safeguards. Passing
only establishes a development candidate. Do not claim better editing than old
fine unless the relevant named judge's family-adjusted interval is positive.
No default promotion follows; later untouched confirmation is separate.

Observed-duration forecast from the prior single app: six-copy training
{train/3600:.3f} h plus matching evaluation {evaluations/3600:.3f} h. With 180 s
staging, 32.1 s checks and 25% wall-time margin, forecast {projected/3600:.3f} h,
US${proposal['forecast_usd']:.3f} including the previously sourced marked-up rate
and reserve. This is a workload extrapolation, not an invoice or guarantee.
Proposed new cap: US$12, one A10/app/container, 6.5 worker hours, no paid retry.
Price inputs are the archived 17 November quote in preparation_20261117_02;
recheck official pricing before a future launch. All prior allowances are closed.

Before any launch, finish a distinct full-exposure trainer/config/checkpoint
format binding gate policy/RNG to strict resume identity; verify full-batch and
all-state routing, lost/missing/corrupt resume rejection, matched actual draws,
frozen bytes and engineering/efficacy separation. Freeze the exact new bundle,
hash ledger, statistical protocol, workload/time/cost guards and retrieval/report
commands. The current prototype has forced all-state batch-five coverage only;
it does not validate full-exposure resume or intended-runtime batch-32 training.
Obtain new-run authorisation before paid work; the previous remainder is not
approval. No Modal launch tool or automatic paid restart is part of this work.
'''
    (output/'NEXT_EXPERIMENT.md').write_text(proposal_text,encoding='utf-8')
    inputs={}
    for home in (analysis,probe,checked,probe_checked,engineering,visuals):
        for name in ('complete.json','inputs.json'):
            path=home/name;inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for path in (project_path(__file__),old_root,plan_path,analysis/'contrast_intervals.csv',analysis/'primary_cases.csv',
                 analysis/'leave_one_patient_out.csv',probe/'paired_intervals.csv',ROOT/'scripts/v11_guide_audit/guide_dropout.py'):
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',inputs)
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.report_v2. RESULTS.md: exploratory diagnosis and '
        'all named-judge decomposition/interactions/forward intervals; patient_contributions.csv: every source patient '
        'and judge, generation/training repetitions averaged, descriptive observations not independent rates; '
        'NEXT_EXPERIMENT.md: one-factor refinement design, pending readiness and separate authorisation; proposal.json: '
        'machine-readable scope/resource extrapolation; inputs.json: exact evidence and producer; complete.json: scope.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,local_diagnosis_complete=True,prototype_engineering_complete=True,
        next_training_execution_ready=False,paid_gpu_calls=0,mechanism_proven=False,clinical_preservation='unknown',default_promoted=False))
    print('DIAGNOSIS AND PROPOSAL COMPLETE',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--run-id',required=True);write(a.parse_args().run_id)
