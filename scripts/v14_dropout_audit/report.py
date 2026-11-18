"""Report local diagnosis without retuning the completed primary requirements."""
import argparse
import json

import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path,verify_inputs
from scripts.v14_dropout_audit.common import warnings


def cell(row,scale=100):
    return f"{scale*row['estimate']:+.3f} [{scale*row['ci_low']:+.3f}, {scale*row['ci_high']:+.3f}]"


def run(prepared,cases,exposure,probe,case_verified,probe_verified,algebra,visuals,run_id):
    prepared,cases,exposure,probe,case_verified,probe_verified,algebra,visuals=map(project_path,
        (prepared,cases,exposure,probe,case_verified,probe_verified,algebra,visuals))
    for home in (case_verified,probe_verified,exposure,algebra,visuals):
        if not json.loads((home/'complete.json').read_text())['complete']:raise ValueError('Incomplete diagnostic prerequisite')
        verify_inputs(json.loads((home/'inputs.json').read_text()))
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.report. RESULTS.md: '
        'patient/control/exposure/epsilon diagnosis with explicit exploratory limits; '
        'patient_contributions.csv: every patient and named judge, signed target/null outcome counts '
        'over the nine fixed runs; inputs.json: verified evidence; complete.json: report scope. '
        'The original failed advancement rule and all adverse cases remain.\n',encoding='utf-8')
    p=json.loads((prepared/'protocol.json').read_text());intervals=pd.read_csv(cases/'contrast_intervals.csv')
    judges=[f'heldout_orientation_robust_seed{j}' for j in range(3)]
    def effect(contrast,judge,metric='paired_hit_delta',stratum='all'):
        part=intervals[(intervals.axis=='manufacturer')&(intervals.contrast==contrast)&(intervals.judge==judge)&
                       (intervals.stratum==stratum)&(intervals.metric==metric)]
        if len(part)!=1:raise ValueError('Ambiguous diagnostic result')
        return part.iloc[0]
    lines=['# Local diagnosis after the failed guide-dropout advancement test','',
        'The completed candidate still fails three primary interval safeguards. This diagnosis '
        'does not retest or relax those requirements. It finds matching implemented exposure, '
        'heterogeneous patient effects and no restoration of the measured fixed-input label-response '
        'strength. Simple guide-state dropout is not established as an editing improvement.','',
        '## Evidence and uncertainty','',
        'All 93 old/new profiles are retained: 4,185 saved cases and 37,665 named-judge/control '
        'grades. The target/null decomposition contains 12,555 case/judge rows. All 69 paired '
        'contrasts retain direction, support, warning and replica strata. An independent raw pivot '
        'and direct index-resampled calculation reproduce all 360 primary diagnostic intervals. '
        'The original ten simultaneous gates remain unchanged and failed.','',
        'Added intervals below are **post-hoc descriptive 95% crossed patient/training-seed CIs**; '
        'generation seeds are averaged inside each cell. They are not the original simultaneous '
        'primary intervals. A separately declared 18-member interaction family is retained in the '
        'full CSV. No member of the correct-guide interaction has a strictly positive lower family bound.','',
        '## Did the fine model improve, or did the coarse comparison change?','',
        '| Named judge | New fine − old fine | New coarse − old coarse | Change in fine − coarse gap |',
        '|---|---|---|---|']
    for judge in judges:
        lines.append('| '+judge+' | '+cell(effect('coarse_fine__dropout_minus_old__correct',judge))+' | '+
            cell(effect('coarse_control__dropout_minus_old__correct',judge))+' | '+cell(effect('guide_effect_interaction__correct',judge))+' |')
    lines += ['', 'Values are percentage-point changes in target-minus-null editing. For robust '
        'seed0 and seed1, the apparent narrowing of the fine/coarse gap partly comes from worse '
        'coarse-model point estimates. It does not establish that fine-guidance editing improved. '
        'Judge seed1 has exactly unchanged observed fine-model editing outcomes.','',
        '## Separate target hits from null-label drift','',
        '| Named judge | Target-hit change | Null-hit change | Target-minus-null change |',
        '|---|---|---|---|']
    for judge in judges:
        name='coarse_fine__dropout_minus_old__correct'
        lines.append('| '+judge+' | '+cell(effect(name,judge,'target_hit'))+' | '+
            cell(effect(name,judge,'null_hit'))+' | '+cell(effect(name,judge))+' |')
    lines += ['', 'Each row compares new fine with old fine. An improvement from fewer null hits '
        'is distinct from more successful target edits. For seed0, the nominal gains on '
        'Breast_MRI_261 and Breast_MRI_445 lie outside the target EchoTime range; the latter '
        'comes from fewer null hits, without a target-hit gain. Losses remain on '
        'Breast_MRI_812 and Breast_MRI_866. All are retained, not selected for advancement.','',
        '| Support stratum, new fine − old fine | Robust seed0 | Robust seed1 | Robust seed2 |',
        '|---|---|---|---|']
    for label,stratum in [('Interpolation-supported','support:interpolation_only'),
                           ('Outside target EchoTime range','support:outside_target_echo_range')]:
        lines.append('| '+label+' | '+' | '.join(cell(effect('coarse_fine__dropout_minus_old__correct',j,stratum=stratum)) for j in judges)+' |')
    lines += ['', 'These are percentage-point changes with descriptive CIs. EchoTime itself is '
        'never edited. Unsupported-request gains do not establish a supported physical scanner '
        'effect. The source warnings Breast_MRI_024 and Breast_MRI_255 stay in every analysis.','',
        '## Actual training exposure','']
    states=pd.read_csv(exposure/'state_exposure.csv');unique=states[states['mode']=='coarse_control']
    full=unique[unique.state=='full'];joint=pd.read_csv(exposure/'joint_exposure.csv')
    old_conditional=[]
    for seed in p['training_seeds']:
        part=joint[(joint.training_seed==seed)&(joint['mode']=='coarse_control')]
        old_conditional.append(part.loc[~part.condition_dropped,'samples'].sum()/88680)
    lines += [f"All 16,680 actual update rows and 532,080 sample exposures are counted. Full-guide exposure is "
        f"{100*full.observed_fraction.min():.3f}–{100*full.observed_fraction.max():.3f}% across unique training seeds. "
        f"Full-guide plus conditional-label exposure is {100*full.conditional_fraction_of_all_samples.min():.3f}–"
        f"{100*full.conditional_fraction_of_all_samples.max():.3f}%, compared with "
        f"{100*min(old_conditional):.3f}–{100*max(old_conditional):.3f}% conditional/full-guide exposure in the old policy.",
        '', 'Actual gate counts match independently audited checkpoints, and paired coarse/fine '
        'streams match exactly. Source indices, patients, noise hashes, timesteps and label-drop '
        'records also match the historical run. No state-routing or draw-mismatch bug is exposed '
        'by this audit. These counts do not measure each state’s causal gradient effect. '
        'An absent adapter contributes no adapter gradient; zero input can still train its biases. '
        'Matched modes are repeated use of the same gate draws, not independent samples.','',
        '## Fixed-input label-response probe','',
        'The laptop evaluates 2,160 source/target/null epsilon observations on the same noisy '
        'source inputs, both old/new models, all replicas/guide controls and t=100/300. One '
        'generation seed is used. Every raw epsilon tensor is saved. Independent float64 Torch '
        'calculations reproduce every metric; all 30 initial noise inputs are exactly replayed. '
        'All 360 absent-adapter observations match the parent exactly.','',
        '| t=300, correct guide comparison | Response/parent-RMS ratio difference | Alignment difference |',
        '|---|---|---|']
    probe_intervals=pd.read_csv(probe/'paired_intervals.csv')
    for name in ('new_fine_minus_old_fine','new_coarse_minus_old_coarse','new_fine_minus_new_coarse','guide_effect_interaction'):
        part=probe_intervals[(probe_intervals.timestep==300)&(probe_intervals.guide_control=='correct')&
                             (probe_intervals.contrast==name)].set_index('metric')
        lines.append('| '+name+' | '+cell(part.loc['response_ratio_to_parent'],1)+' | '+cell(part.loc['response_cosine_to_parent'],1)+' |')
    lines += ['', 'Dropout reduces the fixed-input response magnitude in both arms; the fine-minus-coarse '
        'magnitude gap remains. Relative alignment changes are also affected by worse coarse-model '
        'alignment. These are surrogate signals, not new classifier success rates, actual reverse '
        'states or proof of the cause of every failure. The independent check regrades all stored '
        'epsilon vectors and inputs; it does not replay every network forward.','',
        '## Concrete next local step','',
        'A shared-guide-residual epsilon combiner is implemented and algebra-tested against '
        'all 2,160 saved observations. It preserves the frozen parent’s source/target conditional '
        'difference at a fixed noisy input while adding a common guide residual. Absent-guide '
        'and zero-guidance endpoints are checked. It is not yet a full sampler or an efficacy '
        'result. The parent’s scanner response can itself be spurious; this construction does '
        'not guarantee good reverse trajectories or clinical preservation.','',
        'Next implement the distinct sampler, verify its short-trajectory endpoints and unchanged '
        'weights/random streams, then benchmark and freeze a bounded laptop pilot before judging '
        'editing performance. Existing negative outcomes, all guide controls and independent judges '
        'must remain. No paid cloud run, retraining, new test-cohort use, EchoTime edit or default '
        'promotion occurred here.','',
        f"Detailed proposal: `{(algebra/'NEXT_LOCAL_PROTOTYPE.md').relative_to(ROOT).as_posix()}`.",
        f"Figures: `{visuals.relative_to(ROOT).as_posix()}`. All full contrast, patient and exposure CSVs remain linked by inputs.json."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    points=pd.read_csv(cases/'primary_cases.csv');part=points[(points.axis=='manufacturer')&
        (points.contrast=='coarse_fine__dropout_minus_old__correct')]
    patients=part.groupby(['judge','patient_id','direction','condition_support_status','qc_warning'],as_index=False).agg(
        fixed_runs=('generation_seed','size'),net_target_hits=('target_hit','sum'),net_null_hits=('null_hit','sum'),
        net_editing_outcomes=('paired_hit_delta','sum'),mean_probability_shift_difference=('probability_shift','mean'))
    patients['qc_warning']=warnings(patients.qc_warning)
    csv_write(out/'patient_contributions.csv',patients)
    paths=[prepared/'protocol.json',cases/'complete.json',cases/'contrast_intervals.csv',cases/'primary_cases.csv',
        exposure/'complete.json',exposure/'checks.json',exposure/'state_exposure.csv',exposure/'joint_exposure.csv',
        probe/'complete.json',probe/'paired_intervals.csv',case_verified/'complete.json',probe_verified/'complete.json',
        algebra/'complete.json',algebra/'NEXT_LOCAL_PROTOTYPE.md',visuals/'complete.json',project_path(__file__)]
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/'complete.json',dict(complete=True,post_hoc_diagnosis=True,original_advancement_changed=False,
        paid_gpu_calls=0,new_reverse_sampling=False,new_training=False,full_sampler_implemented=False,
        clinical_preservation='unknown',default_promoted=False,visual_review_still_required=True))
    print('LOCAL DROPOUT DIAGNOSIS REPORTED',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for name in ('prepared','cases','exposure','probe','case-verified','probe-verified','algebra','visuals','run-id'):
        a.add_argument('--'+name,required=True)
    run(**vars(a.parse_args()))
