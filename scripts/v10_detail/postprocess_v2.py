"""Local-only strict verification, frozen analysis and contextual alpha-.50 reference."""
import argparse
import json

import pandas as pd
import numpy as np

from scripts.v8_generator_dev.io import ROOT,csv_write,json_write,digest,project_path
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v10_detail.verify_saved_v3 import verify
from scripts.v10_detail.analyse import analyse,crossed_interval
from scripts.v10_detail.metrics import measure


def contextual_changes(joined,metric,lookup,control):
    seed_col='training_seed_new' if 'training_seed_new' in joined else 'training_seed'
    values=joined[['patient_id',seed_col,'generation_seed']].rename(columns={seed_col:'training_seed'}).copy()
    new_col=metric+'_new' if metric+'_new' in joined else metric
    old=(joined[metric+'_legacy'].to_numpy() if metric in ('target_hit','paired_hit_delta','other_axis_stable') else
         np.array([lookup[(r.dataset_index,r.generation_seed,control)][metric] for r in joined.itertuples()]))
    values['difference']=joined[new_col].to_numpy()-old
    return values


def postprocess(prepared,results,prefix):
    prepared,results=map(project_path,(prepared,results))
    verified=verify(prepared,results,prefix+'_verified')
    analysed=analyse(prepared,results,verified,prefix+'_analysis')
    p=json.loads((prepared/'protocol.json').read_text())
    receipt=json.loads((results/'complete.json').read_text())
    output=ROOT/'reports/v10_detail'/(prefix+'_report');output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.postprocess_v2. RESULTS.md: frozen gate outcome and named-judge '
        'patient/replica intervals; contextual_reference.csv: paired source/generation comparisons with the immutable historical '
        'alpha-.50 reference, holding its single trained adapter fixed; inputs.json: inputs/code; complete.json: scope. '
        'Separate *_verified/ and *_analysis/ reports hold raw verification and all profiles/strata. No cloud call or promotion.\n',encoding='utf-8')
    reference=ROOT/'reports/v8_generator_dev/source_strength_20261117_01_validation_additive_alpha050'
    old_requests=pd.read_csv(reference/'requests.csv')
    old_grades=read_grading_csv(reference/'per_image.csv')
    old_requests=old_requests[old_requests.axis.eq('manufacturer')]
    lookup={}
    evidence={}
    for r in old_requests.itertuples():
        with np.load(reference/r.array_file) as z:
            for control in ('null','target'):
                lookup[(r.dataset_index,r.generation_seed,control)]=measure(z['source'],z[control])
        path=reference/r.array_file;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    frames=[]
    for path in receipt['evaluations'].values():
        home=project_path(path);meta=json.loads((home/'protocol.json').read_text())
        if meta['mode']=='coarse_fine' and meta['axis']=='manufacturer' and meta['guide_control']=='correct':
            frames.append(read_grading_csv(home/'per_image.csv'))
    fine=pd.concat(frames,ignore_index=True)
    controls=[]
    for (judge,control),part in fine.groupby(['judge','variant']):
        if control=='identity':continue
        pairs=old_grades[(old_grades.axis=='manufacturer')&(old_grades.judge==judge)&(old_grades.variant==control)]
        keys=['patient_id','dataset_index','generation_seed','axis','direction','judge','variant']
        joined=part.merge(pairs,on=keys,suffixes=('_new','_legacy'),validate='many_to_one',how='left')
        if len(joined)!=135 or joined.target_hit_legacy.isna().any():raise ValueError('Legacy contextual source pairing failed')
        for metric in ('target_hit','paired_hit_delta','other_axis_stable','interior_mad','interior_hf_residual_mad'):
            values=contextual_changes(joined,metric,lookup,control)
            controls.append(dict(judge=judge,control=control,metric=metric,
                contrast='new_fine_minus_historical_alpha050',legacy_training_replicas=1,
                caveat='Contextual: old single trained adapter held fixed, not a matched training-seed experiment',
                **crossed_interval(values,'difference',p)))
    csv_write(output/'contextual_reference.csv',pd.DataFrame(controls))
    gates=json.loads((analysed/'advancement.json').read_text())
    rates=pd.read_csv(analysed/'crossed_intervals.csv')
    lines=['# Replicated finer-guide development results','',
        'All planned outputs were strictly checked before this interpretation. The cohort is reused development; '
        'clinical preservation, physical scanner realism and independent confirmation remain unknown.','',
        'Frozen research advancement gate: '+str(gates['research_advancement'])+'. '
        'This decision concerns fine versus matched coarse guidance, not superiority to every historical editor. '
        'No model/default is promoted.','',
        '| Gate / named judge | Paired estimate [simultaneous interval] | Pass |','|---|---|---|']
    for g in gates['gates']:
        lines.append(f'| {g["control"]} {g["metric"]} / {g["judge"]} | '
            f'{g["estimate"]:.4f} [{g["ci_low"]:.4f}, {g["ci_high"]:.4f}] | '
            f'{g["interval_passed"] and g["replica_direction_passed"]} |')
    lines += ['', '| Axis / mode / named judge | Target-hit [95% crossed patient/replica CI] |','|---|---|']
    chosen=rates[(rates.guide_control=='correct')&(rates.variant=='target')&(rates.stratum=='all')&(rates.metric=='target_hit')]
    for r in chosen.itertuples():
        lines.append(f'| {r.axis} / {r.mode} / {r.judge} | {100*r.estimate:.1f}% [{100*r.ci_low:.1f}%, {100*r.ci_high:.1f}%] |')
    lines += ['', 'Raw target-hit and target-minus-null editing are different measures. Full named-judge '
        'null/target/identity, guide-control, direction/support/warning and per-training-seed intervals stay in the '
        'analysis tables. Field-strength support constraints remain unchanged and its results are exploratory. '
        'contextual_reference.csv retains the old alpha-.50 research reference on matched sources/generation seeds, '
        'with its single trained adapter held fixed; it cannot establish between-training-seed reproducibility '
        'for that legacy adapter. All adverse results and missing diagnostics remain.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for path in (verified/'complete.json',analysed/'advancement.json',reference/'requests.csv',reference/'per_image.csv',
                 ROOT/'scripts/v10_detail/postprocess_v2.py',ROOT/'scripts/v10_detail/verify_saved_v3.py'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,research_advancement=gates['research_advancement'],
        default_promoted=False,clinical_preservation='unknown',fresh_test=False))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','prefix'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();postprocess(a.prepared,a.results,a.prefix)
