"""Frozen ten-gate comparison with cached old fine guidance and every named profile."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.full_protocol import select_setting
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,project_path,verify_inputs
from scripts.v10_detail.analyse import crossed_interval
from scripts.v11_guide_audit.analyse import CrossedBootstrap
from scripts.v13_guide_dropout.protocol import validate,evaluation_protocol

PAIR_KEYS=['patient_id','dataset_index','training_seed','generation_seed','axis','direction','judge','variant',
           'condition_support_status','timestep','guidance','source_echo_time_norm','target_echo_time_norm','qc_warning']
METRICS=['target_hit','paired_hit_delta','other_axis_stable','ssim','interior_mad','interior_hf_residual_mad']


def pair(left,right):
    joined=left.merge(right,on=PAIR_KEYS,how='outer',suffixes=('_fine','_reference'),
                      validate='one_to_one',indicator=True)
    if not joined._merge.eq('both').all():raise ValueError('Unmatched primary metadata/cells')
    return joined


def gates(new,old,p):
    settings=evaluation_protocol(p);settings['analysis']=p['analysis']
    new_primary=new[(new.axis=='manufacturer')&(new.guide_control=='correct')]
    old_fine=old[(old.axis=='manufacturer')&(old.guide_control=='correct')&(old['mode']=='coarse_fine')]
    fine=new_primary[new_primary['mode']=='coarse_fine'];coarse=new_primary[new_primary['mode']=='coarse_control']
    pairs={'new_fine_minus_new_coarse':pair(fine,coarse),'new_fine_minus_old_fine':pair(fine,old_fine)}
    output=[]
    for rule in p['analysis']['gates']:
        joined=pairs[rule['comparison']];judge=rule.get('judge','heldout_orientation_robust_seed0')
        rows=joined[(joined.judge==judge)&(joined.variant==rule['control'])].copy()
        metric=rule['metric'];rows['difference']=rows[metric+'_fine']-rows[metric+'_reference']
        try:
            interval=crossed_interval(rows,'difference',settings,True)
            point=rows.groupby('training_seed').difference.mean()
            passed=(interval['ci_high']<rule['upper_less_than'] if 'upper_less_than' in rule else
                    interval['ci_low']>rule['lower_greater_than'])
            replica=(bool((point>=-.05).all()) if metric=='paired_hit_delta' else
                     bool((point<0).all()) if rule['comparison']=='new_fine_minus_new_coarse' and
                     rule['control']=='target' and metric.startswith('interior_') else True)
            output.append(dict(rule=rule,judge=judge,available=True,**interval,interval_passed=bool(passed),
                replica_direction_passed=replica,replica_estimates={str(k):float(v) for k,v in point.items()}))
        except ValueError as error:
            output.append(dict(rule=rule,judge=judge,available=False,error=str(error),
                               interval_passed=False,replica_direction_passed=False))
    return output,new_primary


def load_profiles(receipt,study):
    parts=[];evidence={}
    for profile,relative in receipt['evaluations'].items():
        home=project_path(relative);f=read_grading_csv(home/'per_image.csv')
        f['profile']=profile;f['study']=study;parts.append(f)
        for name in ('per_image.csv','requests.csv','protocol.json','complete.json'):
            evidence[(home/name).relative_to(ROOT).as_posix()]=digest(home/name)
    return pd.concat(parts,ignore_index=True),evidence


def analyse(prepared,results,verified,run_id):
    prepared,results,verified=map(project_path,(prepared,results,verified))
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    receipt=json.loads((results/'complete.json').read_text())
    checked=json.loads((verified/'complete.json').read_text())
    if not receipt['workflow_success'] or not checked['saved_outputs_verified'] or checked['cases']!=1890:
        raise ValueError('All saved new profiles must be independently verified first')
    verify_inputs(json.loads((verified/'inputs.json').read_text()))
    old_receipt=ROOT/'reports/v10_detail/results_20261117_01/complete.json'
    old_checked=ROOT/'reports/v10_detail/final_20261117_02_verified'
    if not json.loads((old_checked/'complete.json').read_text())['saved_outputs_verified']:
        raise ValueError('Cached baseline is not verified')
    verify_inputs(json.loads((old_checked/'inputs.json').read_text()))
    new,evidence=load_profiles(receipt,'guide_dropout')
    old,old_evidence=load_profiles(json.loads(old_receipt.read_text()),'old_full_guide');evidence.update(old_evidence)
    gate_rows,primary=gates(new,old,p)
    safeguards=[dict(training_seed=int(seed),check=select_setting(part,'manufacturer',{'evaluation':p['reference_protocol']['evaluation']}))
                for seed,part in primary[primary['mode']=='coarse_fine'].groupby('training_seed')]
    passed=all(g['available'] and g['interval_passed'] and g['replica_direction_passed'] for g in gate_rows)
    passed=passed and len(safeguards)==3 and all(r['check']['selected'] is not None for r in safeguards)
    out=ROOT/'reports/v13_guide_dropout'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.analyse. advancement.json: '
        'all ten prospective simultaneous gates, every replica and original safeguards; '
        'crossed_intervals.csv and per_training_seed_intervals.csv: every new and cached-old '
        'named-judge/control/variant/direction/support/warning stratum, including missing measurements; '
        'all_grades.csv: all underlying new and cached grades; inputs.json: verified provenance; '
        'complete.json: research scope. No default promotion or independent/clinical claim.\n',encoding='utf-8')
    json_write(out/'advancement.json',dict(gates=gate_rows,original_safeguards=safeguards,
        research_advancement=bool(passed),default_promoted=False,clinical_preservation='unknown',
        primary_missingness='Any unavailable gate prevents advancement'))
    combined=pd.concat([new,old],ignore_index=True)
    generations=[p['reference_protocol']['evaluation']['selection_generation_seed'],
                 *p['reference_protocol']['evaluation']['confirmation_generation_seeds']]
    bootstrap=CrossedBootstrap(dict(bootstrap_resamples=10000,bootstrap_seed=583341,
        training_seeds=p['reference_protocol']['training_seeds'],generation_seeds=generations,
        interactions_family=dict(quantiles=[.0025,.9975])))
    summaries=[];replicas=[]
    dimensions=['study','mode','axis','guide_control','judge','variant']
    for keys,group in combined.groupby(dimensions):
        strata=[('all',group),*[(f'direction:{k}',v) for k,v in group.groupby('direction')],
                *[(f'support:{k}',v) for k,v in group.groupby('condition_support_status')],
                *[(f'warning:{k}',v) for k,v in group.groupby('qc_warning')]]
        for stratum,part in strata:
            summary,replica=bootstrap.intervals(part,METRICS)
            metadata=dict(zip(dimensions,keys),stratum=stratum)
            summaries.extend(dict(metadata,**r) for r in summary)
            replicas.extend(dict(metadata,**r) for r in replica)
    csv_write(out/'crossed_intervals.csv',pd.DataFrame(summaries))
    csv_write(out/'per_training_seed_intervals.csv',pd.DataFrame(replicas))
    csv_write(out/'all_grades.csv',combined)
    for path in (prepared/'protocol.json',results/'complete.json',verified/'complete.json',old_receipt,old_checked/'complete.json',
                 project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,research_advancement=bool(passed),
        new_cases=1890,new_named_grades=len(new),cached_named_grades=len(old),
        gates=10,default_promoted=False,fresh_test=False,clinical_preservation='unknown'))
    print('DROPOUT ANALYSIS COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','verified','run-id'):ap.add_argument('--'+name,required=True)
    analyse(**vars(ap.parse_args()))
