"""Paired crossed-replica uncertainty and the frozen development advancement gate."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.full_protocol import select_setting,bootstrap_patients
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT,csv_write,json_write,digest,project_path
from scripts.v10_detail.protocol import validate


def crossed_interval(frame,metric,protocol,adjusted=False):
    rules=protocol['analysis'];seeds=protocol['training_seeds']
    generations=[protocol['evaluation']['selection_generation_seed'],*protocol['evaluation']['confirmation_generation_seeds']]
    keys=['patient_id','training_seed','generation_seed']
    if frame.duplicated(keys).any() or set(frame.training_seed)!=set(seeds):raise ValueError('Missing/duplicate crossed design')
    patients=sorted(frame.patient_id.unique())
    expected=pd.MultiIndex.from_product([patients,seeds,generations],names=keys)
    series=frame.set_index(keys)[metric].reindex(expected)
    if len(series)!=len(frame) or series.isna().any() or not np.isfinite(series).all():
        raise ValueError('Missing crossed primary cell or metric')
    values=series.to_numpy().reshape(len(patients),len(seeds),len(generations)).mean(2)
    rng=np.random.default_rng(rules['bootstrap_seed']);n=rules['bootstrap_resamples']
    pp=rng.integers(len(patients),size=(n,len(patients)))
    ss=rng.integers(len(seeds),size=(n,len(seeds)))
    draws=values[pp[:,:,None],ss[:,None,:]].mean((1,2))
    quantiles=rules['interval_quantiles'] if adjusted else [.025,.975]
    lo,hi=np.quantile(draws,quantiles)
    return dict(estimate=float(values.mean()),ci_low=float(lo),ci_high=float(hi),
        patients=len(patients),training_replicas=len(seeds),generation_seeds=len(generations),
        family_adjusted=adjusted,quantile_low=quantiles[0],quantile_high=quantiles[1])


def gate_rows(joined,protocol):
    output=[]
    for rule in protocol['analysis']['gates']:
        judge=rule.get('judge','heldout_orientation_robust_seed0')
        rows=joined[(joined.judge==judge)&(joined.variant==rule['control'])].copy()
        metric=rule['metric'];rows['difference']=rows[metric+'_fine']-rows[metric+'_coarse']
        result=crossed_interval(rows,'difference',protocol,True)
        passed=(result['ci_high']<rule['upper_less_than'] if 'upper_less_than' in rule else
                result['ci_low']>rule['lower_greater_than'])
        replica_means=rows.groupby('training_seed').difference.mean()
        replica_pass=(bool((replica_means<0).all()) if rule['control']=='target' and metric.startswith('interior_') else
                      bool((replica_means>=-.05).all()) if metric=='paired_hit_delta' else True)
        output.append(dict(metric=metric,control=rule['control'],judge=judge,rule=rule,
            **result,interval_passed=bool(passed),replica_direction_passed=replica_pass,
            replica_point_estimates={str(k):float(v) for k,v in replica_means.items()}))
    return output


def analyse(prepared,results,verified,run_id):
    prepared,results,verified=map(project_path,(prepared,results,verified))
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    checked=json.loads((verified/'complete.json').read_text())
    if not checked['saved_outputs_verified']:raise ValueError('Independent saved-output verification required')
    receipt=json.loads((results/'complete.json').read_text())
    if not receipt['workflow_success']:raise ValueError('Complete replicated workflow required')
    parts=[]
    for name,path in receipt['evaluations'].items():
        home=project_path(path);f=read_grading_csv(home/'per_image.csv');f['profile']=name;parts.append(f)
    frame=pd.concat(parts,ignore_index=True)
    primary=frame[(frame.axis=='manufacturer')&(frame.guide_control=='correct')]
    fine=primary[primary['mode']=='coarse_fine'];coarse=primary[primary['mode']=='coarse_control']
    keys=['patient_id','dataset_index','training_seed','generation_seed','axis','direction','judge','variant',
          'condition_support_status','timestep','guidance','source_echo_time_norm','target_echo_time_norm']
    joined=fine.merge(coarse,on=keys,how='outer',suffixes=('_fine','_coarse'),validate='one_to_one',indicator=True)
    if not joined._merge.eq('both').all():raise ValueError('Unmatched primary evidence')
    gates=gate_rows(joined,p)
    original=[]
    for seed,part in fine.groupby('training_seed'):
        original.append(dict(training_seed=int(seed),check=select_setting(part,'manufacturer',{'evaluation':p['evaluation']})))
    safeguards=all(row['check']['selected'] is not None for row in original)
    passed=all(g['interval_passed'] and g['replica_direction_passed'] for g in gates) and safeguards
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    json_write(output/'protocol.json',p)
    json_write(output/'advancement.json',dict(gates=gates,original_safeguards=original,
        research_advancement=bool(passed),default_promoted=False,clinical_preservation='unknown',
        reason='All six simultaneous development gates, per-replica directions and original support/correspondence safeguards required'))
    summaries=[];replica_summaries=[]
    # Keep named judges, both axes and every guide control; uncertainty across all
    # replicas is only computed for complete crossed cells, never pooled as IID slices.
    for key,group in frame.groupby(['mode','axis','guide_control','judge','variant']):
        strata=[('all',group),*[(f'direction:{k}',v) for k,v in group.groupby('direction')],
                *[(f'support:{k}',v) for k,v in group.groupby('condition_support_status')],
                *[(f'warning:{k}',v) for k,v in group.groupby('qc_warning')]]
        for stratum,part in strata:
            for metric in ('target_hit','paired_hit_delta','other_axis_stable','ssim','interior_mad','interior_hf_residual_mad'):
                for seed,replica in part.groupby('training_seed'):
                    replica_summaries.append(dict(zip(['mode','axis','guide_control','judge','variant'],key),
                        stratum=stratum,metric=metric,training_seed=int(seed),
                        invalid_rows=int(replica[metric].isna().sum()),
                        **bootstrap_patients(replica,metric,p['analysis']['bootstrap_seed'],p['analysis']['bootstrap_resamples'])))
                if part[metric].isna().any():
                    summaries.append(dict(zip(['mode','axis','guide_control','judge','variant'],key),stratum=stratum,
                        metric=metric,estimate=np.nan,ci_low=np.nan,ci_high=np.nan,
                        unavailable='Missing regional diagnostic retained; no complete-case claim'))
                    continue
                summaries.append(dict(zip(['mode','axis','guide_control','judge','variant'],key),stratum=stratum,
                    metric=metric,**crossed_interval(part,metric,p)))
    csv_write(output/'crossed_intervals.csv',pd.DataFrame(summaries))
    csv_write(output/'per_training_seed_intervals.csv',pd.DataFrame(replica_summaries))
    csv_write(output/'primary_joined.csv',joined)
    json_write(output/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in
        (prepared/'protocol.json',results/'complete.json',verified/'complete.json',ROOT/'scripts/v10_detail/analyse.py')})
    (output/'README.md').write_text('Producer: scripts.v10_detail.analyse. protocol.json: frozen settings; advancement.json: '
        'six adjusted gates/replica directions/original safeguards; crossed_intervals.csv: every named-judge/control/stratum '
        'with crossed patient/replica CIs; per_training_seed_intervals.csv: each replica patient CI and missingness; '
        'primary_joined.csv: matched fine/coarse rows; inputs.json: inputs; '
        'complete.json: scope. This is reused development evidence, not a clinical or independent test.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,research_advancement=bool(passed),clinical_preservation='unknown',
        default_promoted=False,fresh_test=False))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','verified','run-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();analyse(a.prepared,a.results,a.verified,a.run_id)
