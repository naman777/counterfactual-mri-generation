"""Independent raw pivot, index-resampled primary CIs and decomposition checks."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v10_detail.analyse import crossed_interval
from scripts.v14_dropout_audit.common import KEYS,METRICS


def run(prepared,cases,run_id):
    prepared,cases=map(project_path,(prepared,cases));p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((cases/'inputs.json').read_text()))
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.verify_cases. checks.json: '
        'independent raw grading pivot and primary CIs calculated by direct index resampling rather '
        'than count-weight bootstrap; inputs.json: raw/derived identities; complete.json: no new '
        'judge regrading or advancement test. Missing cells remain failures.\n',encoding='utf-8')
    parts=[]
    for key,study in [('old_receipt','old_full_guide'),('new_receipt','guide_dropout')]:
        receipt=json.loads(project_path(p[key]).read_text())
        for profile,relative in receipt['evaluations'].items():
            frame=read_grading_csv(project_path(relative)/'per_image.csv');frame['study']=study;parts.append(frame)
    raw=pd.concat(parts,ignore_index=True)
    route=['study','mode','guide_control',*KEYS]
    raw_indexed=raw.set_index(route+['variant']).sort_index()
    observed=pd.read_csv(cases/'case_metrics.csv').set_index(route).sort_index()
    for name,metric,variant in [('target_hit','target_hit','target'),('null_hit','target_hit','null'),
        ('target_probability','target_probability','target'),('null_probability','target_probability','null'),
        ('target_interior_mad','interior_mad','target'),('null_interior_mad','interior_mad','null'),
        ('target_interior_hf_residual_mad','interior_hf_residual_mad','target'),
        ('null_interior_hf_residual_mad','interior_hf_residual_mad','null'),
        ('target_ssim','ssim','target'),('target_other_axis_stable','other_axis_stable','target')]:
        direct=raw_indexed.xs(variant,level='variant')[metric].sort_index()
        if not direct.index.equals(observed.index):raise ValueError('Original decomposition membership changed')
        np.testing.assert_allclose(observed[name],direct,atol=1e-12,rtol=0,equal_nan=True)
    np.testing.assert_allclose(observed.paired_hit_delta,observed.target_hit-observed.null_hit,atol=1e-14,rtol=0)
    np.testing.assert_allclose(observed.probability_shift,observed.target_probability-observed.null_probability,atol=1e-14,rtol=0)
    settings=dict(training_seeds=p['training_seeds'],analysis=dict(bootstrap_seed=p['bootstrap_seed'],
        bootstrap_resamples=p['bootstrap_resamples'],interval_quantiles=p['interactions_family']['quantiles']),
        evaluation=dict(selection_generation_seed=p['generation_seeds'][0],confirmation_generation_seeds=p['generation_seeds'][1:]))
    primary=pd.read_csv(cases/'primary_cases.csv');summaries=pd.read_csv(cases/'contrast_intervals.csv')
    checked=0;max_error=0.;family_count=0
    for (axis,contrast,judge),group in primary.groupby(['axis','contrast','judge']):
        for metric in METRICS:
            calculated=crossed_interval(group,metric,settings,False)
            saved=summaries[(summaries.axis==axis)&(summaries.contrast==contrast)&(summaries.judge==judge)&
                            (summaries.stratum=='all')&(summaries.metric==metric)]
            if len(saved)!=1:raise ValueError('Ambiguous primary diagnostic interval')
            saved=saved.iloc[0]
            for field in ('estimate','ci_low','ci_high'):
                error=abs(calculated[field]-saved[field]);max_error=max(max_error,error)
                if error>1e-12:raise ValueError('Independent crossed calculation differs')
            checked+=1
            if axis=='manufacturer' and contrast=='guide_effect_interaction__correct' and metric=='paired_hit_delta':
                adjusted=crossed_interval(group,metric,settings,True)
                np.testing.assert_allclose([saved.family_ci_low,saved.family_ci_high],
                    [adjusted['ci_low'],adjusted['ci_high']],atol=1e-12,rtol=0)
                family_count+=1
    json_write(out/'checks.json',dict(original_grade_rows=len(raw),decomposed_case_judge_rows=len(observed),
        independent_primary_intervals=checked,independent_primary_interaction_family_intervals=family_count,
        maximum_primary_interval_error=max_error,raw_pivot_matches_all_decomposition_metrics=True))
    paths=[cases/name for name in ('case_metrics.csv','primary_cases.csv','contrast_intervals.csv','inputs.json')]
    paths += [prepared/'protocol.json',project_path(__file__)]
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/'complete.json',dict(complete=True,independent_raw_pivot_verified=True,
        primary_intervals_independently_verified=checked,advancement_retested=False,paid_gpu_calls=0))
    print('INDEPENDENT SAVED-CASE ANALYSIS VERIFIED',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for name in ('prepared','cases','run-id'):a.add_argument('--'+name,required=True)
    run(**vars(a.parse_args()))
