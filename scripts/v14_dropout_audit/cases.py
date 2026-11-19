"""All saved-case target/null decomposition, dropout interactions and patient influence."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path,verify_inputs
from scripts.v13_guide_dropout.analyse import load_profiles
from scripts.v11_guide_audit.analyse import CrossedBootstrap
from scripts.v14_dropout_audit.common import decompose,paired,specifications,strata,METRICS,KEYS,META


def run(prepared,run_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.cases. case_metrics.csv: '
        'every original target/null decomposition, all studies/judges/controls; contrast_intervals.csv: '
        'every paired contrast/stratum with descriptive crossed CIs and the declared interaction family; '
        'per_training_seed.csv: every replica; primary_cases.csv/primary_patients.csv: all source effects; '
        'leave_one_patient_out.csv: full influence points, not CIs; gate_reproduction.json: unchanged primary '
        'point estimates; inputs.json: raw grading identities; complete.json: post-hoc local scope.\n',encoding='utf-8')
    parts=[];evidence={}
    for name,study in (('old_receipt','old_full_guide'),('new_receipt','guide_dropout')):
        receipt=json.loads(project_path(p[name]).read_text());frame,inputs=load_profiles(receipt,study)
        parts.append(frame);evidence.update(inputs)
    grades=pd.concat(parts,ignore_index=True)
    if len(grades)!=37665:raise ValueError('Unexpected original grade count')
    frame=decompose(grades);csv_write(out/'case_metrics.csv',frame)
    bootstrap=CrossedBootstrap(p);summaries=[];replicas=[];primary=[];influence=[];contrasts=0
    for axis,part in frame.groupby('axis'):
        for name,terms,family in specifications(axis):
            rows=paired(part,terms);contrasts+=1
            for judge,group in rows.groupby('judge'):
                for stratum,subset in strata(group):
                    result,per=bootstrap.intervals(subset,METRICS,family=family and stratum=='all')
                    meta=dict(axis=axis,contrast=name,judge=judge,stratum=stratum)
                    summaries.extend(dict(meta,**r) for r in result)
                    replicas.extend(dict(meta,**r) for r in per)
                if name in p['primary_case_contrasts']:
                    for patient in sorted(group.patient_id.unique()):
                        retained=group[group.patient_id!=patient]
                        influence.append(dict(axis=axis,contrast=name,judge=judge,omitted_patient=patient,
                            editing_difference=float(retained.paired_hit_delta.mean()),
                            all_patient_difference=float(group.paired_hit_delta.mean())))
            if name in p['primary_case_contrasts']:
                rows['contrast']=name
                if not np.allclose(rows.paired_hit_delta,rows.target_hit-rows.null_hit,atol=1e-14):
                    raise ValueError('Editing difference does not decompose')
                primary.append(rows)
    points=pd.concat(primary,ignore_index=True)
    csv_write(out/'contrast_intervals.csv',pd.DataFrame(summaries));csv_write(out/'per_training_seed.csv',pd.DataFrame(replicas))
    csv_write(out/'primary_cases.csv',points)
    patient_keys=['axis','contrast','judge','patient_id','dataset_index','training_seed',*META]
    patients=points.groupby(patient_keys,dropna=False)[METRICS].mean().reset_index()
    patients['editing_outcome']=np.select([patients.paired_hit_delta<0,patients.paired_hit_delta>0],['loss','gain'],default='unchanged')
    csv_write(out/'primary_patients.csv',patients);csv_write(out/'leave_one_patient_out.csv',pd.DataFrame(influence))
    frozen=json.loads(project_path(p['frozen_advancement']).read_text());checks=[]
    for g in frozen['gates']:
        if g['rule']['metric']!='paired_hit_delta':continue
        name=('guide_dropout__fine_minus_coarse__correct' if g['rule']['comparison']=='new_fine_minus_new_coarse' else
              'coarse_fine__dropout_minus_old__correct')
        found=points[(points.axis=='manufacturer')&(points.contrast==name)&(points.judge==g['judge'])]
        estimate=float(found.paired_hit_delta.mean())
        if not np.isclose(estimate,g['estimate'],atol=1e-14,rtol=0):raise ValueError('Frozen primary point estimate changed')
        checks.append(dict(comparison=g['rule']['comparison'],judge=g['judge'],estimate=estimate,matched=True))
    json_write(out/'gate_reproduction.json',checks)
    for path in (prepared/'protocol.json',prepared/'inputs.json',project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,post_hoc=True,original_cases=4185,case_judge_rows=len(frame),
        paired_contrasts=contrasts,interval_rows=len(summaries),frozen_primary_points_reproduced=True,
        all_patients_judges_controls_retained=True,paid_gpu_calls=0,new_reverse_sampling=False,
        advancement_retested=False,clinical_preservation='unknown'))
    print('SAVED-CASE DIAGNOSIS COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--prepared',required=True);a.add_argument('--run-id',required=True)
    run(**vars(a.parse_args()))
