"""Diagnose saved-grade mismatch under default versus frozen judge runtime flags."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def audit(fetched,run_id):
    fetched=project_path(fetched)
    receipt=json.loads((fetched/'complete.json').read_text())
    key='additive__soft_source_guide__correct'
    home=project_path(receipt['evaluations'][key])
    config=json.loads((home/'protocol.json').read_text())['config']
    val=CanonicalDataset(config['view'],'val')
    requests=pd.read_csv(home/'requests.csv')
    saved=read_grading_csv(home/'per_image.csv').set_index(['case','judge','variant'],verify_integrity=True)
    path=project_path(config['evaluation']['judge_paths'][0])
    judge=load_judge(path,'cuda').eval().requires_grad_(False)
    name='heldout_orientation_robust_seed0'
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    inputs={path.relative_to(ROOT).as_posix():digest(path),
        (home/'per_image.csv').relative_to(ROOT).as_posix():digest(home/'per_image.csv'),
        'scripts/v8_generator_dev/audit_adapter_grading_runtime.py':digest(__file__)}
    rows,flags=[],{}
    for runtime in ('default','frozen'):
        if runtime=='frozen':
            deterministic_runtime()
        flags[runtime]=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,cpu_threads=torch.get_num_threads())
        for request in requests.itertuples():
            array_path=home/request.array_file
            with np.load(array_path) as arrays:
                stack=np.stack([arrays[key] for key in ('source','null','target')])
            native=source_native(stack,val.plans[int(request.dataset_index)])
            probabilities=predict_judge(judge,native,'cuda')
            axis,other=('mfg','fs') if request.axis=='manufacturer' else ('fs','mfg')
            target=request.target_manufacturer if axis=='mfg' else request.target_field_strength
            null_hit=int(probabilities[axis][1].argmax()==target)
            for j,control in enumerate(('identity','null','target')):
                wanted=saved.loc[(request.case,name,control)]
                hit=int(probabilities[axis][j].argmax()==target)
                delta=hit-null_hit
                error=abs(float(probabilities[axis][j,target])-wanted.target_probability)
                rows.append(dict(runtime=runtime,case=request.case,axis=request.axis,control=control,
                    expected_probability=wanted.target_probability,actual_probability=float(probabilities[axis][j,target]),
                    absolute_error=error,target_decision_matches=hit==wanted.target_hit,
                    paired_delta_matches=delta==wanted.paired_hit_delta,
                    source_correct_matches=int(probabilities[axis][0].argmax()!=target)==wanted.source_correct,
                    other_axis_matches=int(probabilities[other][j].argmax()==probabilities[other][0].argmax())==wanted.other_axis_stable))
            inputs[array_path.relative_to(ROOT).as_posix()]=digest(array_path)
    frame=pd.DataFrame(rows)
    summaries=[]
    for runtime,part in frame.groupby('runtime'):
        decisions=all(part[column].all() for column in ('target_decision_matches','paired_delta_matches','source_correct_matches','other_axis_matches'))
        summaries.append(dict(runtime=runtime,cases=int(part.case.nunique()),
            maximum_probability_error=float(part.absolute_error.max()),all_decisions_exact=bool(decisions),
            passes_declared_check=bool(decisions and part.absolute_error.max()<=1e-5)))
    csv_write(output/'recomputed.csv',frame)
    csv_write(output/'summary.csv',pd.DataFrame(summaries))
    json_write(output/'runtime_flags.json',flags)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,summaries=summaries,
        tolerance_unchanged=1e-5,scientific_model_changed=False,new_gpu_app_created=False))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.audit_adapter_grading_runtime. runtime_flags.json '
        'captures actual default/frozen flags; recomputed.csv retains every first-model/first-judge '
        'control under both runtimes. summary.csv/complete.json record exact decisions and errors '
        'against the unchanged 1e-5 tolerance. inputs.json hashes original arrays/judge/code. '
        'This diagnoses runtime parity only and makes no efficacy/anatomy claim.\n',encoding='utf-8')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    audit(a.fetched,a.run_id)
