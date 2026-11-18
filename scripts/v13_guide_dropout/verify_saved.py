"""Strict local saved-array, metric and frozen named-judge regrading after retrieval."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import source_native,predict_judge
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v10_detail.metrics import measure,warning_for_index
from scripts.v13_guide_dropout.protocol import validate,copy_key,canonical_hash,evaluation_protocol
from scripts.v7_geometry.support import request_support


def metric_agrees(actual, expected, tolerance=3e-6):
    actual_missing=actual is None or bool(pd.isna(actual))
    expected_missing=expected is None or bool(pd.isna(expected))
    if actual_missing or expected_missing:
        return actual_missing and expected_missing
    return bool(np.isfinite(actual) and np.isfinite(expected) and np.isclose(actual,expected,rtol=0,atol=tolerance))


def verify(prepared,results,run_id,device='cuda'):
    prepared,results=map(project_path,(prepared,results))
    dropout=json.loads((prepared/'protocol.json').read_text());validate(dropout);p=evaluation_protocol(dropout)
    receipt=json.loads((results/'complete.json').read_text())
    if not receipt['workflow_success']:raise ValueError('Incomplete workflow retained; no efficacy reclassification')
    verify_inputs(json.loads((results/'inputs.json').read_text()))
    homes={k:project_path(v) for k,v in receipt['evaluations'].items()}
    expected={copy_key(seed,mode)+'__'+axis+'__'+control for seed in p['training_seeds'] for mode in p['modes']
              for axis in p['controls'] for control in p['controls'][axis][mode]}
    if set(homes)!=expected:raise ValueError('Incomplete declared evaluation matrix')
    deterministic_runtime();val=CanonicalDataset(p['view'],'val');train=CanonicalDataset(p['view'],'train')
    trained=json.loads((project_path(receipt['training'])/'complete.json').read_text())
    if trained['engineering'] or trained['common_epoch']!=20 or len(trained['copies'])!=6:
        raise ValueError('Full matched training required')
    for key,copy in trained['copies'].items():
        path=project_path(copy['checkpoint'])
        if digest(path)!=copy['sha256']:raise ValueError('Final trained checkpoint hash mismatch')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['engineering'] or ck['step']!=2780 or ck['protocol_sha256']!=canonical_hash(dropout) or copy_key(ck['training_seed'],ck['mode'])!=key:
            raise ValueError('Final checkpoint exposure/identity mismatch')
        del ck
    selected=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()
    generations=[p['evaluation']['selection_generation_seed'],*p['evaluation']['confirmation_generation_seeds']]
    expected_cases={(int(i),int(s)) for i in selected for s in generations}
    thresholds=json.loads((prepared/'mask_thresholds.json').read_text())['thresholds']['noise_aware_v1']
    judges={f'heldout_orientation_robust_seed{j}':load_judge(project_path(path),device).eval().requires_grad_(False)
            for j,path in enumerate(p['evaluation']['judge_paths'])}
    evidence=dict(json.loads((results/'inputs.json').read_text()));count=0;grades_count=0;max_error=0.
    evidence[project_path(__file__).relative_to(ROOT).as_posix()]=digest(__file__)
    absent={}
    for profile,home in homes.items():
        meta=json.loads((home/'protocol.json').read_text());requests=pd.read_csv(home/'requests.csv')
        table=read_grading_csv(home/'per_image.csv')
        if meta['scientific_protocol']!=p or meta['training_protocol']!=dropout or meta['engineering']:raise ValueError('Efficacy/protocol identity mismatch')
        if profile!=copy_key(meta['training_seed'],meta['mode'])+'__'+meta['axis']+'__'+meta['guide_control']:
            raise ValueError('Evaluation route identity mismatch')
        if meta['checkpoint_sha256']!=trained['copies'][copy_key(meta['training_seed'],meta['mode'])]['sha256']:
            raise ValueError('Evaluation used a different checkpoint')
        if set(zip(requests.dataset_index,requests.generation_seed))!=expected_cases or len(requests)!=45:
            raise ValueError('Missing or duplicate sources/generation seeds')
        if not set(table.judge)==set(judges) or len(table)!=45*9:raise ValueError('Incomplete named-judge grades')
        for r in requests.itertuples():
            source=val.image_01(r.dataset_index);item=val[r.dataset_index]
            point=p['evaluation']['fixed_settings'][r.axis]
            if r.actual_sampling_seed!=r.generation_seed+r.dataset_index or r.timestep!=point['timestep'] or r.guidance!=point['guidance']:
                raise ValueError('Operating point/noise mismatch')
            if r.source_echo_time_norm!=r.target_echo_time_norm:raise ValueError('Edited EchoTime')
            path=home/r.array_file
            if digest(path)!=r.array_sha256:raise ValueError('Saved output changed')
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['source'],source)
                if float(z['source_echo_time_norm'])!=float(z['target_echo_time_norm']):raise ValueError('Array EchoTime changed')
                images={name:z[key].copy() for name,key in [('identity','source'),('null','null'),('target','target')]}
            qc=warning_for_index(source,r.dataset_index,thresholds)
            if qc['qc_warning']!=r.qc_warning:raise ValueError('Mask warning drift')
            if r.guide_control=='absent':
                key=(r.axis,r.dataset_index,r.generation_seed)
                if key in absent:
                    for name in images:np.testing.assert_array_equal(images[name],absent[key][name])
                absent[key]=images
            native=source_native(np.stack([images[k] for k in ('identity','null','target')]),val.plans[r.dataset_index])
            # Image diagnostics do not depend on the judge. Compute each once,
            # then compare with every reported judge/control row below.
            measured={control:measure(source,image) for control,image in images.items()}
            key,other=('mfg','fs') if r.axis=='manufacturer' else ('fs','mfg')
            original=item['manufacturer_label'] if r.axis=='manufacturer' else item['field_strength_label']
            tm,tf=((1-item['manufacturer_label'],item['field_strength_label']) if r.axis=='manufacturer' else
                   (item['manufacturer_label'],1-item['field_strength_label']))
            support=request_support(train.frame,tm,tf,float(item['echo_time_norm']),5)
            if r.target_manufacturer!=tm or r.target_field_strength!=tf or r.condition_support_status!=support['condition_support_status']:
                raise ValueError('Target label or condition support mismatch')
            for name,judge in judges.items():
                prob=predict_judge(judge,native,device)
                for j,control in enumerate(('identity','null','target')):
                    saved=table[(table.case==r.case)&(table.judge==name)&(table.variant==control)]
                    if len(saved)!=1:raise ValueError('Duplicate grading cell')
                    s=saved.iloc[0];hit=int(prob[key][j].argmax()==1-original)
                    err=abs(float(prob[key][j,1-original])-s.target_probability);max_error=max(max_error,err)
                    if err>2e-6 or hit!=s.target_hit or hit-int(prob[key][1].argmax()==1-original)!=s.paired_hit_delta:
                        raise ValueError('Named-judge probability/decision drift')
                    if int(prob[other][j].argmax()==prob[other][0].argmax())!=s.other_axis_stable:
                        raise ValueError('Other-axis grade drift')
                    metrics=measured[control]
                    for metric,value in metrics.items():
                        expected_value=s[metric]
                        if not metric_agrees(value,expected_value):raise ValueError('Image metric or missingness drift: '+metric)
                    grades_count+=1
            count+=1
        for filename in ('protocol.json','requests.csv','per_image.csv','complete.json'):
            path=home/filename;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        print('STRICT REGRADING',profile,flush=True)
    output=ROOT/'reports/v13_guide_dropout'/run_id;output.mkdir(parents=True,exist_ok=False)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,saved_outputs_verified=True,cases=count,
        grades=grades_count,max_probability_error=max_error,absent_copies_equal=True,
        anatomy_validated=False,test_pixels_used=False))
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.verify_saved. inputs.json binds every evaluated array/report; '
        'complete.json records strict source/noise/point/EchoTime/mask-warning/metric/named-judge checks and exact absent-copy equality. '
        'No output is regenerated and no clinical or independent-cohort claim follows.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','run-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();verify(a.prepared,a.results,a.run_id)
