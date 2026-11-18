"""Prospective inference-only source-guide strength sensitivity; frozen learned weights."""
import argparse
import gc
import json
import re

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v5_anatomy.metrics import diagnostics
from scripts.v6_structure.model import BoundStructureModel, StructureConditionedUNet
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.adapter_pilot import guide_for
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native, summarise
from scripts.v8_generator_dev.full_train import deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring


def evaluate(prepared,checkpoint,run_id,strength,device='cuda',engineering=False):
    if strength not in (0.,.25,.5,.75,1.) or not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new run and predeclared quarter-step strength required')
    guide_control='strength'
    prepared,checkpoint=project_path(prepared),project_path(checkpoint)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    protocol=json.loads((prepared/'protocol.json').read_text())
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format']!='soft_adapter_pilot_v1' or ck['settings']!=protocol['settings']:
        raise ValueError('Adapter checkpoint/protocol mismatch')
    if ck['engineering'] and not engineering:
        raise ValueError('Engineering adapter cannot become an efficacy model')
    if ck['mode']!='soft_source_guide':
        raise ValueError('Strength sensitivity requires a trained source-guide adapter')
    parent=project_path(protocol['parents'][ck['backbone']])
    if digest(parent)!=ck['parent_sha256']:
        raise ValueError('Frozen parent bytes changed')
    deterministic_runtime()
    c,base,scheduler,parent_payload=load_ema(parent,device)
    config=parent_payload['config']
    del parent_payload
    model=StructureConditionedUNet(base,ck['settings']['adapter_width']).to(device).eval()
    model.adapter.load_state_dict(ck['adapter_state_dict'])
    model.requires_grad_(False)
    c.requires_grad_(False).eval()
    train,val=[CanonicalDataset(protocol['view'],split) for split in ('train','val')]
    selected=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()
    if engineering:
        selected=selected[:2]
    if len(selected)<2 or val.frame.iloc[selected].patient_id.duplicated().any():
        raise ValueError('Distinct preselected source patients required for wrong-guide controls')
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'arrays').mkdir()
    inputs=dict(json.loads((prepared/'inputs.json').read_text()))
    inputs['scripts/v8_generator_dev/evaluate_strength.py']=digest(__file__)
    inputs.update({path.relative_to(ROOT).as_posix():digest(path) for path in
        (prepared/'protocol.json',prepared/'validation_selection.csv',checkpoint,parent)})
    json_write(output/'protocol.json',dict(config=config,adapter_protocol=protocol,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),
        guide_control=guide_control,guide_strength=strength,backbone=ck['backbone'],adapter_mode=ck['mode'],
        primary_epoch=ck['epoch'],engineering=engineering,
        scope='Existing development-validation patients; no anatomy ground truth or fresh test'))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.evaluate_strength. protocol.json freezes model, guide '
        'control, original fixed points and source selection. arrays/ retains source/null/target '
        'images and fixed EchoTime. requests.csv and per_image.csv retain every named-judge '
        'grade/support/direction; journal.jsonl preserves completed-case metadata on interruption. '
        'summary.csv contains patient-bootstrap CIs. inputs.json binds dependencies. '
        'Only source-guide residual amplitude changes; weights, data and fixed operating points stay frozen. '
        'This is a prospective development sensitivity test, not clinical or untouched-cohort validation.\n',encoding='utf-8')
    judges={f'heldout_orientation_robust_seed{seed}':load_judge(project_path(path),device).eval().requires_grad_(False)
        for seed,path in enumerate(config['evaluation']['judge_paths'])}
    records,requests=[],[]
    seeds=[config['evaluation']['selection_generation_seed'],*config['evaluation']['confirmation_generation_seeds']]
    if engineering:
        seeds=seeds[:1]
    with (output/'journal.jsonl').open('x',encoding='utf-8') as journal,torch.inference_mode():
        for axis in ('manufacturer','field_strength'):
            point=dict(timestep=2,guidance=3.) if engineering else config['evaluation']['fixed_settings'][axis]
            for generation_seed in seeds:
                for position,index in enumerate(selected):
                    item=val[index]
                    source=val.image_01(index)
                    image=item['image'].unsqueeze(0).to(device)
                    guide_index=index
                    guide_image=torch.from_numpy(val.image_01(guide_index).copy())[None,None].to(device)
                    guide=guide_for(guide_image,ck['mode'],0,augment=False)
                    present=torch.full((1,),float(strength),device=device)
                    bound=BoundStructureModel(model,guide,present)
                    m,f=item['manufacturer_label'],item['field_strength_label']
                    original=m if axis=='manufacturer' else f
                    tm,tf=(1-m,f) if axis=='manufacturer' else (m,1-f)
                    support=request_support(train.frame,tm,tf,float(item['echo_time_norm']),config['data']['minimum_cell_patients'])
                    arrays={'identity':source}
                    for control,mfg,fs in [('null',m,f),('target',tm,tf)]:
                        arrays[control]=sample_wiring(c,bound,scheduler,image,torch.tensor([mfg],device=device),
                            torch.tensor([fs],device=device),torch.tensor([item['echo_time_norm']],device=device),
                            int(generation_seed)+int(index),point['timestep'],point['guidance'])[0,0]
                    case=len(requests)
                    path=output/'arrays'/f'case_{case:05d}.npz'
                    with path.open('xb') as handle:
                        np.savez_compressed(handle,source=source,null=arrays['null'],target=arrays['target'],
                            source_echo_time_norm=item['echo_time_norm'],target_echo_time_norm=item['echo_time_norm'],
                            target_manufacturer=tm,target_field_strength=tf)
                    request=dict(case=case,patient_id=item['patient_id'],dataset_index=int(index),axis=axis,
                        direction=f'{original}_to_{1-original}',generation_seed=int(generation_seed),
                        actual_sampling_seed=int(generation_seed)+int(index),
                        evaluation_phase='engineering' if engineering else 'strength_sensitivity_same_development_patients',
                        timestep=point['timestep'],guidance=point['guidance'],
                        source_echo_time_norm=float(item['echo_time_norm']),target_echo_time_norm=float(item['echo_time_norm']),
                        target_manufacturer=tm,target_field_strength=tf,array_file=str(path.relative_to(output)),
                        guide_dataset_index=int(guide_index),guide_patient_id=str(val.frame.iloc[guide_index].patient_id),
                        backbone=ck['backbone'],adapter_mode=ck['mode'],guide_control=guide_control,
                        guide_strength=float(strength),**support)
                    requests.append(request)
                    native=source_native(np.stack([arrays[name] for name in ('identity','null','target')]),val.plans[index])
                    metrics={name:diagnostics(source,arrays[name]) for name in arrays}
                    key,other=('mfg','fs') if axis=='manufacturer' else ('fs','mfg')
                    case_records=[]
                    for name,judge in judges.items():
                        probabilities=predict_judge(judge,native,device)
                        null_hit=int(probabilities[key][1].argmax()==1-original)
                        for j,control in enumerate(('identity','null','target')):
                            hit=int(probabilities[key][j].argmax()==1-original)
                            case_records.append(dict(request,variant=control,judge=name,target_hit=hit,
                                target_probability=float(probabilities[key][j,1-original]),
                                source_correct=int(probabilities[key][0].argmax()==original),
                                other_axis_stable=int(probabilities[other][j].argmax()==probabilities[other][0].argmax()),
                                paired_hit_delta=hit-null_hit,judge_input_frame='source_native_inverse_geometry',
                                anatomy_assessment='unknown_no_independent_review',**metrics[control]))
                    records.extend(case_records)
                    journal.write(json.dumps(dict(request=request,grades=case_records),allow_nan=False)+'\n')
                    journal.flush()
                    if case%10==0:
                        print('ADAPTER VALIDATED CASE',ck['backbone'],ck['mode'],guide_control,case,flush=True)
    table=pd.DataFrame(records)
    csv_write(output/'requests.csv',pd.DataFrame(requests))
    csv_write(output/'per_image.csv',table)
    csv_write(output/'summary.csv',summarise(table,config))
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,engineering=engineering,cases=len(requests),
        patients=len(selected),judges=list(judges),guide_control=guide_control,guide_strength=float(strength),
        fixed_echo_time=True,test_pixels_used=False,judge_gradients_used=False,anatomy_validated=False))
    del c,model,base,scheduler,judges
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('ADAPTER VALIDATION COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--strength',required=True,type=float,choices=[0.,.25,.5,.75,1.])
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--engineering',action='store_true')
    a=p.parse_args()
    evaluate(a.prepared,a.checkpoint,a.run_id,a.strength,a.device,a.engineering)
