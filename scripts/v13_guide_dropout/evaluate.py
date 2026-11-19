# Derived from frozen v10 evaluator; new checkpoint identity, two-arm routes and output namespace.
"""Full fixed-point, frozen-judge evaluation with explicit source/null/target controls."""
import argparse
import gc
import json
import math

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v6_structure.model import BoundStructureModel
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet,multiscale_guide
from scripts.v8_generator_dev.evaluate_full import predict_judge,source_native,summarise
from scripts.v8_generator_dev.full_train import load_ema,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,csv_write,json_write,digest,project_path,verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v13_guide_dropout.protocol import validate,canonical_hash,evaluation_protocol,CHECKPOINT_FORMAT
from scripts.v10_detail.metrics import measure,warning_for_index


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


@torch.inference_mode()
def evaluate(prepared,checkpoint,run_id,axis,guide_control='correct',device='cuda',engineering=False):
    prepared,checkpoint=project_path(prepared),project_path(checkpoint)
    dropout=json.loads((prepared/'protocol.json').read_text());validate(dropout);p=evaluation_protocol(dropout)
    complete_ledger=json.loads((prepared/'inputs.json').read_text())
    # Evaluation consumes manifests, code, the parent/judges and selected source
    # slices. Training pixels and unused checkpoints are not read by this route.
    required_checkpoints={p['parent'],*p['evaluation']['judge_paths']}
    consumed={name:value for name,value in complete_ledger.items()
              if not name.endswith('.npy') and (not name.endswith('.pt') or name in required_checkpoints)}
    verify_inputs(consumed)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format']!=CHECKPOINT_FORMAT or ck['protocol_sha256']!=canonical_hash(dropout) or ck['protocol']!=dropout:
        raise ValueError('Checkpoint/protocol identity mismatch')
    if ck['engineering']!=engineering or (not engineering and (ck['step']!=2780 or ck['step']!=ck['total_steps'] or sum(ck['gate_counts'])!=4434*20)):
        raise ValueError('Incomplete or engineering checkpoint cannot enter efficacy evaluation')
    if axis not in p['controls'] or guide_control not in p['controls'][axis][ck['mode']]:
        raise ValueError('Route outside the prospectively declared matrix')
    parent=project_path(p['parent'])
    if digest(parent)!=p['parent_sha256']:raise ValueError('Parent changed')
    deterministic_runtime()
    c,base,scheduler,payload=load_ema(parent,device);del payload
    model=DetailConditionedUNet(base,p['training']['adapter_width']).to(device).eval()
    model.adapter.load_state_dict(ck['adapter']);model.requires_grad_(False);c.requires_grad_(False)
    train,val=CanonicalDataset(p['view'],'train'),CanonicalDataset(p['view'],'val')
    selected=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()
    if len(selected)!=15 or val.frame.iloc[selected].patient_id.duplicated().any():raise ValueError('Frozen patient selection mismatch')
    if engineering:selected=selected[:2]
    if not run_id.replace('_','').isalnum():raise ValueError('Safe new ID required')
    output=ROOT/'reports/v13_guide_dropout'/run_id;output.mkdir(parents=True,exist_ok=False);(output/'arrays').mkdir()
    json_write(output/'protocol.json',dict(scientific_protocol=p,training_protocol=dropout,checkpoint_sha256=digest(checkpoint),axis=axis,
        guide_control=guide_control,training_seed=ck['training_seed'],mode=ck['mode'],engineering=engineering))
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.evaluate. protocol.json: exact scope; arrays/: every source/null/target '
        'and fixed EchoTime; requests.csv: support/direction/source/guide/QC; per_image.csv: all named-judge decisions and regional metrics; '
        'journal.jsonl: completed case evidence on interruption; summary.csv: patient CIs; inputs.json: consumed hashes; complete.json: scope.\n',encoding='utf-8')
    judges={f'heldout_orientation_robust_seed{j}':load_judge(project_path(path),device).eval().requires_grad_(False)
            for j,path in enumerate(p['evaluation']['judge_paths'])}
    thresholds=json.loads((prepared/'mask_thresholds.json').read_text())['thresholds']['noise_aware_v1']
    seeds=[p['evaluation']['selection_generation_seed'],*p['evaluation']['confirmation_generation_seeds']]
    if engineering:seeds=seeds[:1]
    point=dict(timestep=2,guidance=3.) if engineering else p['evaluation']['fixed_settings'][axis]
    requests,grades=[],[]
    evidence=dict(consumed)
    for path in (prepared/'protocol.json',prepared/'validation_selection.csv',prepared/'mask_thresholds.json',checkpoint):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    with (output/'journal.jsonl').open('x',encoding='utf-8') as journal:
        for seed in seeds:
            for position,index in enumerate(selected):
                item=val[index];source=val.image_01(index)
                guide_index=selected[(position+1)%len(selected)] if guide_control=='wrong' else index
                for consumed_index in (index,guide_index):
                    source_row=val.frame.iloc[consumed_index]
                    source_name=project_path(source_row.source_image_path).relative_to(ROOT).as_posix()
                    evidence[source_name]=source_row.source_array_sha256
                if guide_control=='wrong' and val.frame.iloc[guide_index].patient_id==item['patient_id']:
                    raise ValueError('Wrong guide is from source patient')
                guide=multiscale_guide(torch.from_numpy(val.image_01(guide_index).copy())[None,None].to(device),ck['mode'])
                if guide_control=='zero':guide=torch.zeros_like(guide)
                if guide_control=='coarse_only':guide[:,3:]=0
                if guide_control=='fine_only':guide[:,:3]=0
                present=torch.full((1,),0. if guide_control=='absent' else 1.,device=device)
                bound=BoundStructureModel(model,guide,present)
                m,f=item['manufacturer_label'],item['field_strength_label']
                tm,tf=(1-m,f) if axis=='manufacturer' else (m,1-f)
                original=m if axis=='manufacturer' else f
                arrays=dict(identity=source)
                for control,mm,ff in [('null',m,f),('target',tm,tf)]:
                    arrays[control]=sample_wiring(c,bound,scheduler,item['image'][None].to(device),
                        torch.tensor([mm],device=device),torch.tensor([ff],device=device),
                        torch.tensor([item['echo_time_norm']],device=device),int(seed)+int(index),
                        point['timestep'],point['guidance'])[0,0]
                case=len(requests);array_file=f'arrays/case_{case:05d}.npz'
                with (output/array_file).open('xb') as handle:
                    np.savez_compressed(handle,source=source,null=arrays['null'],target=arrays['target'],
                        source_echo_time_norm=item['echo_time_norm'],target_echo_time_norm=item['echo_time_norm'],
                        target_manufacturer=tm,target_field_strength=tf)
                support=request_support(train.frame,tm,tf,float(item['echo_time_norm']),5)
                request=dict(case=case,patient_id=item['patient_id'],dataset_index=int(index),axis=axis,
                    direction=f'{original}_to_{1-original}',generation_seed=seed,training_seed=ck['training_seed'],
                    actual_sampling_seed=int(seed)+int(index),mode=ck['mode'],guide_control=guide_control,
                    guide_dataset_index=int(guide_index),guide_patient_id=val.frame.iloc[guide_index].patient_id,
                    source_echo_time_norm=float(item['echo_time_norm']),target_echo_time_norm=float(item['echo_time_norm']),
                    target_manufacturer=tm,target_field_strength=tf,array_file=array_file,
                    array_sha256=digest(output/array_file),**point,**support,**warning_for_index(source,index,thresholds),
                    evaluation_phase='engineering' if engineering else 'replicated_development_fixed_points')
                requests.append(request)
                native=source_native(np.stack([arrays[k] for k in ('identity','null','target')]),val.plans[index])
                metrics={k:measure(source,v) for k,v in arrays.items()}
                key,other=('mfg','fs') if axis=='manufacturer' else ('fs','mfg')
                case_grades=[]
                for name,judge in judges.items():
                    probabilities=predict_judge(judge,native,device)
                    null_hit=int(probabilities[key][1].argmax()==1-original)
                    for j,control in enumerate(('identity','null','target')):
                        hit=int(probabilities[key][j].argmax()==1-original)
                        case_grades.append(dict(request,judge=name,variant=control,target_hit=hit,
                            target_probability=float(probabilities[key][j,1-original]),
                            source_correct=int(probabilities[key][0].argmax()==original),paired_hit_delta=hit-null_hit,
                            other_axis_stable=int(probabilities[other][j].argmax()==probabilities[other][0].argmax()),
                            judge_input_frame='source_native_inverse_geometry',anatomy_assessment='unknown',**metrics[control]))
                grades.extend(case_grades)
                journal.write(json.dumps(json_safe(dict(request=request,grades=case_grades)),allow_nan=False)+'\n');journal.flush()
                if case%10==0:print('FULL GUIDE EVALUATION',ck['mode'],guide_control,axis,case,flush=True)
    table=pd.DataFrame(grades)
    csv_write(output/'requests.csv',pd.DataFrame(requests));csv_write(output/'per_image.csv',table)
    csv_write(output/'summary.csv',summarise(table,{'evaluation':p['evaluation']}))
    verify_inputs(evidence);json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,cases=len(requests),engineering=engineering,
        trained_checkpoint_sha256=digest(checkpoint),training_seed=ck['training_seed'],mode=ck['mode'],axis=axis,
        guide_control=guide_control,clinical_preservation='unknown',fresh_test=False,mask_warning_cases_retained=True))
    del model,base,c,scheduler,judges;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','checkpoint','run-id','axis'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--guide-control',default='correct');ap.add_argument('--device',default='cuda')
    ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
    evaluate(a.prepared,a.checkpoint,a.run_id,a.axis,a.guide_control,a.device,a.engineering)
