"""Matched inference wiring checks with correct/absent/wrong source guides.

The adapter is only engineering-trained. These outputs must not be used as
evidence that anatomy preservation or scanner editing has improved.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scripts.v4_convergence.core import digest,json_write,load_models,sample,target,probabilities
from scripts.v2_quickwins.judge_model import load_judge
from scripts.v5_anatomy.run_pilot import read_splits,select_validation,project_path,support_flags,summarise
from scripts.v5_anatomy.metrics import diagnostics
from scripts.v6_structure.model import StructureConditionedUNet,BoundStructureModel,structure_maps

ROOT=Path(__file__).resolve().parents[2]
B2=ROOT/'reports/v4_heldout_judge/b2_local_20261016_01'


def validation_rows(frame,n,seed):
    order=select_validation(frame,frame.patient_id.nunique(),seed)
    first=order.drop_duplicates(['manufacturer_label','field_strength_label'])
    first=first.iloc[:n]
    return pd.concat([first,order[~order.patient_id.isin(first.patient_id)]]).iloc[:n]


def evaluate(checkpoint,output,device='cuda'):
    checkpoint=project_path(checkpoint);output=project_path(output)
    if not output.is_relative_to(ROOT/'reports'):raise ValueError('Use a new reports directory')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('format')!='structure_adapter_v1' or not ck['engineering_smoke']:
        raise ValueError('Expected the engineering adapter checkpoint')
    parent=project_path(ck['base_checkpoint'])
    if digest(parent)!=ck['base_sha256']:raise ValueError('Frozen base mismatch')
    cfg=ck['config'];splits=read_splits()
    chosen=validation_rows(splits['val'],cfg['evaluation_patients'],cfg['seed'])
    if len(chosen)<2:raise ValueError('Need distinct patients for wrong-source controls')
    source=np.stack([np.load(project_path(row.image_path)).astype(np.float32) for _,row in chosen.iterrows()])
    inputs={str(checkpoint.relative_to(ROOT)):digest(checkpoint),str(parent.relative_to(ROOT)):digest(parent)}
    for _,row in chosen.iterrows():
        path=project_path(row.image_path);inputs[str(path.relative_to(ROOT))]=digest(path)
    output.mkdir(parents=True,exist_ok=False);(output/'arrays').mkdir();(output/'review').mkdir()
    protocol=dict(**cfg,split='val',evaluation_role='engineering wiring/ablation check only; not efficacy',
        patients=len(chosen),proxy_thresholds=[.03,.05,.1],
        proxy_gates=dict(min_dice=.9,max_boundary_hd95_px=3.,max_centroid_shift_px=2.,max_area_fraction_change=.1),
        statistics=dict(bootstrap_resamples=10000,bootstrap_seed=731),
        frozen_assessors='three held-out orientation-robust judges; descriptive development assessment only')
    json_write(output/'protocol.json',protocol);chosen.to_csv(output/'selection.csv',index_label='validation_row_index')
    c,base,sched,_=load_models(parent,device);c.requires_grad_(False).eval()
    model=StructureConditionedUNet(base,ck['adapter_width']).to(device)
    model.adapter.load_state_dict(ck['adapter_state_dict']);model.eval();torch.set_num_threads(4)
    with torch.no_grad():guides=structure_maps(torch.tensor(source,device=device).unsqueeze(1))
    generation=[];mapping=[]
    for axis,setting in cfg['settings'].items():
        for i,(_,row) in enumerate(chosen.iterrows()):
            case=f'{axis}_{i:03d}';seed=cfg['sampling_seed']+i;wrong=(i+1)%len(chosen)
            src=dict(mfg=np.array([row.manufacturer_label],np.int64),fs=np.array([row.field_strength_label],np.int64),
                     echo=np.array([row.echo_time_norm],np.float32));tgt=target(src,axis)
            flags=support_flags(splits['train'],src,tgt);outputs={}
            for variant in ['reference','source_guide','guide_absent','wrong_source_guide']:
                if variant=='reference':net=base
                else:
                    guide=guides[wrong:wrong+1] if variant=='wrong_source_guide' else guides[i:i+1]
                    present=torch.zeros(1,device=device) if variant=='guide_absent' else None
                    net=BoundStructureModel(model,guide,present)
                outputs[variant]=sample(c,net,sched,tgt,[seed],setting['t'],setting['w'],source[i:i+1])[0]
            # A missing guide must recover the frozen model exactly after training.
            if not np.array_equal(outputs['reference'],outputs['guide_absent']):
                raise AssertionError('Missing-guide control does not reproduce the frozen sampler')
            np.savez_compressed(output/'arrays'/f'{case}.npz',source=source[i],**outputs)
            for variant,image in outputs.items():
                rid=hashlib.sha256(f'v6/{case}/{variant}/{seed}'.encode()).hexdigest()[:12]
                for suffix,array in [('source',source[i]),('edit',image)]:
                    Image.fromarray(np.round(array*255).astype(np.uint8)).save(output/'review'/f'{rid}_{suffix}.png')
                generation.append(dict(case_id=case,review_id=rid,patient_id=row.patient_id,axis=axis,variant=variant,
                    seed=seed,timestep=setting['t'],guidance=setting['w'],source_path=row.image_path,
                    source_mfg=int(src['mfg'][0]),source_fs=int(src['fs'][0]),target_mfg=int(tgt['mfg'][0]),target_fs=int(tgt['fs'][0]),
                    source_echo=float(src['echo'][0]),target_echo=float(tgt['echo'][0]),
                    wrong_source_patient=chosen.iloc[wrong].patient_id if variant=='wrong_source_guide' else None,
                    human_preservation_pass=None,**flags,**diagnostics(source[i],image,protocol['proxy_thresholds'],protocol['proxy_gates'])))
                mapping.append(dict(review_id=rid,case_id=case,variant=variant,source_png=f'{rid}_source.png',edit_png=f'{rid}_edit.png'))
            print('SAMPLED',case,flush=True)
    del model,base,c,sched,ck;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    generation=pd.DataFrame(generation);generation.to_csv(output/'generation.csv',index=False)
    shuffled=pd.DataFrame(mapping).sample(frac=1,random_state=731)
    shuffled.to_csv(output/'review/private_mapping.csv',index=False)
    review=shuffled[['review_id','source_png','edit_png']].copy()
    for key in ['reviewer','anatomy_preserved','lesion_assessable','lesion_preserved','comment']:review[key]=''
    review.to_csv(output/'review/review_template.csv',index=False)
    json_write(output/'review/annotation_template.json',dict(cases=[dict(review_id=row.review_id,reviewer=None,
        source_mask=None,edit_mask=None,source_landmarks=None,edit_landmarks=None) for row in shuffled.itertuples()]))
    rows=[]
    frozen=B2/'frozen_checkpoints.json';inputs[str(frozen.relative_to(ROOT))]=digest(frozen)
    for entry in json.loads(frozen.read_text()):
        if entry['family']!='orientation_robust':continue
        path=B2/entry['checkpoint']
        if digest(path)!=entry['sha256']:raise ValueError('Frozen judge mismatch')
        inputs[str(path.relative_to(ROOT))]=entry['sha256'];judge=load_judge(path,device)
        for case,group in generation.groupby('case_id',sort=False):
            with np.load(output/'arrays'/f'{case}.npz') as pack:
                variants=group.variant.tolist();images=np.stack([pack['source']]+[pack[v] for v in variants])
            probs=probabilities(judge,images,batch=8)
            for i,record in enumerate(group.to_dict('records'),1):
                head='fs' if record['axis']=='field_strength' else 'mfg';other='mfg' if head=='fs' else 'fs'
                label=record['target_fs' if head=='fs' else 'target_mfg'];hit=bool(probs[head][i].argmax()==label)
                rows.append(dict(**record,judge=f'heldout_orientation_robust{entry["seed"]}',
                    target_hit=hit,target_gain=float(probs[head][i,label]-probs[head][0,label]),
                    target_probability_before=float(probs[head][0,label]),target_probability_after=float(probs[head][i,label]),
                    other_axis_stable=bool(probs[other][i].argmax()==probs[other][0].argmax()),
                    target_hit_and_proxy_screen=hit and record['proxy_screen_pass'],target_hit_and_human_preservation=None))
        del judge
    frame=pd.DataFrame(rows);frame.to_csv(output/'predictions.csv',index=False)
    summarise(frame,protocol).to_csv(output/'summary.csv',index=False)
    for path,expected in inputs.items():
        if digest(ROOT/path)!=expected:raise AssertionError('Frozen input changed')
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,engineering_only=True,patients=len(chosen),
        requests=generation.case_id.nunique(),output_images=len(generation),judge_rows=len(frame),
        missing_guide_matches_base_bitwise=True,efficacy_established=False,expert_anatomy_validated=False,
        test_images_used=False,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint)))
    (output/'README.md').write_text('# Structure adapter inference smoke\n\n'
        'Producer: `python -m scripts.v6_structure.evaluate_smoke`.\n'
        'protocol.json and selection.csv declare validation patients/settings. arrays/ stores matched outputs and source.\n'
        'generation.csv contains proxy measurements and support flags. predictions.csv and summary.csv retain each\n'
        'held-out judge separately with patient-bootstrap CIs. review/ contains blank blinded-review templates and PNGs.\n'
        'inputs.json hashes frozen inputs; complete.json checks bitwise base recovery when the guide is absent.\n'
        'Only a short engineering-trained adapter is evaluated; this is not an efficacy trial or anatomy validation.\n'
        'Correct/absent/wrong-source guide ablations diagnose wiring, not clinical usefulness.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();print(evaluate(a.checkpoint,a.output,a.device))
