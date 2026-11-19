"""Bounded local, one-step manufacturer-response probe; no reverse sampling/training."""
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet,multiscale_guide
from scripts.v8_generator_dev.full_train import load_ema,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path
from scripts.v9_detail.masks import noise_aware_mask
from scripts.v11_guide_audit.analyse import CrossedBootstrap


def response_metrics(prediction,parent,interior):
    a=np.asarray(prediction,dtype=np.float64); b=np.asarray(parent,dtype=np.float64)
    if a.shape!=(3,1,128,128) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid epsilon response arrays')
    x=(a[1,0]-a[0,0])[interior]; y=(b[1,0]-b[0,0])[interior]
    if x.size<64:
        return {k:np.nan for k in ('label_response_rms','response_ratio_to_parent','response_cosine_to_parent','unconditional_adapter_shift_rms')}
    rms=float(np.sqrt(np.mean(x*x))); baseline=float(np.sqrt(np.mean(y*y)))
    norm=float(np.linalg.norm(x)*np.linalg.norm(y))
    return dict(label_response_rms=rms,response_ratio_to_parent=rms/baseline if baseline>1e-10 else np.nan,
        response_cosine_to_parent=float(np.dot(x,y)/norm) if norm>1e-12 else np.nan,
        unconditional_adapter_shift_rms=float(np.sqrt(np.mean((a[2,0]-b[2,0])[interior]**2))))


def prepare(run_id):
    p=ROOT/'reports/v10_detail/preparation_20261117_02'
    scientific=json.loads((p/'protocol.json').read_text())
    trained=ROOT/'reports/v10_detail/guide_replicates_20261117_01_training/complete.json'
    protocol=dict(format='local_epsilon_response_v1',previous_protocol=(p/'protocol.json').relative_to(ROOT).as_posix(),
        training_receipt=trained.relative_to(ROOT).as_posix(),selection=(p/'validation_selection.csv').relative_to(ROOT).as_posix(),
        training_seeds=scientific['training_seeds'],generation_seeds=[scientific['evaluation']['selection_generation_seed']],
        modes=['coarse_control','coarse_fine'],controls=['correct','zero','absent','wrong','coarse_only','fine_only'],
        timesteps=[100,300],source_patients=15,expected_rows=1080,maximum_local_seconds=600,
        bootstrap_seed=583371,bootstrap_resamples=10000,interactions_family=dict(quantiles=[.025,.975]),
        metrics=['label_response_rms','response_ratio_to_parent','response_cosine_to_parent','unconditional_adapter_shift_rms'],
        semantics='RMS/cosine of eps(target)-eps(source) at the same q(source,t,noise), plus eps(null-condition,guide)-eps(null-condition,parent); manufacturer label alone changes, EchoTime and field strength stay source.',
        limits='Post-hoc local surrogate after the paired-control outcome is known. One fixed generation seed; these q-source points are not reverse-trajectory states. No reverse sampling, judge success, anatomy validation, mechanism proof or advancement decision.',
        local_only=True,paid_gpu_calls=0)
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    json_write(output/'protocol.json',protocol)
    paths=[p/'protocol.json',p/'validation_selection.csv',trained,project_path(__file__)]
    trained_json=json.loads(trained.read_text())
    paths += [project_path(c['checkpoint']) for c in trained_json['copies'].values() if c['mode'] in protocol['modes']]
    json_write(output/'inputs.json',{x.relative_to(ROOT).as_posix():digest(x) for x in paths})
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.forward_probe prepare. protocol.json: fixed local '
        'forward-only scope, all source/control/replica/timestep rules and time cap; inputs.json: exact checkpoints/selection/code; '
        'complete.json: preparation only. No cloud permission or efficacy claim.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,probe_executed=False,paid_gpu_calls=0))


@torch.inference_mode()
def run(prepared,run_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    from scripts.v8_generator_dev.io import verify_inputs
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    if not torch.cuda.is_available():raise ValueError('This bounded local probe requires the available local GPU')
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_text(project_path(__file__).read_text(),encoding='utf-8')
    json_write(output/'protocol.json',p)
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.forward_probe run (source.py snapshot). '
        'protocol.json: prepared local-only design; journal.jsonl: all completed one-step observations, including on failure; '
        'per_forward.csv: every epsilon-response observation; paired_intervals.csv: all fine-minus-coarse controls/timesteps '
        'with descriptive crossed CIs; per_training_seed.csv: each replica patient CI; inputs.json: bound weights/code/data; '
        'complete.json: local scope, absent-parent checks and resource observation. No reverse edit or clinical/causal proof.\n',encoding='utf-8')
    started=time.monotonic();deterministic_runtime();torch.set_num_threads(2)
    device='cuda';torch.cuda.reset_peak_memory_stats()
    previous=json.loads(project_path(p['previous_protocol']).read_text())
    dataset=CanonicalDataset(previous['view'],'val')
    selected=pd.read_csv(project_path(p['selection'])).dataset_index.astype(int).tolist()
    c,base,scheduler,payload=load_ema(project_path(previous['parent']),device);del payload
    c.eval().requires_grad_(False);base.eval().requires_grad_(False)
    trained=json.loads(project_path(p['training_receipt']).read_text())
    cache={}; masks={};rows=[]; maximum_absent_error=0.;evidence=dict(json.loads((prepared/'inputs.json').read_text()))
    for index in selected:
        item=dataset[index];source=dataset.image_01(index)
        masks[index]=ndi.binary_erosion(noise_aware_mask(source)[0],iterations=8,border_value=0)
        m,f=int(item['manufacturer_label']),int(item['field_strength_label'])
        label=lambda value:torch.tensor([value],device=device)
        echo=torch.tensor([float(item['echo_time_norm'])],device=device)
        conditions=torch.cat([c(label(m),label(f),echo),c(label(1-m),label(f),echo),
                              c(label(m),label(f),echo,force_unconditional=True)])
        for generation in p['generation_seeds']:
            for t in p['timesteps']:
                generator=torch.Generator(device=device).manual_seed(generation+index)
                x=item['image'][None].to(device);noise=torch.randn(x.shape,device=device,generator=generator)
                xt=scheduler.q_sample(x,label(t),noise).repeat(3,1,1,1)
                times=torch.full((3,),t,device=device,dtype=torch.long)
                parent=base(xt,times,conditions).cpu().numpy()
                cache[index,generation,t]=(xt,times,conditions,parent)
    with (output/'journal.jsonl').open('x',encoding='utf-8') as log:
        for seed in p['training_seeds']:
            for mode in p['modes']:
                ck=trained['copies'][f'seed{seed}__{mode}'];path=project_path(ck['checkpoint'])
                if digest(path)!=ck['sha256']:raise ValueError('Checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=False)
                model=DetailConditionedUNet(base,previous['training']['adapter_width']).to(device).eval()
                model.adapter.load_state_dict(state['adapter']);model.requires_grad_(False);del state
                guides={i:multiscale_guide(torch.from_numpy(dataset.image_01(i).copy())[None,None].to(device),mode) for i in selected}
                for position,index in enumerate(selected):
                    item=dataset[index]
                    for generation in p['generation_seeds']:
                        for t in p['timesteps']:
                            xt,times,conditions,parent=cache[index,generation,t]
                            for control in p['controls']:
                                if time.monotonic()-started>p['maximum_local_seconds']:raise TimeoutError('Local forward probe cap reached; retain partial journal')
                                guide=guides[selected[(position+1)%len(selected)]] if control=='wrong' else guides[index]
                                guide=guide.clone()
                                if control=='zero':guide.zero_()
                                if control=='coarse_only':guide[:,3:]=0
                                if control=='fine_only':guide[:,:3]=0
                                present=torch.full((3,),0. if control=='absent' else 1.,device=device)
                                prediction=model(xt,times,conditions,guide.repeat(3,1,1,1),present).cpu().numpy()
                                if control=='absent':
                                    error=float(np.max(np.abs(prediction-parent)));maximum_absent_error=max(maximum_absent_error,error)
                                    np.testing.assert_allclose(prediction,parent,atol=2e-6,rtol=2e-6)
                                row=dict(patient_id=item['patient_id'],dataset_index=index,training_seed=seed,
                                    generation_seed=generation,timestep=t,mode=mode,guide_control=control,
                                    source_echo_time_norm=float(item['echo_time_norm']),target_echo_time_norm=float(item['echo_time_norm']),
                                    **response_metrics(prediction,parent,masks[index]))
                                rows.append(row);log.write(json.dumps(row,allow_nan=True)+'\n');log.flush()
                del model;gc.collect();torch.cuda.empty_cache()
                print('FORWARD COPY COMPLETE',seed,mode,len(rows),flush=True)
    frame=pd.DataFrame(rows)
    if len(frame)!=p['expected_rows']:raise ValueError('Incomplete local probe')
    csv_write(output/'per_forward.csv',frame)
    boot=CrossedBootstrap(p);summary=[];replicas=[]
    for (t,control),part in frame.groupby(['timestep','guide_control']):
        keys=['patient_id','training_seed','generation_seed']
        fine=part[part['mode']=='coarse_fine'].set_index(keys).sort_index()
        coarse=part[part['mode']=='coarse_control'].set_index(keys).sort_index()
        if not fine.index.equals(coarse.index):raise ValueError('Unpaired forward probe')
        delta=(fine[p['metrics']]-coarse[p['metrics']]).reset_index()
        result,per=boot.intervals(delta,p['metrics'])
        summary.extend(dict(timestep=t,guide_control=control,**r) for r in result)
        replicas.extend(dict(timestep=t,guide_control=control,**r) for r in per)
    csv_write(output/'paired_intervals.csv',pd.DataFrame(summary));csv_write(output/'per_training_seed.csv',pd.DataFrame(replicas))
    for path in (prepared/'protocol.json',project_path(__file__),ROOT/'scripts/v11_guide_audit/analyse.py'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,rows=len(rows),maximum_absent_epsilon_error=maximum_absent_error,
        local_seconds=time.monotonic()-started,peak_local_cuda_bytes=torch.cuda.max_memory_allocated(),
        generation_seeds=1,new_reverse_sampling=False,paid_gpu_calls=0,clinical_preservation='unknown',mechanism_proven=False))
    print('LOCAL FORWARD PROBE COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=['prepare','run']);a.add_argument('--run-id',required=True);a.add_argument('--prepared')
    q=a.parse_args();prepare(q.run_id) if q.action=='prepare' else run(q.prepared,q.run_id)
