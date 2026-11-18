"""Bounded local adapter training check; never full efficacy training or selection."""
import argparse
import hashlib
import json
import platform
import re
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scripts.v4_convergence.core import digest,json_write,load_models,cpu_state
from scripts.v5_anatomy.run_pilot import read_splits,project_path
from scripts.v6_structure.model import StructureConditionedUNet,structure_maps,photometric_view

ROOT=Path(__file__).resolve().parents[2]


def state_digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def training_rows(frame,patients,per_patient,seed):
    unique=frame.drop_duplicates('patient_id').sample(frac=1,random_state=seed)
    groups=[g.patient_id.tolist() for _,g in unique.groupby(['manufacturer_label','field_strength_label'])]
    order=[]
    while groups and len(order)<patients:
        for group in groups:
            if group and len(order)<patients:order.append(group.pop(0))
        groups=[g for g in groups if g]
    selected=[]
    for i,patient in enumerate(order):
        group=frame[frame.patient_id==patient]
        selected.append(group.sample(n=min(per_patient,len(group)),random_state=seed+i))
    return pd.concat(selected)


def train(config,run_id,device='cuda'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid run ID')
    cfg_path=project_path(config);cfg=json.loads(cfg_path.read_text())
    if cfg['stage']!='engineering_smoke_not_efficacy_training' or not 1<=cfg['steps']<=128:
        raise ValueError('This runner is restricted to a bounded engineering smoke')
    if not cfg['base_weights_frozen'] or not cfg['conditioner_frozen'] or cfg['raw_source_to_adapter']:
        raise ValueError('Version 1 requires a frozen base and no raw-source branch')
    parent=project_path(cfg['base_checkpoint'])
    if digest(parent)!=cfg['base_sha256']:raise ValueError('Base checkpoint mismatch')
    splits=read_splits()
    selected=training_rows(splits['train'],cfg['max_training_patients'],cfg['max_slices_per_training_patient'],cfg['seed'])
    sources=[];inputs={str(cfg_path.relative_to(ROOT)):digest(cfg_path),str(parent.relative_to(ROOT)):digest(parent)}
    for _,row in selected.iterrows():
        path=project_path(row.image_path);image=np.load(path).astype(np.float32)
        if image.shape!=(128,128) or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('Invalid frozen training image')
        sources.append(image);inputs[str(path.relative_to(ROOT))]=digest(path)
    for split in splits:
        path=ROOT/f'data/processed/ax_t1_128/splits/{split}.csv';inputs[str(path.relative_to(ROOT))]=digest(path)
    output=ROOT/'reports/v6_structure'/run_id
    checkpoint_dir=ROOT/'checkpoints/diffusion/ax_t1_128_v6_structure'/run_id
    if output.exists() or checkpoint_dir.exists():raise FileExistsError('Use a new run ID')
    output.mkdir(parents=True);checkpoint_dir.mkdir(parents=True)
    (output/'code').mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(path.relative_to(ROOT))]=digest(path);shutil.copyfile(path,output/'code'/path.name)
    json_write(output/'protocol.json',dict(**cfg,run_id=run_id,device=device,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        train_patients=int(selected.patient_id.nunique()),train_images=len(selected),
        training_supervision='self-conditioned real-image denoising only; no paired target scanner or expert mask labels',
        future_training_gate='Representation leakage audit, independent anatomy evaluation and a separate full-training protocol remain open'))
    json_write(output/'inputs.json',inputs);selected.to_csv(output/'training_selection.csv',index_label='training_row_index')
    (output/'README.md').write_text(
        '# Source-structure adapter engineering smoke\n\nProducer: `python -m scripts.v6_structure.train_smoke`.\n'
        'This bounded run tests optimisation and checkpoint/inference wiring; it is not efficacy training.\n'
        '| File | Meaning |\n| --- | --- |\n'
        '| protocol.json | predeclared settings, split and limitations |\n'
        '| inputs.json, code/ | hashes and source snapshot |\n'
        '| training_selection.csv | original training patients/images only |\n'
        '| training_steps.csv | actual noise/proxy losses and gradients |\n'
        '| complete.json | frozen-weight checks, timing, adapter checkpoint hash |\n'
        'All source masks are absent; gradient proxies are not anatomy truth. No clinical or publication performance claim.\n',encoding='utf-8')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    c,base,sched,ck=load_models(parent,device=device)
    c.requires_grad_(False).eval();model=StructureConditionedUNet(base,cfg['adapter_width']).to(device)
    before_base,before_c=state_digest(base),state_digest(c)
    model.train();optim=torch.optim.AdamW(model.adapter.parameters(),lr=cfg['learning_rate'])
    generator=torch.Generator(device=device).manual_seed(cfg['seed'])
    images=torch.tensor(np.stack(sources),device=device).unsqueeze(1)
    labels={k:torch.tensor(selected[column].to_numpy(),device=device) for k,column in [
        ('mfg','manufacturer_label'),('fs','field_strength_label'),('echo','echo_time_norm')]}
    rows=[];start=time.perf_counter()
    for step in range(cfg['steps']):
        idx=torch.randint(0,len(images),(cfg['batch_size'],),device=device,generator=generator)
        image=images[idx];clean=image*2-1
        with torch.no_grad():
            guide=structure_maps(photometric_view(image,generator)).detach()
            true_structure=structure_maps(image).detach()
            embedding=c(labels['mfg'][idx],labels['fs'][idx],labels['echo'][idx])
            unconditional=c(labels['mfg'][idx],labels['fs'][idx],labels['echo'][idx],force_unconditional=True)
            dropped=torch.rand(len(idx),device=device,generator=generator)<cfg['class_condition_dropout']
            embedding=torch.where(dropped[:,None],unconditional,embedding)
            present=(torch.rand(len(idx),device=device,generator=generator)>=cfg['structure_dropout']).float()
        t=torch.randint(0,sched.num_train_timesteps,(len(idx),),device=device,generator=generator)
        noise=torch.randn(clean.shape,device=device,generator=generator)
        xt=sched.q_sample(clean,t,noise)
        optim.zero_grad(set_to_none=True)
        predicted=model(xt,t,embedding,guide,present)
        noise_loss=F.mse_loss(predicted,noise)
        eligible=(t<=cfg['structure_consistency_max_timestep'])&(present>0)
        structural=predicted.sum()*0
        if eligible.any():
            abar=sched.alphas_cumprod[t[eligible]][:,None,None,None]
            x0=(xt[eligible]-torch.sqrt(1-abar)*predicted[eligible])/torch.sqrt(abar)
            recovered=((x0+1)/2).clamp(0,1)
            structural=F.smooth_l1_loss(structure_maps(recovered),true_structure[eligible])
        loss=noise_loss+cfg['structure_consistency_weight']*structural
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite adapter gradient')
        optim.step()
        if any(p.grad is not None for p in base.parameters()) or any(p.grad is not None for p in c.parameters()):
            raise AssertionError('Frozen model received gradients')
        rows.append(dict(step=step+1,noise_mse=float(noise_loss.detach()),
                         structure_proxy_loss=float(structural.detach()),total_loss=float(loss.detach()),
                         adapter_grad_norm=float(norm),structure_present=int(present.sum()),
                         structural_loss_examples=int(eligible.sum())))
        if step==0 or (step+1)%8==0:print('STEP',rows[-1],flush=True)
    assert state_digest(base)==before_base and state_digest(c)==before_c
    pd.DataFrame(rows).to_csv(output/'training_steps.csv',index=False)
    checkpoint=checkpoint_dir/'adapter_smoke.pt'
    payload=dict(format='structure_adapter_v1',adapter_state_dict=cpu_state(model.adapter),
                 adapter_width=cfg['adapter_width'],base_checkpoint=cfg['base_checkpoint'],
                 base_sha256=cfg['base_sha256'],steps=cfg['steps'],config=cfg,
                 engineering_smoke=True,expert_anatomy_validated=False)
    with checkpoint.open('xb') as handle:torch.save(payload,handle)
    for path,expected in inputs.items():
        if digest(ROOT/path)!=expected:raise AssertionError(f'Input changed during training: {path}')
    result=dict(complete=True,steps=len(rows),seconds=time.perf_counter()-start,
                trainable_parameters=sum(p.numel() for p in model.adapter.parameters()),
                frozen_base_parameters=sum(p.numel() for p in base.parameters()),
                base_weights_unchanged=True,conditioner_unchanged=True,train_split_only=True,
                source_checkpoint_unchanged=True,checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=digest(checkpoint),engineering_smoke=True,efficacy_established=False,
                expert_anatomy_validated=False,cloud_used=False)
    json_write(output/'complete.json',result);print(result,flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/structure_adapter_v1.json')
    p.add_argument('--run-id',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();print(train(a.config,a.run_id,a.device))
