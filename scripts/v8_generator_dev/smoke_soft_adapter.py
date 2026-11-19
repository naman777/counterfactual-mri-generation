"""Bounded canonical-backbone adapter integration, with no efficacy claim."""
import argparse
import gc
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.v6_structure.model import StructureConditionedUNet, BoundStructureModel, photometric_view
from scripts.v6_structure.train_smoke import state_digest
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import load_ema, deterministic_runtime
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.soft_structure import soft_structure


def run(run_id, device='cuda'):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new engineering ID required')
    deterministic_runtime()
    base = ROOT / 'reports/v8_generator_dev'
    fetched_path = base / 'failed_pair_checkpoints_20261116_01/complete.json'
    fetched = json.loads(fetched_path.read_text())
    view = 'data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    dataset = CanonicalDataset(view,'train')
    selection = dataset.frame.sample(frac=1, random_state=482947).drop_duplicates('patient_id').iloc[:2]
    output = base / run_id
    checkpoints = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    checkpoints.mkdir(exist_ok=False)
    json_write(output / 'protocol.json',dict(stage='engineering_only',steps_per_variant=2,batch_size=1,
        source_selection_seed=482947,noise_seed=482948,learning_rate=1e-4,adapter_width=16,
        structure_proxy_weight=.1,timestep=50,guide='soft_local_order_3x32x32',
        guide_augmentation='Source-only geometry-preserving photometric view',
        supervision='Self-conditioned denoising plus differentiable guide consistency; no paired target scanner/anatomy labels',
        guide_and_acquisition_inputs_independent=True,base_and_conditioner_frozen=True,
        raw_source_bypass=False,efficacy_established=False,anatomy_validated=False,
        inference_cross_version_only=True,full_training_resume_claimed=False,device=device))
    csv_write(output / 'training_selection.csv',selection.reset_index(names='dataset_index'))
    inputs = {fetched_path.relative_to(ROOT).as_posix():digest(fetched_path)}
    source_images = [dataset.image_01(int(index)) for index in selection.index]
    for source in selection.itertuples():
        path=project_path(source.source_image_path)
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    records, checks = [], {}
    for variant in ('additive','scale_shift'):
        parent=project_path(fetched['checkpoints'][variant])
        parent_hash=digest(parent)
        inputs[parent.relative_to(ROOT).as_posix()]=parent_hash
        c,unet,scheduler,payload=load_ema(parent,device)
        del payload
        c.requires_grad_(False).eval()
        torch.manual_seed(482948)
        model=StructureConditionedUNet(unet,width=16).to(device).train()
        original_base,original_c=state_digest(unet),state_digest(c)
        original_adapter=state_digest(model.adapter)
        optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=1e-4)
        generator=torch.Generator(device=device).manual_seed(482948)
        zero_equal=False
        for step,(index,row) in enumerate(selection.iterrows()):
            image=torch.from_numpy(source_images[step]).to(device)[None,None]
            clean=image*2-1
            guide=soft_structure(photometric_view(image,generator)).detach()
            true_structure=soft_structure(image).detach()
            condition=c(torch.tensor([int(row.manufacturer_label)],device=device),
                        torch.tensor([int(row.field_strength_label)],device=device),
                        torch.tensor([np.float32(row.echo_time_norm)],device=device),force_unconditional=step==1)
            t=torch.tensor([50],device=device)
            noise=torch.randn(clean.shape,device=device,generator=generator)
            xt=scheduler.q_sample(clean,t,noise)
            if step==0:
                with torch.no_grad():
                    zero_equal=torch.equal(unet(xt,t,condition),model(xt,t,condition,guide))
                if not zero_equal:
                    raise AssertionError('Zero adapter changed the canonical backbone output')
            optimizer.zero_grad(set_to_none=True)
            prediction=model(xt,t,condition,guide)
            noise_loss=F.mse_loss(prediction,noise)
            abar=scheduler.alphas_cumprod[t][:,None,None,None]
            x0=(xt-torch.sqrt(1-abar)*prediction)/torch.sqrt(abar)
            structure_loss=F.smooth_l1_loss(soft_structure(((x0+1)/2).clamp(0,1)),true_structure)
            loss=noise_loss+.1*structure_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adapter smoke loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite adapter smoke gradient')
            optimizer.step()
            if any(p.grad is not None for p in list(unet.parameters())+list(c.parameters())):
                raise AssertionError('Frozen backbone/conditioner received gradients')
            records.append(dict(model=variant,step=step+1,patient_id=row.patient_id,dataset_index=int(index),
                noise_mse=float(noise_loss.detach()),guide_proxy_loss=float(structure_loss.detach()),
                total_loss=float(loss.detach()),gradient_norm=float(norm),condition_dropped=step==1,
                anatomy_validated=False))
        if original_base!=state_digest(unet) or original_c!=state_digest(c) or digest(parent)!=parent_hash:
            raise AssertionError('Frozen base/conditioner/parent bytes changed')
        if original_adapter==state_digest(model.adapter):
            raise AssertionError('Adapter parameters did not update')
        path=checkpoints/(variant+'.pt')
        with path.open('xb') as handle:
            torch.save(dict(engineering=True,variant=variant,base_sha256=parent_hash,
                guide='soft_local_order_3x32x32',adapter_state_dict=model.adapter.state_dict(),
                efficacy_established=False,anatomy_validated=False),handle)
        checks[variant]=dict(zero_adapter_equal=zero_equal,frozen_base_and_conditioner_unchanged=True,
            adapter_updated=True,parameters=sum(p.numel() for p in model.adapter.parameters()),
            checkpoint=str(path.relative_to(ROOT)),sha256=digest(path))
        print('SOFT ADAPTER ENGINEERING',variant,flush=True)
        del c,unet,scheduler,model,optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    csv_write(output/'steps.csv',pd.DataFrame(records))
    json_write(output/'checks.json',checks)
    for name in ('smoke_soft_adapter.py','soft_structure.py','test_soft_structure.py'):
        path=ROOT/'scripts/v8_generator_dev'/name
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in ('model.py','train_smoke.py'):
        path=ROOT/'scripts/v6_structure'/name
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,engineering_integration_passed=True,
        steps_per_variant=2,base_weights_changed=False,efficacy_established=False,
        acquisition_invariance_established=False,anatomy_validated=False,test_pixels_used=False))
    (output/'README.md').write_text(
        '# Canonical soft-guide adapter integration\n\nProducer: `python -m scripts.v8_generator_dev.smoke_soft_adapter`.\n'
        'protocol.json/training_selection.csv freeze the two training-only sources and engineering scope. '
        'steps.csv records actual noise/guide losses and gradients. checks.json maps new engineering-only '
        'adapter checkpoints, parameter counts, zero-init and unchanged frozen weights. inputs.json hashes '
        'sources/code/parents. complete.json describes integration only. The representation retains '
        'acquisition information; no preservation efficacy, anatomy truth or backbone winner is established.\n',encoding='utf-8')
    print('SOFT ADAPTER SMOKE COMPLETE',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    run(args.run_id,args.device)
