"""Matched source-guide/learned-zero-guide pilot with immutable frozen parents."""
import argparse
import gc
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.v6_structure.model import StructureConditionedUNet, photometric_view
from scripts.v8_generator_dev.data import CanonicalDataset, sampling_probabilities
from scripts.v8_generator_dev.full_train import CachedCanonical, deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.model import cpu_state
from scripts.v8_generator_dev.precision_step import precision_step
from scripts.v8_generator_dev.soft_structure import soft_structure


MODES = ('soft_source_guide', 'learned_zero_guide')
SEED = 482952


def state_digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pilot_settings():
    return dict(seed=SEED, epochs=20, block_epochs=5, batch_size=32, learning_rate=1e-4,
        weight_decay=.01, gradient_clip=1., acquisition_drop_probability=.1,
        structure_proxy_weight=.1, structure_proxy_max_timestep=300, adapter_width=16,
        max_same_batch_precision_retries=8, guide='soft_local_order_3x32x32',
        primary_adapter_weights='raw common final epoch; no checkpoint selection or EMA',
        training_objective='Full-timestep noise MSE plus low-timestep self-source guide consistency',
        guide_supervision='Source only; no paired scanner target or clinical annotations',
        scope='Development pilot; acquisition leakage and source-copy shortcuts unresolved')


def draw(dataset, weights, step, batch_size, seed=SEED):
    """Independent explicit streams; guide availability cannot change data/noise/dropout."""
    batches = (len(dataset) + batch_size-1)//batch_size
    epoch, within = divmod(step, batches)
    order = torch.multinomial(weights, len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(seed+epoch))
    indices = order[within*batch_size:(within+1)*batch_size].tolist()
    items = [dataset[index] for index in indices]
    clean = torch.stack([item['image'] for item in items])
    rng = torch.Generator().manual_seed(seed+2000000+step)
    ts = torch.randint(1000, (len(items),), generator=rng)
    noise = torch.randn(clean.shape, generator=rng)
    dropped = torch.rand(len(items), generator=torch.Generator().manual_seed(seed+3000000+step)) < .1
    return items, clean, ts, noise, dropped, dict(global_step=step+1, epoch_at_draw=epoch,
        sample_indices=indices, patient_ids=[item['patient_id'] for item in items],
        timesteps=ts.tolist(), condition_dropped=dropped.tolist(),
        noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())


def guide_for(image, mode, step, augment=True):
    if mode not in MODES:
        raise ValueError('Unknown predeclared guide mode')
    # Both conditions consume the identical photometric draw and compute their
    # guide; only the resulting source information is removed for the baseline.
    rng = torch.Generator(device=image.device).manual_seed(SEED+4000000+step)
    with torch.no_grad():
        guide = soft_structure(photometric_view(image, rng) if augment else image)
    return guide if mode == 'soft_source_guide' else torch.zeros_like(guide)


def independent_condition(conditioner, items, dropped, device):
    conditioner.eval()
    m = torch.tensor([item['manufacturer_label'] for item in items], device=device)
    f = torch.tensor([item['field_strength_label'] for item in items], device=device)
    echo = torch.tensor([item['echo_time_norm'] for item in items], device=device)
    with torch.no_grad():
        conditional = conditioner(m, f, echo)
        unconditional = conditioner(m, f, echo, force_unconditional=True)
    return torch.where(dropped.to(device)[:, None], unconditional, conditional)


def objective(prediction, noise, xt, ts, scheduler, true_structure, settings):
    noise_loss = F.mse_loss(prediction.float(), noise.float())
    chosen = ts <= settings['structure_proxy_max_timestep']
    structure_loss = prediction.float().sum()*0
    if chosen.any():
        abar = scheduler.alphas_cumprod[ts[chosen]][:, None, None, None].float()
        x0 = (xt[chosen].float()-torch.sqrt(1-abar)*prediction[chosen].float())/torch.sqrt(abar)
        structure_loss = F.smooth_l1_loss(soft_structure(((x0+1)/2).clamp(0,1)), true_structure[chosen])
    return noise_loss + settings['structure_proxy_weight']*structure_loss, noise_loss, structure_loss


def train_copy(parent, mode, dataset, settings, destination, log_path, start_step=0,
               stop_step=None, resume=None, device='cuda', engineering=False):
    parent = project_path(parent)
    parent_hash = digest(parent)
    c, base, scheduler, payload = load_ema(parent, device)
    config, backbone = payload['config'], payload['variant']
    del payload
    if scheduler.num_train_timesteps != 1000:
        raise ValueError('Pilot protocol requires the original 1000 timesteps')
    c.requires_grad_(False).eval()
    torch.manual_seed(settings['seed'])
    model = StructureConditionedUNet(base, settings['adapter_width']).to(device).train()
    frozen_base, frozen_c = state_digest(base), state_digest(c)
    initial_adapter = state_digest(model.adapter)
    params = list(model.adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    amp = torch.device(device).type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if resume:
        ck = torch.load(project_path(resume), map_location='cpu', weights_only=False)
        for key, expected in [('parent_sha256',parent_hash),('mode',mode),('settings',settings),
                              ('global_step',start_step),('backbone',backbone),('engineering',engineering)]:
            if ck[key] != expected:
                raise ValueError('Adapter continuation mismatch: '+key)
        model.adapter.load_state_dict(ck['adapter_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scaler.load_state_dict(ck['scaler_state_dict'])
        del ck
    elif start_step:
        raise ValueError('Nonzero progress requires an exact previous adapter state')
    weights = torch.from_numpy(sampling_probabilities(dataset.frame, config['data']['minimum_cell_patients']))
    batches = (len(dataset)+settings['batch_size']-1)//settings['batch_size']
    stop_step = settings['epochs']*batches if stop_step is None else stop_step
    if not start_step < stop_step <= settings['epochs']*batches:
        raise ValueError('Invalid planned adapter exposure')
    first_zero_equal, count, loss_sums = None, 0, np.zeros(3)
    with log_path.open('x', encoding='utf-8') as log:
        for step in range(start_step, stop_step):
            items, clean, ts, noise, dropped, record = draw(dataset, weights, step, settings['batch_size'], settings['seed'])
            clean, ts, noise = [value.to(device) for value in (clean,ts,noise)]
            image = (clean+1)/2
            guide = guide_for(image,mode,step)
            true_structure = soft_structure(image).detach()
            condition = independent_condition(c,items,dropped,device)
            xt = scheduler.q_sample(clean,ts,noise)
            if step == 0:
                with torch.no_grad():
                    first_zero_equal = torch.equal(base(xt,ts,condition),model(xt,ts,condition,guide))
                if not first_zero_equal:
                    raise AssertionError('Zero-initialised adapter altered the parent output')
            pieces = {}
            def forward_loss():
                with torch.autocast(torch.device(device).type,dtype=torch.float16,enabled=amp):
                    prediction = model(xt,ts,condition,guide)
                total, noise_loss, structure_loss = objective(prediction,noise,xt,ts,scheduler,true_structure,settings)
                pieces.update(noise_mse=float(noise_loss.detach()),structure_proxy_loss=float(structure_loss.detach()))
                return total
            loss, norm, precision = precision_step(forward_loss,params,optimizer,scaler,device,
                settings['gradient_clip'],settings['max_same_batch_precision_retries'])
            if any(p.grad is not None for p in list(base.parameters())+list(c.parameters())):
                raise AssertionError('Frozen parent received gradients')
            record.update(mode=mode,backbone=backbone,**precision,**pieces,
                total_loss=float(loss.detach()),gradient_norm=float(norm),echo_time_fixed=True)
            log.write(json.dumps(record,allow_nan=False)+'\n')
            log.flush()
            loss_sums += [record['total_loss'],record['noise_mse'],record['structure_proxy_loss']]
            count += 1
            if step%100 == 0:
                print('ADAPTER UPDATE',backbone,mode,step+1,'epoch',step//batches,flush=True)
    checks = dict(parent_bytes_unchanged=digest(parent)==parent_hash,
        base_weights_unchanged=state_digest(base)==frozen_base,
        conditioner_unchanged=state_digest(c)==frozen_c,
        adapter_updated=state_digest(model.adapter)!=initial_adapter,
        adapter_finite=all(torch.isfinite(value).all().item() for value in model.adapter.state_dict().values()),
        zero_init_equal=first_zero_equal)
    if not all(value for key,value in checks.items() if key!='zero_init_equal'):
        raise AssertionError('Adapter/frozen-parent post-update audit failed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as handle:
        torch.save(dict(format='soft_adapter_pilot_v1',parent=str(parent.relative_to(ROOT)),parent_sha256=parent_hash,
            backbone=backbone,mode=mode,settings=settings,global_step=stop_step,epoch=stop_step//batches,
            adapter_state_dict=cpu_state(model.adapter),optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),engineering=engineering,checks=checks,
            torch_version=str(torch.__version__),anatomy_validated=False),handle)
    result = dict(checks=checks,checkpoint=str(destination.relative_to(ROOT)),sha256=digest(destination),
        global_step=stop_step,epoch=stop_step//batches,updates=count,mean_losses=(loss_sums/count).tolist())
    del model,base,c,scheduler,optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train(prepared,run_id,device='cuda',engineering_steps=None):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new pilot ID required')
    prepared = project_path(prepared)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    protocol = json.loads((prepared/'protocol.json').read_text())
    settings = protocol['settings']
    if settings != pilot_settings():
        raise ValueError('Pilot settings changed from the frozen proposal implementation')
    if engineering_steps is not None and not 1 <= engineering_steps <= 4:
        raise ValueError('Engineering exposure is limited to four batches per copy')
    if engineering_steps is None and protocol.get('approved_additional_usd') != 8:
        raise ValueError('A separate explicit additional US$8 approval is required')
    deterministic_runtime()
    dataset = CachedCanonical(CanonicalDataset(protocol['view'],'train'))
    output = ROOT/'reports/v8_generator_dev'/run_id
    checkpoints = ROOT/'checkpoints/diffusion/ax_t1_128_v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    checkpoints.mkdir(exist_ok=False)
    json_write(output/'protocol.json',protocol)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.adapter_pilot. protocol.json binds parents/settings; '
        'block_*/ contains append-only per-copy draw/update JSONL and audit results, then a common '
        'completion marker. complete.json exists only after all planned copies complete. '
        'inputs.json hashes frozen dependencies. Engineering runs are not efficacy training.\n',encoding='utf-8')
    batches = (len(dataset)+settings['batch_size']-1)//settings['batch_size']
    epochs = [None] if engineering_steps else list(range(settings['block_epochs'],settings['epochs']+1,settings['block_epochs']))
    previous, start_step = {}, 0
    for number,epoch in enumerate(epochs,1):
        block = output/f'block_{number:04d}'
        block.mkdir()
        stop_step = engineering_steps or epoch*batches
        results, matched = {}, None
        for backbone,parent in protocol['parents'].items():
            for mode in MODES:
                key = backbone+'__'+mode
                result = train_copy(parent,mode,dataset,settings,checkpoints/f'{key}_{stop_step:06d}.pt',
                    block/(key+'.jsonl'),start_step,stop_step,previous.get(key),device,engineering_steps is not None)
                json_write(block/(key+'.json'),result)
                # Compare every actual source/timestep/noise/dropout draw, not just seeds.
                draws = [{name:row[name] for name in ('global_step','sample_indices','patient_ids','timesteps',
                    'condition_dropped','noise_sha256')} for row in map(json.loads,(block/(key+'.jsonl')).read_text().splitlines())]
                if matched is None:
                    matched = draws
                elif matched != draws:
                    raise AssertionError('Pilot copies received unmatched training draws')
                results[key] = result
                previous[key] = result['checkpoint']
        json_write(block/'complete.json',dict(common_step=stop_step,common_epoch=stop_step//batches,
            copies=results,matched_draws=True,engineering=engineering_steps is not None))
        start_step = stop_step
        print('ADAPTER COMMON BLOCK COMPLETE',stop_step//batches,flush=True)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    json_write(output/'inputs.json',{(prepared/'protocol.json').relative_to(ROOT).as_posix():digest(prepared/'protocol.json'),
        **json.loads((prepared/'inputs.json').read_text())})
    json_write(output/'complete.json',dict(complete=True,engineering=engineering_steps is not None,
        epoch=start_step//batches,global_step=start_step,copies=results,all_four_copies_matched=True,
        frozen_parents=True,test_pixels_used=False,anatomy_validated=False,efficacy_established=False))
    print('ADAPTER TRAINING COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--engineering-steps',type=int)
    a=p.parse_args()
    train(a.prepared,a.run_id,a.device,a.engineering_steps)
