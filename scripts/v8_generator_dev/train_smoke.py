"""Matched, capped local R0/R1 optimisation/EMA/checkpoint/inference check."""
import argparse
import gc
import json
import platform
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.v8_generator_dev.data import CanonicalDataset, PatientCellSampler
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.model import build_models, clone_ema, cpu_state, update_ema
from scripts.v8_generator_dev.prepare import validate_config


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'canonical_generator_dev_v1':
        raise ValueError('Use the dedicated loader for canonical generator checkpoints')
    conditioner, model, scheduler = build_models(checkpoint['config'], checkpoint['variant'], device)
    conditioner.load_state_dict(checkpoint['conditioning_state_dict'])
    model.load_state_dict(checkpoint['unet_state_dict'])
    return conditioner.eval(), model.eval(), scheduler.eval(), checkpoint


@torch.inference_mode()
def validate(dataset, indices, conditioner, model, scheduler, seed):
    conditioner.eval()
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for index in indices:
        item = dataset[index]
        generator = torch.Generator().manual_seed(seed + int(index))
        timestep = torch.randint(scheduler.num_train_timesteps, (1,), generator=generator)
        noise = torch.randn((1, 1, 128, 128), generator=generator).to(device)
        clean = item['image'].unsqueeze(0).to(device)
        timestep = timestep.to(device)
        condition = conditioner(torch.tensor([item['manufacturer_label']], device=device),
                                torch.tensor([item['field_strength_label']], device=device),
                                torch.tensor([item['echo_time_norm']], device=device))
        prediction = model(scheduler.q_sample(clean, timestep, noise), timestep, condition)
        loss = float(F.mse_loss(prediction, noise))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite fixed validation loss')
        rows.append(dict(patient_id=item['patient_id'], dataset_index=int(index),
                         timestep=int(timestep.item()), noise_mse=loss))
    return pd.DataFrame(rows)


@torch.inference_mode()
def sample_wiring(conditioner, model, scheduler, images, mfg, fs, echo, seed, timestep, guidance):
    """Very short ancestral CFG smoke; not a clinically meaningful edit setting."""
    conditioner.eval()
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = images.shape
    noise = torch.randn(shape, device=device, generator=generator)
    ts = torch.full((len(images),), timestep, device=device, dtype=torch.long)
    x = scheduler.q_sample(images, ts, noise)
    ce = conditioner(mfg, fs, echo)
    ue = conditioner(mfg, fs, echo, force_unconditional=True)
    for t in range(timestep, -1, -1):
        ts = torch.full((len(images),), t, device=device, dtype=torch.long)
        unconditional = model(x, ts, ue)
        conditional = model(x, ts, ce)
        eps = unconditional + guidance * (conditional - unconditional)
        beta, alpha, abar = scheduler.betas[t], scheduler.alphas[t], scheduler.alphas_cumprod[t]
        x = torch.rsqrt(alpha) * (x - beta / torch.sqrt(1 - abar) * eps)
        if t:
            x = x + torch.sqrt(beta) * torch.randn(shape, device=device, generator=generator)
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite CFG inference')
    return ((x.clamp(-1, 1) + 1) / 2).cpu().numpy().astype(np.float32)


def run_variant(config, variant, train, val, indices, output, checkpoint_dir, device):
    smoke, training = config['smoke'], config['training']
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    conditioner, model, scheduler = build_models(config, variant, device)
    ema_c, ema_u = clone_ema(conditioner), clone_ema(model)
    initial_c, initial_u = cpu_state(conditioner), cpu_state(model)
    params = list(conditioner.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=training['learning_rate'], weight_decay=training['weight_decay'])
    use_amp = bool(training['mixed_precision'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    sampler = PatientCellSampler(train, config['seed'], smoke['steps_per_variant'] * smoke['batch_size'],
                                  config['data']['minimum_cell_patients'])
    # DataLoader and denoising randomness have dedicated streams, so model
    # construction and differing parameter counts do not change data/noise draws.
    loader = DataLoader(train, sampler=sampler, batch_size=smoke['batch_size'], num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed'] + 10))
    random = torch.Generator().manual_seed(config['seed'] + 20)
    torch.manual_seed(config['seed'] + 30)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'] + 30)
        torch.cuda.reset_peak_memory_stats(device)
    conditioner.train()
    model.train()
    rows, draws = [], []
    successful_steps = 0
    start = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        clean = batch['image'].to(device)
        t = torch.randint(scheduler.num_train_timesteps, (len(clean),), generator=random).to(device)
        noise = torch.randn(clean.shape, generator=random).to(device)
        optimizer.zero_grad(set_to_none=True)
        before = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            condition = conditioner(batch['manufacturer_label'].to(device), batch['field_strength_label'].to(device),
                                    batch['echo_time_norm'].to(device))
            predicted = model(scheduler.q_sample(clean, t, noise), t, condition)
            loss = F.mse_loss(predicted, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite noise-prediction loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, training['grad_clip_norm'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite generator gradient')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        applied = scaler.get_scale() >= old_scale
        if applied:
            successful_steps += 1
            update_ema(ema_c, conditioner, training['ema_decay'])
            update_ema(ema_u, model, training['ema_decay'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        rows.append(dict(step=step, noise_mse=float(loss.detach()), grad_norm=float(norm),
                         optimizer_step_applied=bool(applied), seconds=time.perf_counter() - before))
        for j, index in enumerate(batch['index'].tolist()):
            draws.append(dict(step=step, batch_position=j, dataset_index=index,
                              patient_id=batch['patient_id'][j], timestep=int(t[j].item()),
                              noise_sha256=__import__('hashlib').sha256(noise[j].cpu().numpy().tobytes()).hexdigest()))
        if step == 1 or step % 8 == 0:
            print(variant, rows[-1], flush=True)
    seconds = time.perf_counter() - start
    changed_u = sum(not torch.equal(initial_u[key], value) for key, value in cpu_state(model).items())
    changed_c = sum(not torch.equal(initial_c[key], value) for key, value in cpu_state(conditioner).items())
    if not successful_steps or not changed_u or not changed_c:
        raise AssertionError('Optimisation did not update both model and conditioner')
    projection_changes = {
        name: bool(not torch.equal(initial_u[name], value.detach().cpu()))
        for name, value in model.state_dict().items() if 'time_condition_proj.weight' in name
    }
    if not all(projection_changes.values()):
        raise AssertionError('A conditioning projection received no update')
    csv_write(output / f'{variant}_steps.csv', pd.DataFrame(rows))
    csv_write(output / f'{variant}_draws.csv', pd.DataFrame(draws))
    val_rows = validate(val, indices, ema_c, ema_u, scheduler, smoke['validation_seed'])
    csv_write(output / f'{variant}_validation.csv', val_rows)
    path = checkpoint_dir / f'{variant}_smoke.pt'
    payload = dict(
        format='canonical_generator_dev_v1', variant=variant, config=config,
        coordinate_frame='canonical_axial_view_v1', steps=len(rows), successful_optimizer_steps=successful_steps,
        engineering_smoke=True, efficacy_established=False,
        conditioning_state_dict=cpu_state(ema_c), unet_state_dict=cpu_state(ema_u),
        raw_conditioning_state_dict=cpu_state(conditioner), raw_unet_state_dict=cpu_state(model),
        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
        protocol_sha256=digest(output / 'protocol.json'),
    )
    with path.open('xb') as handle:
        torch.save(payload, handle)
    # Compare exactly in the same eval mode/batch after disk round-trip.
    items = [val[index] for index in indices[:smoke['inference_patients']]]
    images = torch.stack([item['image'] for item in items]).to(device)
    mfg = torch.tensor([item['manufacturer_label'] for item in items], device=device)
    fs = torch.tensor([item['field_strength_label'] for item in items], device=device)
    echo = torch.tensor([item['echo_time_norm'] for item in items], device=device)
    ts = torch.ones(len(items), dtype=torch.long, device=device)
    with torch.inference_mode():
        before_reload = ema_u(images, ts, ema_c(mfg, fs, echo)).cpu()
    del optimizer, scaler, model, conditioner, ema_c, ema_u, initial_c, initial_u, params
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    restored_c, restored_u, restored_s, restored_ck = load_checkpoint(path, device)
    with torch.inference_mode():
        after_reload = restored_u(images, ts, restored_c(mfg, fs, echo)).cpu()
    if not torch.equal(before_reload, after_reload):
        raise AssertionError('EMA checkpoint round-trip prediction changed')
    arrays = {'source': ((images + 1) / 2).cpu().numpy(), 'source_echo_time_norm': echo.cpu().numpy()}
    for name, target_m, target_f in [('null', mfg, fs), ('manufacturer', 1 - mfg, fs), ('field_strength', mfg, 1 - fs)]:
        arrays[name] = sample_wiring(restored_c, restored_u, restored_s, images, target_m, target_f, echo,
                                     config['seed'] + 40, smoke['inference_timestep'], smoke['inference_guidance'])
        arrays[f'{name}_echo_time_norm'] = echo.cpu().numpy()
    repeat = sample_wiring(restored_c, restored_u, restored_s, images, mfg, fs, echo,
                            config['seed'] + 40, smoke['inference_timestep'], smoke['inference_guidance'])
    if not np.array_equal(arrays['null'], repeat):
        raise AssertionError('Seeded inference is not reproducible')
    with (output / f'{variant}_wiring_arrays.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    result = dict(
        variant=variant, steps=len(rows), successful_optimizer_steps=successful_steps,
        seconds=seconds, images_per_second=len(draws) / seconds,
        unet_parameters=sum(p.numel() for p in restored_u.parameters()),
        conditioner_parameters=sum(p.numel() for p in restored_c.parameters()),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        ema_validation_noise_mse=float(val_rows.noise_mse.mean()), validation_patients=len(val_rows),
        changed_model_tensors=changed_u, changed_conditioner_tensors=changed_c,
        conditioning_projections_updated=projection_changes, checkpoint=str(path.relative_to(ROOT)),
        checkpoint_sha256=digest(path), checkpoint_roundtrip_bitwise=True,
        seeded_inference_bitwise=True, source_echo_time_fixed=True,
        engineering_smoke=True, efficacy_established=False, anatomy_validated=False,
    )
    del restored_ck, restored_c, restored_u, restored_s, images, mfg, fs, echo
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def run(prepared, run_id, device='cuda'):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    prepared = project_path(prepared)
    if not json.loads((prepared / 'complete.json').read_text())['complete']:
        raise ValueError('Preparation is incomplete')
    protocol = json.loads((prepared / 'protocol.json').read_text())
    config = protocol['config']
    validate_config(config)
    inputs = json.loads((prepared / 'inputs.json').read_text())
    verify_inputs(inputs)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    train, val = [CanonicalDataset(config['view'], split) for split in ('train', 'val')]
    indices = pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int).tolist()
    output = ROOT / 'reports/v8_generator_dev' / run_id
    checkpoint_dir = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / run_id
    if output.exists() or checkpoint_dir.exists():
        raise FileExistsError('Use a new smoke run ID')
    output.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(
        prepared_protocol=str(prepared.relative_to(ROOT)), config=config,
        protocol_sha256=digest(prepared / 'protocol.json'), device=str(device),
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        scope='Capped fresh-weight engineering check only; no target-hit or anatomy validation',
    ))
    (output / 'README.md').write_text(
        '# Matched canonical generator smoke\n\n'
        'Producer: `python -m scripts.v8_generator_dev.train_smoke`.\n'
        'protocol.json identifies the immutable preparation with source snapshots. '
        'inputs.json fingerprints frozen inputs before/after the run.\n\n'
        '| File | Meaning |\n| --- | --- |\n'
        '| *_steps.csv | Actual optimisation losses/gradients/applied steps/timing |\n'
        '| *_draws.csv | Exact patient/slice order, timesteps and noise hashes |\n'
        '| *_validation.csv | Fixed one-slice-per-patient EMA noise prediction checks |\n'
        '| *_wiring_arrays.npz | Sources/nulls/both label flips at capped timestep, fixed EchoTime |\n'
        '| *_complete.json | Per-variant model/checkpoint/inference checks and measured resources |\n'
        '| complete.json | Matched draw verification and scope limits |\n\n'
        'New raw/EMA checkpoints are under checkpoints/diffusion/ax_t1_128_v8_generator_dev/. '
        'These short-run images/losses do not establish useful acquisition edits or preservation. '
        'No historic checkpoint, judge, test image, cloud service or paper is used.\n', encoding='utf-8')
    inputs[str((prepared / 'protocol.json').relative_to(ROOT)).replace('\\', '/')] = digest(prepared / 'protocol.json')
    inputs[str((prepared / 'validation_selection.csv').relative_to(ROOT)).replace('\\', '/')] = digest(prepared / 'validation_selection.csv')
    json_write(output / 'inputs.json', inputs)
    results = []
    for variant in config['variants']:
        result = run_variant(config, variant, train, val, indices, output, checkpoint_dir, device)
        json_write(output / f'{variant}_complete.json', result)
        results.append(result)
        print('COMPLETE', variant, flush=True)
    draws = [pd.read_csv(output / f'{variant}_draws.csv') for variant in config['variants']]
    pd.testing.assert_frame_equal(draws[0], draws[1])
    verify_inputs(inputs)
    json_write(output / 'complete.json', dict(
        complete=True, prepared_protocol=str(prepared.relative_to(ROOT)), variants=results,
        matched_patient_slice_order=True, matched_timesteps_and_noise=True,
        frozen_inputs_unchanged=True, test_pixels_used=False, cloud_used=False,
        pretrained_checkpoints_loaded=False, efficacy_established=False, anatomy_validated=False,
        winner_selected=False, next_gate='Separate full training and preservation/target-editing validation protocol',
    ))
    print('SMOKE COMPLETE', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    run(args.prepared, args.run_id, args.device)
