"""Bounded failed-block diagnostics using the unchanged historical training engine."""
import argparse
import copy
import gc
import hashlib
import json
import time
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path
from scripts.v8_generator_dev.precision_step import precision_step


def rng_state(device):
    return dict(cpu=torch.random.get_rng_state().clone(),
                cuda=[value.clone() for value in torch.cuda.get_rng_state_all()] if torch.device(device).type == 'cuda' else [])


def set_rng(state):
    torch.random.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def equal_state(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first.cpu(), second.cpu())
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_state(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal_state(a, b) for a, b in zip(first, second))
    return first == second


@contextmanager
def observe_clipping(observations):
    """Observe gradients before clipping; do not change tensors, RNG or return values."""
    original = torch.nn.utils.clip_grad_norm_

    def observed(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients = [p.grad for p in parameters if p.grad is not None]
        with torch.no_grad():
            norms = torch._foreach_norm(gradients, 2.0)
            total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
            if not torch.isfinite(total):
                observations.append(dict(
                    individual_gradient_tensors=len(gradients),
                    nonfinite_gradient_tensors=sum(not bool(torch.isfinite(g).all()) for g in gradients),
                    all_individual_elements_finite=all(bool(torch.isfinite(g).all()) for g in gradients)))
        return original(parameters, *args, **kwargs)

    with patch.object(torch.nn.utils, 'clip_grad_norm_', observed):
        yield


def draw(engine):
    """The historical engine's deterministic draw, without changing global RNG."""
    indices = engine.epoch_order()[engine.progress['sample_offset']:engine.progress['sample_offset'] + engine.batch_size]
    items = [engine.dataset[index] for index in indices]
    clean = torch.stack([item['image'] for item in items]).to(engine.device)
    m = torch.tensor([item['manufacturer_label'] for item in items], device=engine.device)
    f = torch.tensor([item['field_strength_label'] for item in items], device=engine.device)
    e = torch.tensor([item['echo_time_norm'] for item in items], device=engine.device)
    generator = torch.Generator().manual_seed(engine.config['seed'] + 2000000 + engine.progress['global_step'])
    ts = torch.randint(engine.scheduler.num_train_timesteps, (len(items),), generator=generator).to(engine.device)
    noise = torch.randn(clean.shape, generator=generator).to(engine.device)
    metadata = dict(global_step_before=engine.progress['global_step'], sample_indices=indices,
                    patient_ids=[item['patient_id'] for item in items], timesteps=ts.cpu().tolist(),
                    source_echo_time_norm=e.cpu().tolist(),
                    noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest())
    return (clean, m, f, e, ts, noise), metadata


def isolated_step(engine, retries=0, fp32=False):
    values, metadata = draw(engine)
    clean, m, f, e, ts, noise = values
    if fp32:
        engine.scaler = torch.amp.GradScaler(engine.device.type, enabled=False)

    def loss():
        with torch.autocast(engine.device.type, dtype=torch.float16, enabled=engine.amp and not fp32):
            prediction = engine.u(engine.scheduler.q_sample(clean, ts, noise), ts, engine.c(m, f, e))
            return F.mse_loss(prediction, noise)

    loss_value, norm, result = precision_step(loss, engine.params, engine.optimizer, engine.scaler,
        engine.device, engine.config['training']['grad_clip_norm'], retries)
    return dict(draw=metadata, loss=float(loss_value.detach()), grad_norm=float(norm), **result)


def update_state(engine):
    return copy.deepcopy(dict(c=engine.c.state_dict(), u=engine.u.state_dict(),
                              optimizer=engine.optimizer.state_dict(), rng=rng_state(engine.device)))


def snapshot_before_failure(engine, destination, before_rng, before_progress, before_scaler):
    if engine.progress != before_progress or engine.scaler.state_dict() != before_scaler:
        raise AssertionError('Failure advanced cursor or scaler; cannot call this a pre-step snapshot')
    set_rng(before_rng)
    engine.optimizer.zero_grad(set_to_none=True)
    engine.save(destination, engineering=False)


def replay_snapshot(snapshot, dataset, config, protocol_hash, output, device):
    from scripts.v8_generator_dev.full_train import TrainingEngine
    results, candidate_state, direct_state = {}, None, None

    def engine():
        return TrainingEngine(config, 'additive', dataset, protocol_hash, device, snapshot)

    control = engine()
    observations = []
    try:
        with observe_clipping(observations):
            control.train_batch()
        results['original_control'] = dict(failed=False)
    except FloatingPointError as error:
        results['original_control'] = dict(failed=True, error=str(error), gradient_observations=observations)
    del control
    gc.collect()
    accepted_scale = None
    for mode in ('retry_amp', 'direct_safe_amp', 'fp32'):
        if mode == 'direct_safe_amp' and accepted_scale is None:
            continue
        current = engine()
        if mode == 'direct_safe_amp':
            scaler = current.scaler.state_dict()
            scaler['scale'] = accepted_scale
            if results['retry_amp'].get('amp_retries', 0):
                scaler['_growth_tracker'] = 0
            current.scaler.load_state_dict(scaler)
        try:
            row = isolated_step(current, retries=8 if mode == 'retry_amp' else 0, fp32=mode == 'fp32')
            results[mode] = dict(finite=True, **row, cursor_not_advanced=True, model_not_promoted=True)
            if mode == 'retry_amp':
                accepted_scale = row['amp_scale_before'] * current.scaler.get_backoff_factor() ** row['amp_retries']
                candidate_state = update_state(current)
            elif mode == 'direct_safe_amp':
                direct_state = update_state(current)
        except FloatingPointError as error:
            results[mode] = dict(finite=False, error=str(error))
        del current
        gc.collect()
    results['retry_matches_direct_safe_update_bitwise'] = bool(
        candidate_state is not None and direct_state is not None and equal_state(candidate_state, direct_state))
    results['same_draw_for_successful_modes'] = all(
        results[mode]['draw'] == results['retry_amp']['draw'] for mode in ('direct_safe_amp', 'fp32')
        if results.get(mode, {}).get('finite') and results.get('retry_amp', {}).get('finite'))
    json_write(output / 'replay.json', results)
    return results


def run(prepared, checkpoint, run_id, wall_seconds=720):
    from scripts.v8_generator_dev.data import CanonicalDataset
    from scripts.v8_generator_dev.full_train import CachedCanonical, TrainingEngine, deterministic_runtime
    if not run_id.replace('_', '').isalnum() or not 30 <= wall_seconds <= 720:
        raise ValueError('Safe ID and bounded diagnostic duration required')
    started = time.monotonic()
    prepared, checkpoint = map(project_path, (prepared, checkpoint))
    config = json.loads((prepared / 'protocol.json').read_text())['config']
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Failed-block numerical diagnostic\n\nProducer: `python -m scripts.v8_generator_dev.numerical_replay`.\n'
        'protocol.json fixes original engine/checkpoint/runtime and diagnostic caps. steps.jsonl retains each '
        'applied replay-block draw. failure.json and pre_step.pt retain a recovered failure and its pre-step '
        'raw/EMA/optimizer/scaler/RNG/cursor state. replay.json compares original, retry, direct safe-scale '
        'and FP32 controls on that state. complete.json means diagnostics completed, never full training, '
        'new efficacy, convergence or anatomical validity. inputs.json hashes code and immutable inputs.\n', encoding='utf-8')
    deterministic_runtime()
    dataset = CachedCanonical(CanonicalDataset(config['view'], 'train'))
    protocol_hash = digest(prepared / 'protocol.json')
    engine = TrainingEngine(config, 'additive', dataset, protocol_hash, 'cuda', checkpoint)
    if engine.progress['epoch'] != 70 or engine.progress['sample_offset'] != 0:
        raise ValueError('Reproduction requires the last matched epoch-70 state')
    json_write(output / 'protocol.json', dict(prepared=str(prepared.relative_to(ROOT)),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint),
        original_engine_sha256=digest(ROOT / 'scripts/v8_generator_dev/full_train.py'),
        torch_version=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
        original_package_pins_and_gpu_class=True, original_driver_identity_unknown=True,
        maximum_epoch=75, wall_seconds=wall_seconds, diagnostic_only=True,
        full_training_completed=False, test_pixels_used=False, anatomy_validated=False))
    observations, failure = [], None
    with (output / 'steps.jsonl').open('x', encoding='utf-8') as log:
        with observe_clipping(observations):
            while engine.progress['epoch'] < 75:
                if time.monotonic() - started > wall_seconds - 90:
                    raise TimeoutError('Diagnostic cap before full failed-block replay; no negative conclusion')
                before_rng = rng_state(engine.device)
                before_progress = dict(engine.progress)
                before_scaler = copy.deepcopy(engine.scaler.state_dict())
                try:
                    row = engine.train_batch()
                except FloatingPointError as error:
                    failure = dict(error=str(error), progress_before=before_progress,
                        scaler_before=before_scaler, gradient_observations=observations,
                        draw=draw(engine)[1], historical_failing_draw_was_not_saved=True,
                        numerical_failure_recovered_from_block=True)
                    if str(error) != 'Nonfinite full-training gradient':
                        raise
                    snapshot_before_failure(engine, output / 'pre_step.pt', before_rng, before_progress, before_scaler)
                    json_write(output / 'failure.json', failure)
                    print('RECOVERED FAILURE', before_progress, flush=True)
                    break
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                if engine.progress['global_step'] % 100 == 0:
                    print('DIAGNOSTIC STEP', engine.progress, flush=True)
    del engine
    gc.collect()
    replay = replay_snapshot(output / 'pre_step.pt', dataset, config, protocol_hash, output, 'cuda') if failure else None
    hashes = {str(checkpoint.relative_to(ROOT)): digest(checkpoint),
              str((prepared / 'protocol.json').relative_to(ROOT)): protocol_hash}
    for name in ('full_train.py', 'data.py', 'model.py', 'numerical_replay.py', 'precision_step.py'):
        path = ROOT / 'scripts/v8_generator_dev' / name
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output / 'inputs.json', hashes)
    json_write(output / 'complete.json', dict(complete=True, diagnostic_completed=True,
        failure_recovered=failure is not None, replay_control_failed=replay['original_control']['failed'] if replay else False,
        retry_finite=replay.get('retry_amp', {}).get('finite', False) if replay else False,
        safe_update_matches_bitwise=replay['retry_matches_direct_safe_update_bitwise'] if replay else False,
        fp32_finite=replay.get('fp32', {}).get('finite', False) if replay else False,
        full_training_completed=False, converged=False, anatomy_validated=False,
        new_efficacy_checkpoint=False, seconds=time.monotonic() - started))
    print('NUMERICAL DIAGNOSTIC COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, default=720)
    args = parser.parse_args()
    run(args.prepared, args.checkpoint, args.run_id, args.wall_seconds)
