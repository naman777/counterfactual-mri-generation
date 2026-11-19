"""Resumable canonical training engine and sequential paired training controller.

Full GPU work is intended for a single Modal app. --engineering-steps runs a
bounded local integration check of the same engine, not full efficacy training.
"""
import argparse
import gc
import hashlib
import json
import math
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.v8_generator_dev.data import CanonicalDataset, sampling_probabilities
from scripts.v8_generator_dev.full_protocol import plateau, validate_full_config
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.model import build_models, clone_ema, cpu_state, update_ema
from scripts.v8_generator_dev.precision_step import precision_step


def deterministic_runtime():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class CachedCanonical:
    """Verify/load canonical pixels once into host RAM, never GPU or test data.

    This avoids repeated remote-volume opens for every epoch. Source hashes are
    still verified before and after the segment by the frozen input ledger.
    """
    def __init__(self, dataset):
        if dataset.split not in ('train', 'val'):
            raise ValueError('Cache is restricted to train/validation')
        self.split, self.frame = dataset.split, dataset.frame
        self.items = [dataset[index] for index in range(len(dataset))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class TrainingEngine:
    def __init__(self, config, variant, dataset, protocol_sha256, device='cpu', resume=None, batch_size_override=None):
        validate_full_config(config)
        if dataset.split != 'train':
            raise ValueError('Training engine requires the training split')
        self.config, self.variant, self.dataset = config, variant, dataset
        self.protocol_sha256, self.device = protocol_sha256, torch.device(device)
        self.started = time.perf_counter()
        self.prior_seconds = 0.0
        torch.manual_seed(config['seed'])
        self.c, self.u, self.scheduler = build_models(config, variant, self.device)
        self.ec, self.eu = clone_ema(self.c), clone_ema(self.u)
        self.params = list(self.c.parameters()) + list(self.u.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=config['training']['learning_rate'],
                                           weight_decay=config['training']['weight_decay'])
        self.amp = bool(config['training']['mixed_precision'] and self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.weights = torch.from_numpy(sampling_probabilities(dataset.frame, config['data']['minimum_cell_patients']))
        self.batch_size = config['full_training']['batch_size'] if batch_size_override is None else batch_size_override
        if not 1 <= self.batch_size <= config['full_training']['batch_size']:
            raise ValueError('Invalid effective batch size')
        self.progress = dict(epoch=0, sample_offset=0, global_step=0, applied_steps=0)
        self.history = []
        self.data_identity = hashlib.sha256(dataset.frame.to_csv(index=False).encode()).hexdigest()
        self._order_epoch, self._order = None, None
        # Construction differs between architectures; training dropout starts
        # from the same stream after construction and has an exact saved state.
        torch.manual_seed(config['seed'] + 30)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config['seed'] + 30)
        if resume is not None:
            self.restore(resume)

    @property
    def wall_seconds(self):
        return self.prior_seconds + time.perf_counter() - self.started

    def epoch_order(self):
        epoch = self.progress['epoch']
        if self._order_epoch != epoch:
            generator = torch.Generator().manual_seed(self.config['seed'] + epoch)
            self._order = torch.multinomial(self.weights, len(self.dataset), replacement=True,
                                             generator=generator).tolist()
            self._order_epoch = epoch
        return self._order

    def train_batch(self):
        if self.progress['epoch'] >= self.config['full_training']['max_epochs']:
            raise StopIteration('Maximum training epoch reached')
        if self.wall_seconds >= self.config['resources']['max_variant_wall_seconds']:
            raise TimeoutError('Variant wall-time ceiling reached; training is incomplete')
        self.c.train()
        self.u.train()
        offset = self.progress['sample_offset']
        indices = self.epoch_order()[offset:offset + self.batch_size]
        items = [self.dataset[index] for index in indices]
        clean = torch.stack([item['image'] for item in items]).to(self.device)
        m = torch.tensor([item['manufacturer_label'] for item in items], device=self.device)
        f = torch.tensor([item['field_strength_label'] for item in items], device=self.device)
        e = torch.tensor([item['echo_time_norm'] for item in items], device=self.device)
        generator = torch.Generator().manual_seed(self.config['seed'] + 2000000 + self.progress['global_step'])
        ts = torch.randint(self.scheduler.num_train_timesteps, (len(items),), generator=generator).to(self.device)
        noise = torch.randn(clean.shape, generator=generator).to(self.device)
        before = time.perf_counter()
        def forward_loss():
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                prediction = self.u(self.scheduler.q_sample(clean, ts, noise), ts, self.c(m, f, e))
                return F.mse_loss(prediction, noise)
        loss, norm, precision = precision_step(
            forward_loss, self.params, self.optimizer, self.scaler, self.device,
            self.config['training']['grad_clip_norm'], self.config['training'].get('amp_gradient_retries', 0))
        applied = precision['applied']
        if applied:
            self.progress['applied_steps'] += 1
            update_ema(self.ec, self.c, self.config['training']['ema_decay'])
            update_ema(self.eu, self.u, self.config['training']['ema_decay'])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.progress['global_step'] += 1
        self.progress['sample_offset'] += len(indices)
        epoch_at_draw = self.progress['epoch']
        if self.progress['sample_offset'] == len(self.dataset):
            self.progress['epoch'] += 1
            self.progress['sample_offset'] = 0
        return dict(global_step=self.progress['global_step'], epoch_at_draw=epoch_at_draw,
                    sample_indices=indices, patient_ids=[item['patient_id'] for item in items],
                    timesteps=ts.cpu().tolist(),
                    noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    noise_mse=float(loss.detach()), grad_norm=float(norm), **precision,
                    seconds=time.perf_counter() - before)

    @torch.inference_mode()
    def validate(self, dataset, indices=None):
        if dataset.split != 'val':
            raise ValueError('Noise validation requires the validation split')
        self.ec.eval()
        self.eu.eval()
        rows = []
        for index in range(len(dataset)) if indices is None else indices:
            if self.wall_seconds >= self.config['resources']['max_variant_wall_seconds']:
                raise TimeoutError('Wall-time ceiling reached during validation')
            item = dataset[index]
            generator = torch.Generator().manual_seed(self.config['evaluation']['selection_seed'] + index)
            ts = torch.randint(self.scheduler.num_train_timesteps, (1,), generator=generator).to(self.device)
            noise = torch.randn(item['image'].unsqueeze(0).shape, generator=generator).to(self.device)
            clean = item['image'].unsqueeze(0).to(self.device)
            condition = self.ec(torch.tensor([item['manufacturer_label']], device=self.device),
                                torch.tensor([item['field_strength_label']], device=self.device),
                                torch.tensor([item['echo_time_norm']], device=self.device))
            predicted = self.eu(self.scheduler.q_sample(clean, ts, noise), ts, condition)
            loss = float(F.mse_loss(predicted, noise))
            if not math.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            rows.append(dict(dataset_index=int(index), patient_id=item['patient_id'], noise_mse=loss,
                             manufacturer_label=item['manufacturer_label'], field_strength_label=item['field_strength_label']))
        table = pd.DataFrame(rows)
        record = dict(epoch=self.progress['epoch'], sample_offset=self.progress['sample_offset'],
                      natural_slice_noise_mse=float(table.noise_mse.mean()),
                      patient_macro_noise_mse=float(table.groupby('patient_id').noise_mse.mean().mean()),
                      validation_slices=len(table), validation_patients=int(table.patient_id.nunique()))
        return table, record

    def save(self, path, engineering=False):
        payload = dict(
            format='canonical_generator_full_v1', config=self.config, variant=self.variant,
            protocol_sha256=self.protocol_sha256, data_identity=self.data_identity,
            coordinate_frame='canonical_axial_view_v1', engineering=engineering,
            device_type=self.device.type, torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
            effective_batch_size=self.batch_size,
            progress=dict(self.progress), history=list(self.history), wall_seconds=self.wall_seconds,
            conditioning_state_dict=cpu_state(self.ec), unet_state_dict=cpu_state(self.eu),
            raw_conditioning_state_dict=cpu_state(self.c), raw_unet_state_dict=cpu_state(self.u),
            optimizer_state_dict=self.optimizer.state_dict(), scaler_state_dict=self.scaler.state_dict(),
            rng_torch=torch.random.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            efficacy_established=False, anatomy_validated=False,
        )
        with project_path(path).open('xb') as handle:
            torch.save(payload, handle)

    def restore(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('format') != 'canonical_generator_full_v1' or ck.get('coordinate_frame') != 'canonical_axial_view_v1':
            raise ValueError('Not a resumable canonical full-training checkpoint')
        for key, expected in [('config', self.config), ('variant', self.variant),
                               ('protocol_sha256', self.protocol_sha256), ('data_identity', self.data_identity),
                               ('device_type', self.device.type), ('torch_version', str(torch.__version__)),
                               ('effective_batch_size', self.batch_size),
                               ('cpu_threads', torch.get_num_threads())]:
            if ck[key] != expected:
                raise ValueError(f'Resume incompatibility: {key}')
        self.c.load_state_dict(ck['raw_conditioning_state_dict'])
        self.u.load_state_dict(ck['raw_unet_state_dict'])
        self.ec.load_state_dict(ck['conditioning_state_dict'])
        self.eu.load_state_dict(ck['unet_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.scaler.load_state_dict(ck['scaler_state_dict'])
        self.progress, self.history = dict(ck['progress']), list(ck['history'])
        self.prior_seconds, self.started = float(ck['wall_seconds']), time.perf_counter()
        torch.random.set_rng_state(ck['rng_torch'])
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['rng_cuda'])


def load_ema(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('format') != 'canonical_generator_full_v1' or ck.get('coordinate_frame') != 'canonical_axial_view_v1':
        raise ValueError('Expected canonical full-training checkpoint')
    c, u, scheduler = build_models(ck['config'], ck['variant'], device)
    c.load_state_dict(ck['conditioning_state_dict'])
    u.load_state_dict(ck['unet_state_dict'])
    return c.eval(), u.eval(), scheduler.eval(), ck


def run_pair(prepared, run_id, device='cuda', engineering_steps=None, resume_pair=None):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    if engineering_steps is not None and not 1 <= engineering_steps <= 32:
        raise ValueError('Local integration cap is 1-32 batches per variant')
    prepared = project_path(prepared)
    protocol = json.loads((prepared / 'protocol.json').read_text())
    config = protocol['config']
    validate_full_config(config)
    inputs = json.loads((prepared / 'inputs.json').read_text())
    verify_inputs(inputs)
    deterministic_runtime()
    train, val = [CachedCanonical(CanonicalDataset(config['view'], split)) for split in ('train', 'val')]
    selection = pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int).tolist()
    output = ROOT / 'reports/v8_generator_dev' / run_id
    ck_dir = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / run_id
    if output.exists() or ck_dir.exists():
        raise FileExistsError('Use a new training segment ID')
    if engineering_steps is None and not config['resources']['monetary_budget_authorised']:
        # No budget is inferred from an engineering command or old cloud allowance.
        raise ValueError('Full execution needs a newly frozen resource-approved protocol; use --engineering-steps for local verification')
    output.mkdir(parents=True)
    ck_dir.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(prepared=str(prepared.relative_to(ROOT)),
                                              prepared_sha256=digest(prepared / 'protocol.json'),
                                              config=config, engineering_steps=engineering_steps,
                                              device=device, resume_pair=str(resume_pair) if resume_pair else None))
    (output / 'README.md').write_text(
        '# Resumable paired training segment\n\nProducer: `python -m scripts.v8_generator_dev.full_train`.\n'
        'protocol.json freezes inputs, execution scope and parent/resume identity. '
        'block_*/*_steps.json stores actual sampled batches/noise hashes/losses. '
        'block_*/*_validation.csv and *_validation.json store EMA validation observations. '
        'block_*/complete.json maps immutable raw/EMA/resume checkpoints. '
        'complete.json records common epoch, stop reason, checkpoints and selection status; '
        'resource_stop.json marks an incomplete budget/time stop. '
        'inputs.json records source hashes. No clinical preservation or fresh-test claim.\n', encoding='utf-8')
    inputs[(prepared / 'protocol.json').relative_to(ROOT).as_posix()] = digest(prepared / 'protocol.json')
    prior = json.loads(project_path(resume_pair).read_text()) if resume_pair else None
    checkpoints = dict(prior['checkpoints']) if prior else {}
    if prior and prior['prepared_sha256'] != digest(prepared / 'protocol.json'):
        raise ValueError('Resume pair belongs to a different frozen protocol')
    engineering = engineering_steps is not None
    if prior and prior['engineering'] != engineering:
        raise ValueError('Engineering checkpoints cannot be promoted into a full run')
    if prior:
        inputs[project_path(resume_pair).relative_to(ROOT).as_posix()] = digest(project_path(resume_pair))
        for variant, path in checkpoints.items():
            path = project_path(path)
            if digest(path) != prior['checkpoint_hashes'][variant]:
                raise ValueError('Resume checkpoint changed')
            inputs[path.relative_to(ROOT).as_posix()] = prior['checkpoint_hashes'][variant]
    json_write(output / 'inputs.json', inputs)
    parent_checkpoint_bytes = int(prior.get('checkpoint_bytes', 0)) if prior else 0
    block_number, all_rows = 0, {}
    while True:
        block_number += 1
        block = output / f'block_{block_number:04d}'
        block.mkdir()
        histories, progress, draws = {}, {}, {}
        for variant in config['variants']:
            engine = TrainingEngine(config, variant, train, digest(prepared / 'protocol.json'), device,
                                    checkpoints.get(variant),
                                    config['resources']['local_engineering_batch_size'] if engineering else None)
            current_epoch = engine.progress['epoch']
            interval = config['full_training']['validation_every_epochs']
            until_epoch = min(config['full_training']['max_epochs'], (current_epoch // interval + 1) * interval)
            records = []
            try:
                while (len(records) < engineering_steps if engineering else engine.progress['epoch'] < until_epoch):
                    records.append(engine.train_batch())
                    if len(records) == 1 or len(records) % 100 == 0:
                        print(variant, 'step', engine.progress['global_step'], 'epoch', engine.progress['epoch'], flush=True)
                table, record = engine.validate(val, selection if engineering else None)
                if not engineering:
                    engine.history.append(dict(record, checkpoint=str(
                        (ck_dir / f'{variant}_block_{block_number:04d}.pt').relative_to(ROOT))))
            except TimeoutError as error:
                emergency = ck_dir / f'{variant}_resource_stop_{block_number:04d}.pt'
                engine.save(emergency, engineering)
                json_write(output / 'resource_stop.json', dict(complete=False, reason=str(error), variant=variant,
                                                               checkpoint=str(emergency.relative_to(ROOT)),
                                                               other_checkpoints=checkpoints, efficacy_established=False))
                return output
            csv_write(block / f'{variant}_validation.csv', table)
            json_write(block / f'{variant}_validation.json', record)
            json_write(block / f'{variant}_steps.json', records)
            next_checkpoint = ck_dir / f'{variant}_block_{block_number:04d}.pt'
            existing_bytes = parent_checkpoint_bytes + sum(p.stat().st_size for p in ck_dir.glob('*.pt'))
            anticipated = sum(p.numel() * p.element_size() for p in engine.u.parameters()) * 5 + 16 * 1024 ** 2
            if existing_bytes + anticipated > config['resources']['max_pair_checkpoint_bytes']:
                json_write(output / 'resource_stop.json', dict(complete=False, reason='checkpoint_storage_ceiling', efficacy_established=False))
                return output
            engine.save(next_checkpoint, engineering)
            checkpoints[variant] = str(next_checkpoint.relative_to(ROOT))
            histories[variant] = list(engine.history)
            progress[variant] = dict(engine.progress)
            draws[variant] = [{key: row[key] for key in ('global_step', 'epoch_at_draw', 'sample_indices', 'patient_ids', 'timesteps', 'noise_sha256')}
                              for row in records]
            all_rows.setdefault(variant, []).append(dict(record, checkpoint=checkpoints[variant]))
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if draws['additive'] != draws['scale_shift'] or progress['additive'] != progress['scale_shift']:
            raise AssertionError('Paired training budgets/data/noise/updates diverged')
        json_write(block / 'complete.json', dict(checkpoints=checkpoints, progress=progress,
                                                 prepared_sha256=digest(prepared / 'protocol.json'), engineering=engineering,
                                                 checkpoint_hashes={v: digest(project_path(p)) for v, p in checkpoints.items()},
                                                 checkpoint_bytes=parent_checkpoint_bytes + sum(p.stat().st_size for p in ck_dir.glob('*.pt')),
                                                 matched_sources_timesteps_noise=True))
        epoch = progress['additive']['epoch']
        reason = ('engineering_segment_complete' if engineering else
                  'paired_validation_plateau' if all(plateau(histories[v], config) for v in config['variants']) else
                  'maximum_epoch' if epoch >= config['full_training']['max_epochs'] else None)
        if reason:
            break
    selection = {}
    if not engineering:
        # Earliest minimum loss is secondary; primary checkpoints remain common-epoch.
        for variant, history in histories.items():
            best = min(history, key=lambda row: (row['patient_macro_noise_mse'], row['epoch']))
            selection[variant] = dict(epoch=best['epoch'], checkpoint=best['checkpoint'], role='secondary_loss_selected')
    verify_inputs(inputs)
    json_write(output / 'complete.json', dict(
        complete=True, engineering=engineering, prepared_sha256=digest(prepared / 'protocol.json'),
        checkpoints=checkpoints, progress=progress, stop_reason=reason,
        checkpoint_hashes={v: digest(project_path(p)) for v, p in checkpoints.items()},
        checkpoint_bytes=parent_checkpoint_bytes + sum(p.stat().st_size for p in ck_dir.glob('*.pt')),
        secondary_checkpoint_selection=selection, primary_comparison='common final epoch',
        matched_sources_timesteps_noise=True, test_pixels_used=False,
        efficacy_established=False, anatomy_validated=False,
    ))
    print('PAIRED TRAINING SEGMENT COMPLETE', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--engineering-steps', type=int)
    parser.add_argument('--resume-pair')
    args = parser.parse_args()
    run_pair(args.prepared, args.run_id, args.device, args.engineering_steps, args.resume_pair)
