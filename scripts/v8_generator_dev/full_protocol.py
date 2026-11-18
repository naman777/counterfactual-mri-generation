"""Frozen full-training rules, validation selection and explicit resource ceilings."""
import copy
import itertools
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, project_path


def read_config(path, root=ROOT):
    overrides = json.loads(project_path(path, root).read_text(encoding='utf-8'))
    config = json.loads(project_path(overrides['base_config'], root).read_text(encoding='utf-8'))
    config = copy.deepcopy(config)
    config.pop('smoke')
    config.update(overrides)
    validate_full_config(config)
    return config


def validate_full_config(config):
    retries = config['training'].get('amp_gradient_retries', 0)
    if type(retries) is not int or not 0 <= retries <= 16:
        raise ValueError('AMP gradient retries must be an integer between zero and sixteen')
    if config['format'] != 'generator_full_protocol_v1' or config['variants'] != ['additive', 'scale_shift']:
        raise ValueError('Full protocol requires the matched additive/scale-shift pair')
    if config['data']['splits'] != ['train', 'val'] or config['data']['image_size'] != 128:
        raise ValueError('Full development is restricted to canonical 128px train/val')
    if config['interpretation']['echo_time_edited'] or config['interpretation']['test_pixels_permitted']:
        raise ValueError('No EchoTime edits or test-pixel use')
    training = config['full_training']
    if not 1 <= training['batch_size'] <= 32 or not 1 <= training['max_epochs'] <= 200:
        raise ValueError('Training exceeds frozen batch/epoch ceilings')
    if training['validation_every_epochs'] < 1 or training['plateau_checks'] < 1:
        raise ValueError('Positive validation interval and patience required')
    if not 1 <= training['minimum_stop_epoch'] <= training['max_epochs']:
        raise ValueError('Invalid minimum stopping epoch')
    if not 0 <= training['minimum_relative_improvement'] < 1:
        raise ValueError('Invalid relative improvement threshold')
    if config['resources']['max_gpus'] != 1 or config['resources']['max_containers'] != 1:
        raise ValueError('This paired experiment is sequential on one GPU')
    if config['resources']['max_variant_wall_seconds'] <= 0 or config['resources']['max_pair_checkpoint_bytes'] <= 0:
        raise ValueError('Positive time/storage caps required')
    evaluation = config['evaluation']
    if evaluation['source_slices_per_patient'] != 1 or evaluation['minimum_eligible_patients_per_direction'] < 1:
        raise ValueError('Selection requires patient-separated sources and direction coverage')
    if set(evaluation['selection_support_statuses']) - {'observed_condition', 'interpolation_only'}:
        raise ValueError('Outside-range/missing/sparse requests cannot select operating points')
    if len(evaluation['judge_paths']) != 3 or len(set(evaluation['judge_paths'])) != 3:
        raise ValueError('Retain all three distinct frozen judges')
    seeds = [evaluation['selection_generation_seed'], *evaluation['confirmation_generation_seeds']]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Selection and confirmation seeds must be distinct')
    if not 0 <= evaluation['other_axis_stability_floor'] <= 1:
        raise ValueError('Invalid stability floor')
    for axis in ('manufacturer', 'field_strength'):
        if not 0 <= evaluation['ssim_floor'][axis] <= 1:
            raise ValueError('Invalid correspondence floor')
        for setting in settings(config, axis):
            if not 0 <= setting['timestep'] < config['diffusion']['num_train_timesteps'] or setting['guidance'] < 0:
                raise ValueError('Invalid editing setting')


def settings(config, axis):
    grid = config['evaluation']['selection_grid'][axis]
    candidates = [dict(timestep=int(t), guidance=float(w)) for t, w in itertools.product(grid['timesteps'], grid['guidance'])]
    fixed = config['evaluation']['fixed_settings'][axis]
    if fixed not in candidates:
        candidates.append(dict(fixed))
    return sorted(candidates, key=lambda setting: (setting['timestep'], setting['guidance']))


def plateau(history, config):
    """Patience counts validation checks since the last material improvement."""
    rules = config['full_training']
    if not history or history[-1]['epoch'] < rules['minimum_stop_epoch']:
        return False
    best, stale = float('inf'), 0
    for row in history:
        value = float(row['patient_macro_noise_mse'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation history')
        if value < best * (1 - rules['minimum_relative_improvement']):
            best, stale = value, 0
        else:
            stale += 1
    return stale >= rules['plateau_checks']


def select_setting(rows, axis, config):
    """Select only on declared validation sources/seed/support; no fallback win."""
    rules = config['evaluation']
    table = rows[(rows.axis == axis) & (rows.generation_seed == rules['selection_generation_seed']) &
                 (rows.variant == 'target')].copy()
    if table.empty:
        return dict(selected=None, reason='no_selection_rows', candidates=[])
    candidates = []
    for (timestep, guidance), group in table.groupby(['timestep', 'guidance']):
        eligible = group[group.condition_support_status.isin(rules['selection_support_statuses'])]
        counts = {direction: int(eligible.loc[eligible.direction == direction, 'patient_id'].nunique())
                  for direction in ('0_to_1', '1_to_0')}
        covered = all(n >= rules['minimum_eligible_patients_per_direction'] for n in counts.values())
        if eligible.empty:
            ssim, stability, delta = None, None, None
        else:
            # Average seeds/judges within each patient before patient averaging.
            per_patient = eligible.groupby('patient_id')[['ssim', 'other_axis_stable', 'paired_hit_delta']].mean()
            ssim, stability, delta = map(float, per_patient.mean().tolist())
        feasible = bool(covered and ssim is not None and ssim >= rules['ssim_floor'][axis] and
                        stability >= rules['other_axis_stability_floor'] and delta > 0)
        candidates.append(dict(timestep=int(timestep), guidance=float(guidance),
                               eligible_patients_by_direction=counts, patient_macro_ssim=ssim,
                               mean_judge_other_axis_stability=stability, mean_judge_paired_hit_delta=delta,
                               feasible=feasible))
    feasible = [row for row in candidates if row['feasible']]
    if not feasible:
        return dict(selected=None, reason='no_supported_positive_candidate_meets_provisional_constraints', candidates=candidates)
    chosen = min(feasible, key=lambda row: (-row['mean_judge_paired_hit_delta'], row['timestep'], row['guidance']))
    return dict(selected=dict(timestep=chosen['timestep'], guidance=chosen['guidance']),
                reason='provisional_validation_selection_not_anatomy_approval', candidates=candidates)


def bootstrap_patients(frame, metric, seed, resamples=10000):
    """Equal-patient estimate, averaging repeated generation seeds within patient."""
    values = frame.groupby('patient_id')[metric].mean().dropna().to_numpy(float)
    if not len(values):
        return dict(patients=0, mean=None, ci95_lo=None, ci95_hi=None)
    if not np.isfinite(values).all() or resamples < 1:
        raise ValueError('Invalid bootstrap data/settings')
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(0, len(values), (min(1000, resamples - i), len(values)))].mean(1)
                            for i in range(0, resamples, 1000)])
    return dict(patients=len(values), mean=float(values.mean()),
                ci95_lo=float(np.quantile(means, 0.025)), ci95_hi=float(np.quantile(means, 0.975)))


def workload(config, validation_patients):
    """Exact planned model calls per image, not a cloud-time or price estimate."""
    selection = 0
    confirmation = 0
    for axis in ('manufacturer', 'field_strength'):
        selection += sum(4 * (row['timestep'] + 1) for row in settings(config, axis)) * validation_patients
        # Fixed + possibly different selected point; duplicates are an upper bound.
        largest = max(row['timestep'] for row in settings(config, axis))
        confirmation += 2 * 4 * (largest + 1) * validation_patients * len(config['evaluation']['confirmation_generation_seeds'])
    return dict(selection_unet_image_calls_per_variant=selection,
                confirmation_unet_image_calls_upper_bound_per_variant=confirmation,
                meaning='Two CFG arms times target/null trajectories; excludes identity/judge calls and training. No price inferred.')
