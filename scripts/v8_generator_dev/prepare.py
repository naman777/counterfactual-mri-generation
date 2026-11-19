"""Freeze a canonical-data R0/R1 development protocol and support profile."""
import argparse
import json
import re
import shutil

import numpy as np
import pandas as pd

from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset, sampling_probabilities, validation_indices
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def validate_config(config):
    if config['format'] != 'generator_dev_protocol_v1':
        raise ValueError('Unsupported protocol format')
    if config['data']['splits'] != ['train', 'val'] or config['data']['image_size'] != 128:
        raise ValueError('This milestone is canonical 128px train/val only')
    if config['variants'] != ['additive', 'scale_shift']:
        raise ValueError('Protocol must retain matched R0/R1 variants')
    limits = config['interpretation']
    if limits['stage'] != 'engineering_smoke_not_efficacy_training':
        raise ValueError('Full efficacy training requires a separately versioned protocol')
    if limits['echo_time_edited'] or limits['test_pixels_permitted']:
        raise ValueError('No EchoTime edits or test-pixel use permitted')
    smoke = config['smoke']
    if not 1 <= smoke['steps_per_variant'] <= 128 or not 1 <= smoke['batch_size'] <= 4:
        raise ValueError('Smoke exceeds local engineering bounds')
    if not 0 <= smoke['inference_timestep'] <= 5 or not 1 <= smoke['inference_patients'] <= 4:
        raise ValueError('Inference is a capped wiring check, not efficacy evaluation')
    if smoke['validation_slices_per_patient'] != 1:
        raise ValueError('Smoke validation uses one fixed slice per patient')
    if not 0 <= config['training']['ema_decay'] < 1:
        raise ValueError('Invalid EMA decay')
    if config['condition']['condition_dim'] != config['model']['condition_dim']:
        raise ValueError('Condition and U-Net dimensions disagree')


def prepare(config_path, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    config_path = project_path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    datasets = {split: CanonicalDataset(config['view'], split) for split in ('train', 'val')}
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new protocol run ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', {
        'run_id': run_id, 'config': config,
        'comparison': 'Same source view, sampling, shared-weight initialisation, draw order, '
                      'noise/timesteps, optimisation budget and validation sources; '
                      'R1 changes conditioning operation and projection parameter count.',
        'historical_models_loaded': False, 'test_pixels_used': False,
        'role': 'Frozen engineering protocol; does not select an efficacy winner',
    })
    (output / 'README.md').write_text(
        '# Canonical R0/R1 protocol preparation\n\n'
        'Producer: `python -m scripts.v8_generator_dev.prepare`.\n\n'
        '| File | Meaning |\n| --- | --- |\n'
        '| protocol.json | Frozen config, comparison and engineering limits |\n'
        '| inputs.json, code/ | Hashes and snapshots of source/config/manifests/arrays |\n'
        '| train_manifest.csv, val_manifest.csv | Exact canonical source selections |\n'
        '| source_checks.csv | Every included array: checksum, range and exact geometry round trip |\n'
        '| sampling_probabilities.csv | Training slice probabilities and resulting patient/cell mass |\n'
        '| cells.csv | Actual patient/slice coverage, echo/geometry ranges and sampling policy |\n'
        '| protocol_counts.csv | Recorded protocol/model/matrix categories and patient coverage |\n'
        '| support_requests.csv | Both opposite-label validation requests with fixed source EchoTime |\n'
        '| validation_selection.csv | One seeded slice per validation patient, selected before training |\n'
        '| complete.json | Coverage, hashes verified and explicit unresolved validity limits |\n\n'
        'Canonical orientation is not registration, protocol harmonisation or validated '
        'anatomy preservation. No original data, checkpoint, result or paper is rewritten.\n',
        encoding='utf-8')
    inputs = {}
    dependencies = [config_path, project_path(config['view']) / 'view.json']
    for split in ('train', 'val'):
        dependencies.extend([
            project_path(config['view']) / 'splits' / f'{split}.csv',
            ROOT / 'data/processed/ax_t1_128/splits' / f'{split}.csv',
        ])
    dependencies.extend(sorted((ROOT / 'scripts/v8_generator_dev').glob('*.py')))
    dependencies.extend(ROOT / name for name in (
        'scripts/v7_geometry/orientation.py', 'scripts/v7_geometry/support.py',
        'src/models/diffusion_unet.py', 'src/models/conditioning.py',
        'src/models/ddpm_scheduler.py',
    ))
    for path in dependencies:
        relative = path.relative_to(ROOT)
        inputs[relative.as_posix()] = digest(path)
        if path.suffix in {'.py', '.json'}:
            snapshot = output / 'code' / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)

    weights = sampling_probabilities(datasets['train'].frame, config['data']['minimum_cell_patients'])
    weighted = datasets['train'].frame[['patient_id', 'sop_uid', *['manufacturer_label', 'field_strength_label']]].copy()
    weighted['sampling_probability'] = weights
    csv_write(output / 'sampling_probabilities.csv', weighted)
    checks, cells, categories = [], [], []
    for split, dataset in datasets.items():
        csv_write(output / f'{split}_manifest.csv', dataset.frame)
        for index, row in enumerate(dataset.frame.itertuples()):
            image = dataset.image_01(index)
            path = project_path(row.source_image_path)
            inputs[path.relative_to(ROOT).as_posix()] = row.source_array_sha256
            checks.append(dict(split=split, patient_id=row.patient_id, sop_uid=row.sop_uid,
                               sha256_verified=True, exact_roundtrip=True,
                               minimum=float(image.min()), maximum=float(image.max())))
        for (mfg, fs), group in dataset.frame.groupby(['manufacturer_label', 'field_strength_label']):
            row = dict(split=split, manufacturer_label=int(mfg), field_strength_label=int(fs),
                       patients=int(group.patient_id.nunique()), slices=len(group),
                       sufficient_cell_patient_count=bool(group.patient_id.nunique() >= config['data']['minimum_cell_patients']))
            for field in ('echo_time_norm', 'echo_time', 'native_row_spacing_mm', 'native_column_spacing_mm',
                          'canonical_row_spacing_mm', 'canonical_column_spacing_mm'):
                values = pd.to_numeric(group[field], errors='coerce')
                row[f'{field}_min'] = float(values.min())
                row[f'{field}_max'] = float(values.max())
            for axis, size in [('row', 'native_rows'), ('column', 'native_columns')]:
                fov = group[size] * group[f'native_{axis}_spacing_mm']
                row[f'FOV_{axis}_mm_min'] = float(fov.min())
                row[f'FOV_{axis}_mm_max'] = float(fov.max())
            if split == 'train':
                row['training_sampling_mass'] = float(weights[group.index].sum())
                row['sampling_policy'] = 'balanced_supported_cell' if row['sufficient_cell_patient_count'] else 'retained_patient_uniform_sparse_cell'
            cells.append(row)
        for field in ('series_description', 'protocol_name', 'acquisition_matrix', 'scanner_model'):
            table = dataset.frame.copy()
            table[field] = table[field].fillna('(missing)').astype(str)
            for (mfg, fs, value), group in table.groupby(['manufacturer_label', 'field_strength_label', field]):
                categories.append(dict(split=split, manufacturer_label=int(mfg), field_strength_label=int(fs),
                                       field=field, value=value, patients=int(group.patient_id.nunique()), slices=len(group)))
        print(f'CHECKED {split}: {len(dataset)} canonical arrays', flush=True)
    csv_write(output / 'source_checks.csv', pd.DataFrame(checks))
    csv_write(output / 'cells.csv', pd.DataFrame(cells))
    csv_write(output / 'protocol_counts.csv', pd.DataFrame(categories))
    selection = validation_indices(datasets['val'].frame, config['smoke']['validation_seed'])
    selected = datasets['val'].frame.loc[selection].copy()
    selected.insert(0, 'dataset_index', selection)
    csv_write(output / 'validation_selection.csv', selected)
    requests = []
    for row in datasets['val'].frame.itertuples():
        for axis in ('manufacturer', 'field_strength'):
            mfg, fs = int(row.manufacturer_label), int(row.field_strength_label)
            mfg, fs = (1 - mfg, fs) if axis == 'manufacturer' else (mfg, 1 - fs)
            requests.append(dict(patient_id=row.patient_id, sop_uid=row.sop_uid, axis=axis,
                                 target_manufacturer=mfg, target_field_strength=fs,
                                 source_echo_time_norm=float(row.echo_time_norm),
                                 target_echo_time_norm=float(row.echo_time_norm),
                                 **request_support(datasets['train'].frame, mfg, fs, float(row.echo_time_norm),
                                                   config['data']['minimum_cell_patients'])))
    requests = pd.DataFrame(requests)
    csv_write(output / 'support_requests.csv', requests)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, train_images=len(datasets['train']), validation_images=len(datasets['val']),
        train_patients=int(datasets['train'].frame.patient_id.nunique()),
        validation_patients=int(datasets['val'].frame.patient_id.nunique()),
        source_arrays_verified=len(checks), test_pixels_used=False, historical_models_loaded=False,
        support_counts=requests.groupby(['axis', 'condition_support_status']).size().rename('requests').reset_index().to_dict('records'),
        cell_balance_is_not_protocol_balance=True, scanner_invariance_established=False,
        anatomy_validated=False, efficacy_established=False,
    ))
    print('PREPARED', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/generator_dev_v1.json')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    prepare(args.config, args.run_id)
