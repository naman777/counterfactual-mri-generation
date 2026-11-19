"""CPU-only exploratory canonical guide audit; no generator/adapter training."""
import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import rankdata

from scripts.v6_structure.model import structure_maps
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def guides(image):
    image = np.asarray(image)
    if image.shape != (128, 128) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Expected finite canonical 128px image in [0,1]')
    tensor = torch.tensor(image.copy(), dtype=torch.float32)[None, None]
    ranks = ((rankdata(image.ravel(), method='average') - 1) / (image.size - 1)).reshape(image.shape)
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)[None, None]
    with torch.no_grad():
        gradients = structure_maps(tensor)[0].numpy()
        rank_gradients = structure_maps(rank_tensor)[0].numpy()
    distances, valid = [], []
    for threshold in (0.03, 0.05, 0.1):
        mask = ndi.gaussian_filter(image, 1, mode='reflect') > threshold
        valid.append(bool(mask.any() and not mask.all()))
        value = (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)) / np.hypot(*image.shape)
        if not valid[-1]:
            value = np.zeros_like(value)
        distances.append(value)
    distance = F.interpolate(torch.tensor(np.stack(distances), dtype=torch.float32)[None],
                             size=(32, 32), mode='area')[0].numpy()
    return {'canonical_gradients': gradients, 'canonical_rank_gradients': rank_gradients,
            'canonical_proxy_distance': distance}, valid


def balanced_score(y, prediction):
    if any(not np.any(y == label) for label in (0, 1)):
        raise ValueError('Both classes are required for balanced accuracy')
    return float(np.mean([(prediction[y == label] == label).mean() for label in (0, 1)]))


def confidence(y, prediction, seed=731, n=10000):
    rng = np.random.default_rng(seed)
    classes = [np.where(y == label)[0] for label in (0, 1)]
    values = []
    for _ in range(n):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        values.append(balanced_score(y[indices], prediction[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid guide audit ID')
    torch.set_num_threads(4)
    view = 'data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new guide-audit ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(
        view=view, selection_seed=731, train_validation_only=True, source_slices_per_patient=1,
        representations=['canonical_gradients', 'canonical_rank_gradients', 'canonical_proxy_distance'],
        feature_dimensions=192, probe='Class-balanced ridge alpha=10; train-only standardisation; no tuning',
        perturbations=['positive gamma 0.7', 'positive gamma 1.3', 'roll 8 pixels on rows and columns'],
        role='Exploratory development audit on previously used patients; not a new holdout or clinical anatomy test',
        cautions='Rank/threshold distance are nondifferentiable offline guides, not ready replacements for a differentiable preservation loss. Threshold maps are contrast-sensitive proxies. No weak-probe result proves scanner invariance.',
    ))
    inputs = {}
    inputs[(project_path(view) / 'view.json').relative_to(ROOT).as_posix()] = digest(project_path(view) / 'view.json')
    for path in (ROOT / 'scripts/v8_generator_dev/guide_audit.py', ROOT / 'scripts/v6_structure/model.py',
                 ROOT / 'scripts/v5_anatomy/anchored_sampler.py', ROOT / 'scripts/v8_generator_dev/data.py',
                 ROOT / 'scripts/v8_generator_dev/io.py',
                 ROOT / 'scripts/v7_geometry/orientation.py'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    rows, sensitivities, features, maps = [], [], {}, {}
    for split in ('train', 'val'):
        dataset = CanonicalDataset(view, split)
        manifest = project_path(view) / 'splits' / f'{split}.csv'
        original = ROOT / 'data/processed/ax_t1_128/splits' / f'{split}.csv'
        inputs[manifest.relative_to(ROOT).as_posix()] = digest(manifest)
        inputs[original.relative_to(ROOT).as_posix()] = digest(original)
        chosen = dataset.frame.sample(frac=1, random_state=731).drop_duplicates('patient_id')
        for index, row in chosen.iterrows():
            image = dataset.image_01(index)
            path = project_path(row.source_image_path)
            inputs[path.relative_to(ROOT).as_posix()] = row.source_array_sha256
            values, valid = guides(image)
            photos = [guides(image.astype(np.float64) ** gamma)[0] for gamma in (0.7, 1.3)]
            moved = guides(np.roll(image, (8, 8), (0, 1)))[0]
            for name, value in values.items():
                features.setdefault(name, []).append(F.adaptive_avg_pool2d(torch.tensor(value)[None], (8, 8)).flatten().numpy())
                maps.setdefault(name, []).append(value)
                sensitivities.append(dict(split=split, patient_id=row.patient_id, representation=name,
                    gamma07_map_l1=float(np.abs(value - photos[0][name]).mean()),
                    gamma13_map_l1=float(np.abs(value - photos[1][name]).mean()),
                    translation8_map_l1=float(np.abs(value - moved[name]).mean()),
                    proxy_channels_valid=all(valid) if name == 'canonical_proxy_distance' else True,
                    anatomy_validated=False))
            rows.append(dict(split=split, patient_id=row.patient_id, sop_uid=row.sop_uid,
                             dataset_index=int(index), manufacturer=int(row.manufacturer_label),
                             field_strength=int(row.field_strength_label),
                             proxy_distance_channels_valid=all(valid),
                             source_image_path=row.source_image_path))
    frame = pd.DataFrame(rows)
    train = frame.split.eq('train').to_numpy()
    val = ~train
    probes, predictions = [], []
    stacked = {name: np.stack(values) for name, values in features.items()}
    for name, values in stacked.items():
        mean, std = values[train].mean(0), values[train].std(0).clip(1e-4)
        x = np.column_stack([np.ones(len(values)), (values - mean) / std]).astype(np.float64)
        for axis in ('manufacturer', 'field_strength'):
            y = frame[axis].to_numpy()
            counts = np.bincount(y[train], minlength=2)
            weights = 1 / counts[y[train]]
            weights /= weights.mean()
            penalty = np.eye(x.shape[1]) * 10
            penalty[0, 0] = 0
            coefficient = np.linalg.solve(x[train].T @ (weights[:, None] * x[train]) + penalty,
                                          x[train].T @ (weights * y[train]))
            score = x[val] @ coefficient
            pred = (score >= 0.5).astype(int)
            low, high = confidence(y[val], pred)
            probes.append(dict(representation=name, axis=axis, balanced_accuracy=balanced_score(y[val], pred),
                               ci95_lo=low, ci95_hi=high, train_patients=int(train.sum()),
                               validation_patients=int(val.sum()), scanner_invariance_established=False,
                               interval='10000 class-stratified patient bootstrap; boundary intervals can degenerate'))
            predictions.extend(dict(representation=name, axis=axis, patient_id=pid, truth=int(t), prediction=int(p), score=float(s))
                               for pid, t, p, s in zip(frame[val].patient_id, y[val], pred, score))
    csv_write(output / 'patient_selection.csv', frame)
    csv_write(output / 'sensitivities.csv', pd.DataFrame(sensitivities))
    csv_write(output / 'leakage_probe.csv', pd.DataFrame(probes))
    csv_write(output / 'probe_predictions.csv', pd.DataFrame(predictions))
    with (output / 'features.npz').open('xb') as handle:
        np.savez_compressed(handle, **stacked)
    with (output / 'guide_maps.npz').open('xb') as handle:
        np.savez_compressed(handle, **{name: np.stack(values) for name, values in maps.items()})
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(
        complete=True, source_patients=len(frame), train_patients=int(train.sum()), validation_patients=int(val.sum()),
        same_feature_budget=True, source_inputs_unchanged=True, test_pixels_used=False,
        gpu_used=False, generator_trained=False, adapter_trained=False,
        scanner_invariance_established=False, anatomy_validated=False, winner_selected=False,
    ))
    (output / 'README.md').write_text(
        '# Exploratory canonical guide audit\n\nProducer: `python -m scripts.v8_generator_dev.guide_audit`.\n'
        'protocol.json fixes representation/probe/perturbation rules. patient_selection.csv fixes train/val sources. '
        'sensitivities.csv records photometric and spatial responses; leakage_probe.csv and probe_predictions.csv '
        'record fixed patient-separated label probes with bootstrap CIs. features.npz and guide_maps.npz retain '
        'all descriptors/maps. inputs.json fingerprints unchanged sources/code; complete.json limits interpretation. '
        'Rank/distance guides are nondifferentiable and not clinical anatomy labels or approved preservation losses. '
        'No generator/adapter is trained, no test image is used, and no representation is declared invariant.\n', encoding='utf-8')
    print(pd.DataFrame(probes).to_string(index=False), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(audit(args.run_id))
