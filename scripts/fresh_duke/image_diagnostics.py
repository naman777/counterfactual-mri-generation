"""Measure source correspondence and controlled target distances on frozen Duke arrays."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import laplace
from threadpoolctl import threadpool_limits

from scripts.fresh_duke import replication as r
from scripts.fresh_duke.paired_stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, patient_estimate
from scripts.fresh_duke.verify_replication import read_images
from scripts.v4_convergence.analyze import features, fid
from src.explainability.ssim_map import compute_local_ssim_map


RUN = r.REPORT / 'replication_20261118_02'
PHASES = {'field_strength': (10, 110), 'manufacturer': (10,)}
FEATURE_JUDGES = ('original_predictor', 'training_robust_seed0')
PROJECTIONS = (16, 32)
REPORT = r.ROOT / 'reports/v25_fresh_image_diagnostics'


def check_digest(path, expected):
    if r.digest(path) != expected:
        raise ValueError(f'Changed input: {path}')


def load_archive():
    protocol_path = RUN / 'protocol.json'
    complete = r.load_json(RUN / 'complete.json')
    check_digest(protocol_path, complete['protocol_sha256'])
    protocol = r.load_json(protocol_path)
    check_digest(RUN / 'inputs.csv', protocol['input_sha256'])
    cohort = pd.read_csv(RUN / 'inputs.csv')
    cells = cohort.groupby(['mfg', 'fs']).patient_id.nunique()
    if (not cohort.slice_id.is_unique or len(cells) != 4 or not cells.eq(15).all()
            or cohort.patient_id.nunique() != 60 or not cohort.groupby('patient_id').size().eq(5).all()):
        raise ValueError('Unexpected patient or scanner cell coverage')
    native = np.stack([r.image_array(r.ROOT / row.image_path, row.image_sha256)
                       for row in cohort.itertuples()])
    static_path = RUN / 'static_field_strength.npz'
    check_digest(static_path, complete['static_sha256'])
    with np.load(static_path, allow_pickle=False) as pack:
        static = {arm: pack[arm] for arm in r.STATIC}
    np.testing.assert_array_equal(static['identity'], native)
    provenance = {r.rel(p): r.digest(p) for p in (protocol_path, RUN / 'complete.json', RUN / 'inputs.csv', static_path)}
    images = {}
    for axis, epochs in PHASES.items():
        for epoch in epochs:
            phase = f'epoch{epoch}_{axis}'
            folder = RUN / phase
            receipt = r.load_json(folder / 'complete.json')
            if axis == 'manufacturer':
                verified_path = r.REPORT / 'verification_20261119_02/verified.json'
                verified = r.load_json(verified_path)
                if (verified['run'] != RUN.name or verified['protocol_sha256'] != complete['protocol_sha256']
                        or verified['manufacturer']['arm_count'] != 2):
                    raise ValueError('Missing verified manufacturer follow-up')
                provenance[r.rel(verified_path)] = r.digest(verified_path)
            elif receipt != complete['phases'][phase]:
                raise ValueError(f'Phase receipt mismatch: {phase}')
            if (receipt['status'] != 'complete' or receipt['paired_target_slices'] != len(cohort)
                    or receipt['failed_arm_slices']):
                raise ValueError(f'Incomplete phase: {phase}')
            log = folder / 'generation.jsonl'
            check_digest(log, receipt['generation_log_sha256'])
            for path in (log, folder / 'complete.json'):
                provenance[r.rel(path)] = r.digest(path)
            controls = static if axis == 'field_strength' else {'identity': native}
            arrays, _ = read_images(folder, controls, len(cohort))
            generated = r.GENERATED if axis == 'field_strength' else ('target', 'source_label_null')
            if arrays.keys() != set(generated) | controls.keys():
                raise ValueError(f'Unexpected arms: {phase}')
            images[axis, epoch] = arrays
    return protocol, cohort, native, images, provenance


def pixel_metrics(cohort, native, images):
    foreground = native > .05
    count = foreground.sum(axis=(1, 2))
    if np.any(count == 0):
        raise ValueError('Empty source foreground')

    def energy(array):
        # Laplacian is computed separately per image, with reflect padding.
        return np.array([1000 * np.mean(laplace(x)[mask] ** 2)
                         for x, mask in zip(array, foreground, strict=True)])

    metadata = cohort[['slice_id', 'patient_id', 'mfg', 'fs']].assign(foreground_pixels=count)
    per_slice, summaries = [], []
    for (axis, epoch), arrays in images.items():
        label = r.AXES[axis]
        scores = {}
        for arm, array in arrays.items():
            scores[arm] = metadata.assign(
                axis=axis, epoch=epoch, arm=arm, source_label=cohort[label], target_label=1 - cohort[label],
                ssim=[compute_local_ssim_map(x, y)[0] for x, y in zip(native, array, strict=True)],
                hf_energy=energy(array),
            )
            per_slice.append(scores[arm])

        target = scores['target']
        for arm, values in scores.items():
            estimates = values[['ssim', 'hf_energy']].copy()
            if arm != 'target':
                if not values.hf_energy.gt(0).all():
                    raise ValueError(f'Undefined energy ratio: {axis}/{epoch}/{arm}')
                estimates['target_minus_control_ssim'] = target.ssim - values.ssim
                estimates['target_over_control_hf'] = target.hf_energy / values.hf_energy
            estimates = pd.concat([metadata, estimates], axis=1)
            for direction, subset in [('pooled', estimates), *estimates.groupby(label)]:
                for metric in estimates.columns[len(metadata.columns):]:
                    summaries.append(dict(axis=axis, epoch=epoch, source_label=direction, arm=arm,
                                          metric=metric, **patient_estimate(subset, metric, stratified=True)))
    return pd.concat(per_slice, ignore_index=True), pd.DataFrame(summaries)


def patient_draws(cohort):
    groups = list(cohort.groupby('patient_id').indices.values())
    indices = np.stack(groups)
    patient_cells = cohort.iloc[indices[:, 0]][['mfg', 'fs']].to_numpy()
    if not np.all(cohort[['mfg', 'fs']].to_numpy()[indices] == patient_cells[:, None]):
        raise ValueError('Inconsistent patient scanner labels')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.ones((BOOTSTRAP_DRAWS + 1, len(indices)))
    for cell in np.unique(patient_cells, axis=0):
        members = np.all(patient_cells == cell, axis=1)
        n = int(members.sum())
        weights[1:, members] = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_DRAWS)
    return indices, patient_cells, weights


def moments(vectors, weights):
    """Aggregate slice moments using each patient's bootstrap multiplicity."""
    _, slices, dim = vectors.shape
    count = weights.sum(axis=1) * slices
    mean = (weights @ vectors.sum(axis=1)) / count[:, None]
    scatter = vectors.swapaxes(1, 2) @ vectors
    second = (weights @ scatter.reshape(-1, dim * dim)).reshape(-1, dim, dim)
    covariance = (second - count[:, None, None] * mean[:, :, None] * mean[:, None, :]) / (count - 1)[:, None, None]
    return mean, covariance


def psd_root(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.swapaxes(-1, -2)) / 2)
    return (vectors * np.sqrt(values.clip(0))[:, None, :]) @ vectors.swapaxes(-1, -2)


def distance(vectors, weights, real_mean, real_cov, real_root):
    mean, cov = moments(vectors, weights)
    cross = real_root @ cov @ real_root
    eigenvalues = np.linalg.eigvalsh((cross + cross.swapaxes(1, 2)) / 2)
    value = ((mean - real_mean) ** 2).sum(axis=1)
    value += np.trace(cov + real_cov, axis1=1, axis2=2) - 2 * np.sqrt(eigenvalues.clip(0)).sum(axis=1)
    return value.clip(0)


def distance_rows(judge, raw, basis, cohort, axis, epoch, arms):
    indices, cells, weights = patient_draws(cohort)
    native = raw['identity']
    label_idx = ('mfg', 'fs').index(r.AXES[axis])
    rows = []
    for dim in (*PROJECTIONS, native.shape[1]):
        projected = raw if dim == native.shape[1] else {key: array @ basis[:, :dim] for key, array in raw.items()}
        directional = {}
        for source in (0, 1):
            source_mask = cells[:, label_idx] == source
            source_idx, ref_idx = indices[source_mask], indices[~source_mask]
            real = projected['identity'][ref_idx]
            if dim != native.shape[1]:
                mean, cov = moments(real, weights[:, ~source_mask])
                root = psd_root(cov)
            results = {}
            for arm in arms:
                generated = projected[arm][source_idx]
                direct = fid(generated.reshape(-1, dim), real.reshape(-1, dim))
                if dim == native.shape[1]:
                    results[arm] = np.array([direct])
                else:
                    draws = distance(generated, weights[:, source_mask], mean, cov, root)
                    np.testing.assert_allclose(draws[0], direct, atol=1e-6, rtol=1e-6)
                    results[arm] = draws
            directional[source] = results
        pooled = {arm: (directional[0][arm] + directional[1][arm]) / 2 for arm in arms}
        for source, results in [*directional.items(), ('pooled', pooled)]:
            for arm, draws in results.items():
                comparisons = [('distance', draws)]
                if arm != 'target':
                    comparisons.append(('target_minus_control', results['target'] - draws))
                for metric, values in comparisons:
                    lo, hi = np.quantile(values[1:], [.025, .975]) if len(values) > 1 else (None, None)
                    rows.append(dict(axis=axis, epoch=epoch, feature_model=judge, dimension=dim,
                                     source_label=source, arm=arm, metric=metric, estimate=values[0],
                                     lo95=lo, hi95=hi, bootstrap_draws=len(values) - 1,
                                     source_slices_per_direction=150, real_slices_per_direction=150,
                                     source_patients_per_direction=30, real_patients_per_direction=30))
    return rows


def distribution(protocol, cohort, images, provenance, out):
    train_path = r.TRAIN
    check_digest(train_path, protocol['files_sha256'][r.rel(train_path)])
    train = pd.read_csv(train_path)
    if set(train.patient_id) & set(cohort.patient_id):
        raise ValueError('Projection fit includes fresh patients')
    provenance[r.rel(train_path)] = r.digest(train_path)
    fit = np.stack([r.image_array(r.ROOT / p) for p in train.image_path])
    provenance['pca_fit_array_sha256'] = hashlib.sha256(fit).hexdigest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    archive = {}
    rows = []
    for judge in FEATURE_JUDGES:
        model_path = protocol['judges'][judge]
        check_digest(r.ROOT / model_path, protocol['files_sha256'][model_path])
        provenance[model_path] = r.digest(r.ROOT / model_path)
        model = r.load_assessor(r.ROOT / model_path, judge, device)
        original = judge == 'original_predictor'
        training = features(model, fit, original)
        _, eigenvectors = np.linalg.eigh(np.cov(training, rowvar=False))
        basis = eigenvectors[:, ::-1][:, :max(PROJECTIONS)].copy()
        archive[judge + '__pca_basis'] = basis
        archive[judge + '__training_features'] = training
        cached = {}
        for (axis, epoch), arrays in images.items():
            if axis != 'field_strength':
                continue
            raw = {}
            for arm, array in arrays.items():
                if id(array) not in cached:
                    cached[id(array)] = features(model, array, original)
                raw[arm] = cached[id(array)]
                archive[f'{judge}__{axis}__epoch{epoch}__{arm}'] = raw[arm]
            print(f'Distance bootstrap: {judge}, epoch {epoch}', flush=True)
            rows.extend(distance_rows(judge, raw, basis, cohort, axis, epoch, arrays))
        del model
    with (out / 'features.npz').open('xb') as stream:
        np.savez_compressed(stream, **archive)
    return pd.DataFrame(rows)


def main(out):
    protocol, cohort, native, images, provenance = load_archive()
    out.mkdir(parents=True, exist_ok=False)
    per_slice, summaries = pixel_metrics(cohort, native, images)
    r.save_csv(out / 'slice_image_metrics.csv', per_slice)
    r.save_csv(out / 'patient_image_summary.csv', summaries)
    print(f'Wrote source correspondence and texture for {len(cohort)} fresh slices', flush=True)
    with threadpool_limits(limits=2, user_api='blas'):
        distances = distribution(protocol, cohort, images, provenance, out)
    r.save_csv(out / 'target_distances.csv', distances)
    r.json_write(out / 'inputs.json', {'inputs': provenance, 'script_sha256': r.digest(Path(__file__)),
                                     'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
                                     'fit_split': 'train', 'torch': torch.__version__})
    (out / 'README.md').write_text(
        '# Fresh Duke image diagnostics\n\n'
        'Run from the project code folder with `python -m scripts.fresh_duke.image_diagnostics --out <new-folder>`. '
        'The script creates all five data files listed below from frozen fresh arrays. These diagnostics were added '
        'after the judge analysis and are descriptive.\n\n'
        '- `slice_image_metrics.csv` records source SSIM and scaled foreground Laplacian energy on each slice and arm. '
        'All arms use the same source mask (intensity > .05).\n'
        '- `patient_image_summary.csv` reports equal-patient averages and scanner-cell-stratified percentile intervals. '
        'Energy ratios are computed within a slice before averaging within and across patients.\n'
        '- `target_distances.csv` reports Fréchet distances to opposite-label fresh native images within each direction. '
        'Each direction uses two balanced manufacturer strata on each side and disjoint source and reference patients. '
        'The pooled distance is the arithmetic mean of the two directional distances. Bootstraps sample whole patients '
        'within scanner cells and reuse cell weights wherever those patients contribute, including across arms and '
        'directions. Target-minus-control differences use common reference draws and paired source draws.\n'
        '- `features.npz` contains saved embeddings in frozen input order and PCA bases from original training features. '
        'Distances in 16 dimensions are primary for this diagnostic and 32 dimensions are a sensitivity check. '
        'Full-dimensional values are point estimates. Projection reduces covariance dimension. Finite-sample bias '
        'and the choice of embedding still limit interpretation.\n'
        '- `inputs.json` records source receipts, checkpoint hashes, the PCA input array hash and resampling settings.\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPORT / 'analysis_20261119_01')
    main(parser.parse_args().out)
