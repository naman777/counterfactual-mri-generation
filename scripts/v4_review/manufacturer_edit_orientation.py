"""Orientation check of the existing (epoch-10) manufacturer edits used in paper v4.

Inputs are read, never changed:
  t=300 w=5: reports/phase12/manufacturer/{manufacturer_results.csv, counterfactuals/}
  t=500/999 w=5: reports/v2_quickwins/stronger_edits/manufacturer_t{500,999}_w5.npz
Source images are the processed .npy files named in each set's image_path.

Two questions per edited slice:
  1. Anatomy: does the edit resemble the source, or the vertically flipped source?
  2. Judges: do target-vendor hits survive a vertical flip of the edited image?

python scripts/v4_review/manufacturer_edit_orientation.py --run-id mfg_edit_orientation_20261016_01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4_convergence.core import digest, json_write, judges, load_pack, probabilities  # noqa: E402
from scripts.v4_heldout.orientation_probe import intervals  # noqa: E402

B1 = ROOT / 'reports/v4_diffusion/b1_20261016_01'
PHASE12 = ROOT / 'reports/phase12/manufacturer'
STRONGER = ROOT / 'reports/v2_quickwins/stronger_edits'


def load_source(path):
    return np.clip(np.load(ROOT / str(path).replace('\\', '/')).astype(np.float32), 0, 1)


def load_sets():
    sets, inputs = [], []
    table = pd.read_csv(PHASE12 / 'manufacturer_results.csv')
    files = [PHASE12 / 'counterfactuals' / f'{e}_counterfactual.npy' for e in table.example_id]
    sets.append(('t300_w5', 300, np.stack([np.load(f).astype(np.float32) for f in files]),
                 table.image_path.to_numpy(str), table.patient_id.to_numpy(str)))
    inputs += [PHASE12 / 'manufacturer_results.csv', *files]
    for t in (500, 999):
        path = STRONGER / f'manufacturer_t{t}_w5.npz'
        with np.load(path, allow_pickle=False) as pack:
            sets.append((f't{t}_w5', t, pack['images'].astype(np.float32), pack['image_path'], pack['patient_id']))
        inputs.append(path)
    return sets, inputs


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / max(np.sqrt((a @ a) * (b @ b)), 1e-12))


def upper_half_fraction(x):
    total = x.sum()
    return float(x[:x.shape[0] // 2].sum() / total) if total >= 1 else np.nan


def summarise(frame, by, metrics):
    rows = []
    for keys, group in frame.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for direction in ['all', 'GE_to_Siemens', 'Siemens_to_GE']:
            sub = group if direction == 'all' else group[group.direction == direction]
            if not len(sub):
                continue
            row = {**dict(zip(by, keys)), 'direction': direction, 'n': len(sub), 'patients': sub.patient_id.nunique()}
            for metric in metrics:
                values = sub[metric].to_numpy(float)
                values = values[np.isfinite(values)]
                if len(values):
                    stats = intervals(values, sub.patient_id.to_numpy()[np.isfinite(sub[metric].to_numpy(float))])
                    row.update({f'{metric}_{k}': v for k, v in stats.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def main(run_id):
    out = ROOT / 'reports/v4_diffusion' / run_id
    out.mkdir(parents=True, exist_ok=False)
    sets, input_files = load_sets()
    data = load_pack(B1 / 'inputs/pack.npz', splits=['test'])
    label = {str(p): int(m) for p, m in zip(data['test_path'], data['test_mfg'])}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = judges(B1 / 'inputs/assets', device=device)

    anatomy, grades = [], []
    for name, t, edits, paths, patients in sets:
        sources = np.stack([load_source(p) for p in paths])
        src = np.array([label[str(p)] for p in paths])
        tgt = 1 - src
        direction = np.where(src == 0, 'GE_to_Siemens', 'Siemens_to_GE')
        for i in range(len(edits)):
            e, s, fs = edits[i], sources[i], np.flip(sources[i], 0)
            anatomy.append({
                'set': name, 't': t, 'example_index': i, 'patient_id': str(patients[i]), 'image_path': str(paths[i]),
                'direction': direction[i],
                'ssim_to_source': float(structural_similarity(s, e, data_range=1.0, win_size=7)),
                'ssim_to_flipped_source': float(structural_similarity(np.ascontiguousarray(fs), e, data_range=1.0, win_size=7)),
                'corr_to_source': corr(e, s), 'corr_to_flipped_source': corr(e, fs),
                'upper_half_source': upper_half_fraction(s), 'upper_half_edit': upper_half_fraction(e),
            })
        for judge, model in models.items():
            original = judge == 'original_predictor'
            before = probabilities(model, sources, original=original)['mfg'].argmax(1)
            for transform, images in [('identity', edits), ('vertical_flip', np.flip(edits, -2))]:
                pred = probabilities(model, np.ascontiguousarray(images), original=original)['mfg'].argmax(1)
                for i in range(len(edits)):
                    grades.append({'set': name, 't': t, 'judge': judge, 'transform': transform, 'example_index': i,
                                   'patient_id': str(patients[i]), 'direction': direction[i],
                                   'source_correct': float(before[i] == src[i]), 'hit': float(pred[i] == tgt[i])})
        print('done', name, flush=True)

    anatomy = pd.DataFrame(anatomy)
    anatomy['flipped_anatomy'] = (anatomy.corr_to_flipped_source > anatomy.corr_to_source).astype(float)
    anatomy['upper_half_change'] = anatomy.upper_half_edit - anatomy.upper_half_source
    anatomy.to_csv(out / 'per_slice_anatomy.csv', index=False)
    summarise(anatomy, ['set', 't'], ['flipped_anatomy', 'ssim_to_source', 'ssim_to_flipped_source',
                                      'corr_to_source', 'corr_to_flipped_source', 'upper_half_change']
              ).to_csv(out / 'summary_anatomy.csv', index=False)

    grades = pd.DataFrame(grades)
    grades.to_csv(out / 'per_slice_judges.csv', index=False)
    summarise(grades, ['set', 't', 'judge', 'transform'], ['hit', 'source_correct']).to_csv(out / 'summary_judges.csv', index=False)
    # Hits split by whether the edit's anatomy is upright or upside down relative to its source.
    joined = grades[grades['transform'] == 'identity'].merge(anatomy[['set', 'example_index', 'flipped_anatomy']], on=['set', 'example_index'])
    split = (joined.groupby(['set', 't', 'judge', 'flipped_anatomy'], as_index=False)
             .agg(n=('hit', 'size'), hits=('hit', 'sum'), hit_rate=('hit', 'mean')))
    split.to_csv(out / 'hits_by_anatomy_orientation.csv', index=False)

    json_write(out / 'run.json', {
        'script': 'scripts/v4_review/manufacturer_edit_orientation.py', 'script_sha256': digest(Path(__file__)),
        'device': device, 'judges': list(models), 'sets': {name: len(edits) for name, _, edits, _, _ in sets},
        'inputs': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for p in input_files],
        'labels_from': {'path': 'reports/v4_diffusion/b1_20261016_01/inputs/pack.npz', 'sha256': digest(B1 / 'inputs/pack.npz')},
        'flipped_anatomy_rule': 'Pearson correlation of the edit with the vertically flipped source exceeds its correlation with the source',
        'intervals': '10,000-resample bootstrap, slice-level and patient-clustered (orientation_probe.intervals, seed 42)',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
