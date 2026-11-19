"""Orientation check of the B1 synthetic-generation gate samples.

Re-grades the saved t=999 gate images with the three robust judges under
identity, horizontal flip, vertical flip and 180-degree rotation. Conditions
and labels stay fixed. Read-only with respect to B1: gate files are copied
into this report's inputs/ folder and hashed; nothing is generated or trained.

python scripts/v4_review/gate_orientation_check.py --run-id gate_orientation_check_20261016_01 --gate-source <folder holding gate_initial/, gate_0020/, ...>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4_convergence.core import digest, json_write, judges, load_pack, probabilities  # noqa: E402
from scripts.v4_heldout.evaluate_heldout import wilson  # noqa: E402

B1 = ROOT / 'reports/v4_diffusion/b1_20261016_01'
TRANSFORMS = {'identity': lambda x: x, 'horizontal_flip': lambda x: np.flip(x, -1),
              'vertical_flip': lambda x: np.flip(x, -2), 'rotate_180': lambda x: np.flip(x, (-2, -1))}


def upper_half_fraction(images):
    total = images.sum((1, 2))
    upper = images[:, :images.shape[1] // 2].sum((1, 2))
    # Blank samples have no mass; they are counted separately, not as orientation evidence.
    return np.where(total >= 1, upper / np.maximum(total, 1e-12), np.nan)


def copy_inputs(source, out):
    rows = []
    for folder in sorted(Path(source).glob('gate_*')):
        for path in sorted(folder.glob('epoch_*_w*.json')) + sorted(folder.glob('epoch_*_images.npz')):
            dest = out / 'inputs' / folder.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            rows.append({'path': dest.relative_to(out).as_posix(), 'sha256': digest(dest),
                         'remote_path': f'capstone-judge-data:/v4_convergence/b1_20261016_01/full/{folder.name}/{path.name}'})
    if not rows:
        raise ValueError('No gate files found')
    return rows


def main(run_id, gate_source):
    out = ROOT / 'reports/v4_diffusion' / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = copy_inputs(gate_source, out)
    json_write(out / 'input_manifest.json', manifest)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = judges(B1 / 'inputs/assets', device=device, robust_only=True)

    data = load_pack(B1 / 'inputs/pack.npz', splits=['train'])
    real_x, real_m, real_f = data['train_x'].astype(np.float32), data['train_mfg'], data['train_fs']
    real_up = upper_half_fraction(real_x)
    reference = {
        'pack_sha256': digest(B1 / 'inputs/pack.npz'),
        'train_slices_by_mfg_fs': {f'mfg{m}_fs{f}': int(((real_m == m) & (real_f == f)).sum()) for m in (0, 1) for f in (0, 1)},
        'real_train_upper_half_mass_fraction': {
            name: {'mean': float(np.nanmean(real_up[real_m == m])), 'sd': float(np.nanstd(real_up[real_m == m])),
                   'n_slices': int((real_m == m).sum())} for m, name in [(0, 'GE'), (1, 'Siemens')]},
        'real_train_blank_slices': int((real_x.sum((1, 2)) < 1).sum()),
    }
    json_write(out / 'real_reference.json', reference)

    rows, blank_rows, layout_rows, check_rows = [], [], [], []
    for meta_path in sorted((out / 'inputs').glob('gate_*/epoch_*_w*.json')):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.stem
        with np.load(meta_path.with_name(stem + '_images.npz'), allow_pickle=False) as pack:
            images, cond = pack['images'], {k: pack[k].astype(int) for k in ('mfg', 'fs')}
        epoch, w = int(meta['epoch']), float(meta['guidance'])
        mass = images.sum((1, 2))
        up = upper_half_fraction(images)
        for m in (0, 1):
            for f in (0, 1):
                cell = (cond['mfg'] == m) & (cond['fs'] == f)
                blank_rows.append({'epoch': epoch, 'guidance': w, 'mfg': m, 'fs': f, 'n': int(cell.sum()),
                                   'blank_samples': int((cell & (mass < 1)).sum())})
            layout_rows.append({'epoch': epoch, 'guidance': w, 'vendor': ['GE', 'Siemens'][m],
                                'n_nonblank': int(np.isfinite(up[cond['mfg'] == m]).sum()),
                                'upper_half_mass_fraction_mean': float(np.nanmean(up[cond['mfg'] == m]))})
        for name, model in models.items():
            for transform, fn in TRANSFORMS.items():
                probs = probabilities(model, np.ascontiguousarray(fn(images)))
                for axis in ('mfg', 'fs'):
                    correct = (probs[axis].argmax(1) == cond[axis]).astype(float)
                    rows.append({'epoch': epoch, 'global_step': int(meta['global_step']), 'guidance': w,
                                 'judge': name, 'axis': axis, 'transform': transform, 'n': len(correct),
                                 'correct': int(correct.sum()), 'rate': float(correct.mean()), **wilson(correct)})
                    if transform == 'identity':
                        check_rows.append({'epoch': epoch, 'guidance': w, 'judge': name, 'axis': axis,
                                           'stored_gate_rate': meta['by_judge'][name][axis],
                                           'recomputed_rate': float(correct.mean())})
        print('done', stem, flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary_by_judge.csv', index=False)
    pooled = (summary.groupby(['epoch', 'global_step', 'guidance', 'axis', 'transform'], as_index=False)
              .agg(n_per_judge=('n', 'first'), mean_rate=('rate', 'mean'), min_rate=('rate', 'min'), max_rate=('rate', 'max')))
    pooled.to_csv(out / 'summary_mean_of_judges.csv', index=False)
    pd.DataFrame(blank_rows).to_csv(out / 'blank_samples.csv', index=False)
    pd.DataFrame(layout_rows).to_csv(out / 'generated_vertical_layout.csv', index=False)
    checks = pd.DataFrame(check_rows)
    checks['abs_difference'] = (checks.stored_gate_rate - checks.recomputed_rate).abs()
    checks.to_csv(out / 'reproduction_check.csv', index=False)
    json_write(out / 'run.json', {
        'script': 'scripts/v4_review/gate_orientation_check.py', 'script_sha256': digest(Path(__file__)),
        'device': device, 'judges': list(models), 'gate_files': len(manifest),
        'max_identity_reproduction_difference': float(checks.abs_difference.max()),
        'interval': 'Wilson 95% over the 100 synthetic samples per gate; samples share fixed noise seeds across epochs; no patients are involved',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gate-source', required=True)
    args = parser.parse_args()
    main(args.run_id, args.gate_source)
