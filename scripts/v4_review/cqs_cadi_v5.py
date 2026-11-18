"""CQS/CADI for both models under the held-out judges, and the SSIM-vs-gain figure.

Uses the paper's definitions (src/explainability/engine.py, analyze.py):
  CQS  = (SSIM + clip((gain + 1) / 2) + other-parameter stability) / 3, averaged over slices
  CADI = hit rate x mean gain / mean MAD
SSIM and MAD come from the saved per-slice results; gains, hits and stability
from each judge. Inputs are read only; nothing is generated.

python scripts/v4_review/cqs_cadi_v5.py --run-id v4_cqs_cadi_20261016_01
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4_convergence.core import digest, json_write, load_pack  # noqa: E402

R = ROOT / 'reports'
AN = R / 'v4_diffusion/b1_20261016_01/remote/analysis/20261016T153744Z'
B2 = R / 'v4_heldout_judge/b2_local_20261016_01'
DEV = {'field_strength': R / 'v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv',
       'manufacturer': R / 'phase12/manufacturer/manufacturer_results.csv'}
DEV_SET = {'field_strength': 'old_fs_t100_w5', 'manufacturer': 'old_mfg_t300_w5'}
CONV_JOB = {'field_strength': 'main_field_strength_t100_w5_s0', 'manufacturer': 'main_manufacturer_t300_w3_s0',
            'manufacturer_w5': 'fixed_manufacturer_t300_w5_s0'}
KEY = {'field_strength': ('fs', 'mfg'), 'manufacturer': ('mfg', 'fs')}


def scores(frame):
    cqs = (frame.ssim.clip(0, 1) + ((frame.gain + 1) / 2).clip(0, 1) + frame.stable) / 3
    return dict(n=len(frame), ssim=frame.ssim.mean(), mad=frame.mad.mean(), hit=frame.hit.mean(), gain=frame.gain.mean(),
                stability=frame.stable.mean(), cqs=cqs.mean(), cadi=frame.hit.mean() * frame.gain.mean() / frame.mad.mean())


def main(run_id):
    out = R / run_id
    out.mkdir(parents=True, exist_ok=False)
    pack = load_pack(R / 'v4_diffusion/b1_20261016_01/inputs/pack.npz', splits=['test'])
    path_to_index = {str(p): i for i, p in enumerate(pack['test_path'])}
    old = pd.read_csv(B2 / 'regrade_old/per_slice.csv')
    new = pd.read_csv(B2 / 'regrade_b3/per_slice.csv')
    an = pd.read_csv(AN / 'per_example.csv')
    rows = []
    for axis, path in DEV.items():
        key, other = KEY[axis]
        d = pd.read_csv(path)
        d['test_index'] = d.image_path.map(path_to_index)
        base = d[['test_index', 'global_ssim', 'mean_abs_diff']].rename(columns={'global_ssim': 'ssim', 'mean_abs_diff': 'mad'})
        orig = base.assign(gain=d[f'{key}_gain'], hit=d[f'{key}_hit'].astype(float), stable=d[f'{other}_stable'].astype(float),
                           judge='original_predictor')
        rows.append(orig.assign(model='epoch10', axis=axis))
        for (family, seed), g in old[old.set == DEV_SET[axis]].groupby(['family', 'seed']):
            m = base.merge(g[['test_index', 'gain', 'hit', 'other_stability']], on='test_index', validate='one_to_one')
            rows.append(m.rename(columns={'other_stability': 'stable'}).assign(judge=f'heldout_{family}{seed}', model='epoch10', axis=axis))
    for name, job in CONV_JOB.items():
        axis = 'manufacturer' if name.startswith('manufacturer') else name
        key, other = KEY[axis]
        a = an[(an.job == job) & (an.variant == 'cf')].copy()
        a['test_index'] = a.image_path.map(path_to_index)
        base = a[a.judge == 'original_predictor'][['test_index', 'ssim', 'mad']]
        for judge, g in a.groupby('judge'):
            rows.append(base.merge(pd.DataFrame({'test_index': g.test_index, 'gain': g[f'{key}_gain'], 'hit': g[f'{key}_hit'].astype(float),
                                                 'stable': g[f'{other}_stable'].astype(float)}), on='test_index', validate='one_to_one')
                        .assign(judge=judge, model='epoch110' + ('_w5' if name.endswith('w5') else ''), axis=axis))
        for (family, seed), g in new[(new.set == job) & (new.variant == 'cf')].groupby(['family', 'seed']):
            m = base.merge(g[['test_index', 'gain', 'hit', 'other_stability']], on='test_index', validate='one_to_one')
            rows.append(m.rename(columns={'other_stability': 'stable'}).assign(judge=f'heldout_{family}{seed}',
                        model='epoch110' + ('_w5' if name.endswith('w5') else ''), axis=axis))
    per_slice = pd.concat(rows, ignore_index=True)
    if per_slice.test_index.isna().any():
        raise ValueError('Unmatched test slice')
    per_slice.to_csv(out / 'per_slice.csv', index=False)
    table = pd.DataFrame([dict(model=m, axis=a, judge=j, **scores(g)) for (m, a, j), g in per_slice.groupby(['model', 'axis', 'judge'])])
    table.to_csv(out / 'cqs_cadi.csv', index=False)

    # SSIM against target gain, original predictor vs held-out judge 0, both models.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    style = {'epoch10': dict(color='#9a9a9a', marker='o', label='Epoch 10'),
             'epoch110': dict(color='#1f5fae', marker='s', label='Epoch 110 (selected)')}
    for col, axis in enumerate(['field_strength', 'manufacturer']):
        for row, judge in enumerate(['original_predictor', 'heldout_orientation_robust0']):
            ax = axes[row, col]
            for model, st in style.items():
                g = per_slice[(per_slice.axis == axis) & (per_slice.model == model) & (per_slice.judge == judge)]
                ax.scatter(g.ssim, g.gain, s=10, alpha=0.65, color=st['color'], marker=st['marker'], label=st['label'], linewidths=0)
            ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
            ax.set_title(f"{'Field strength' if col == 0 else 'Manufacturer'}, {'original predictor' if row == 0 else 'held-out judge 0'}", fontsize=9)
            if row == 1:
                ax.set_xlabel('SSIM to input', fontsize=9)
            if col == 0:
                ax.set_ylabel('Target-probability gain', fontsize=9)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, loc='lower left', frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_tradeoff_v5.{ext}', dpi=200)
    inputs = [AN / 'per_example.csv', B2 / 'regrade_old/per_slice.csv', B2 / 'regrade_b3/per_slice.csv', *DEV.values()]
    json_write(out / 'run.json', {'script': 'scripts/v4_review/cqs_cadi_v5.py', 'script_sha256': digest(Path(__file__)),
                                  'inputs': {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}})
    (out / 'README.md').write_text(
        '# CQS/CADI under every judge, and the SSIM-vs-gain figure (2026-10-16)\n\n'
        'All files made by `scripts/v4_review/cqs_cadi_v5.py` (input hashes in `run.json`). Definitions as in the paper and '
        '`src/explainability/engine.py`: CQS = (SSIM + clip((gain+1)/2) + other-parameter stability)/3 averaged over slices; '
        'CADI = hit rate x mean gain / mean MAD. The script reproduces the v4 development-model values for the original predictor '
        '(FS w=5: 0.901 / 31.17; manufacturer: 0.727 / 5.99).\n\n'
        'Models: `epoch10` = development model (FS t=100 w=5 from `reports/v2_quickwins/fs_t100_w5/`, manufacturer t=300 w=5 from '
        '`reports/phase12/manufacturer/`); `epoch110` = converged model at the validation-selected settings (FS t=100 w=5, '
        'manufacturer t=300 w=3); `epoch110_w5` = converged manufacturer at t=300 w=5. Same 100 test slices throughout.\n\n'
        '| File | Content |\n|---|---|\n'
        '| `per_slice.csv` | SSIM, MAD, and per judge the gain, hit and other-parameter stability for every slice |\n'
        '| `cqs_cadi.csv` | CQS, CADI and their components per model, axis and judge |\n'
        '| `fig_tradeoff_v5.pdf/.png` | SSIM against target gain, original predictor (top) and held-out orientation-robust judge 0 (bottom) |\n',
        encoding='utf-8')
    print(table[table.judge.isin(['original_predictor', 'robust0', 'heldout_orientation_robust0', 'heldout_orientation_robust1',
                                  'heldout_orientation_robust2'])].round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
