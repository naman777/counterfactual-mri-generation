"""Paired tests and seed variance for B3 under the B2 held-out judges.

Reads the per-slice re-grades made by scripts/v4_heldout/local_run.py and applies
the same tests as scripts/v4_convergence/analyze.py (exact McNemar on hits,
Wilcoxon on gains, Holm over the five baselines, exact patient sign test).
Adds a matched comparison of the converged (epoch-110) edits against the
epoch-10 edits on the same 100 test slices.

python scripts/v4_review/heldout_b3_stats.py --run-id v4_stats_heldout_20261016_01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4_convergence.analyze import holm  # noqa: E402
from scripts.v4_convergence.core import digest, json_write  # noqa: E402

B2 = ROOT / 'reports/v4_heldout_judge/b2_local_20261016_01'
OLD_SETS = {'field_strength': 'old_fs_t100_w5', 'manufacturer': 'old_mfg_t300_w5'}


def compare(d, b, key='hit', gain='gain'):
    m = d.merge(b, on=['test_index', 'patient_id'], suffixes=('_d', '_b'), validate='one_to_one')
    if len(m) != 100:
        raise ValueError('Incomplete paired comparison')
    hd, hb = m[f'{key}_d'], m[f'{key}_b']
    n10, n01 = int(((hd == 1) & (hb == 0)).sum()), int(((hd == 0) & (hb == 1)).sum())
    diff = m[f'{gain}_d'] - m[f'{gain}_b']
    per_patient = (hd - hb).groupby(m.patient_id).mean()
    pos, neg = int((per_patient > 0).sum()), int((per_patient < 0).sum())
    return dict(n=len(m), hit_a=float(hd.mean()), hit_b=float(hb.mean()), n_a_only=n10, n_b_only=n01,
                p_mcnemar=float(binomtest(min(n10, n01), n10 + n01).pvalue) if n10 + n01 else 1.,
                median_gain_diff=float(diff.median()),
                p_wilcoxon=float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.,
                patients_a_better=pos, patients_b_better=neg,
                p_sign_patient=float(binomtest(min(pos, neg), pos + neg).pvalue) if pos + neg else 1.)


def main(run_id):
    out = ROOT / 'reports' / run_id
    out.mkdir(parents=True, exist_ok=False)
    b3 = pd.read_csv(B2 / 'regrade_b3/per_slice.csv')
    old = pd.read_csv(B2 / 'regrade_old/per_slice.csv')
    paired, versus_old, variance = [], [], []
    for (family, seed), g in b3.groupby(['family', 'seed']):
        for axis in ('field_strength', 'manufacturer'):
            main_s0 = g[(g.axis == axis) & (g.kind == 'main') & (g.offset == 0)]
            bases = g[(g.axis == axis) & (g.kind == 'baselines')]
            block = []
            for baseline, b in bases.groupby('variant'):
                block.append(dict(family=family, seed=seed, axis=axis, a='diffusion_epoch110', b=baseline, **compare(main_s0, b)))
            if len(block) != 5:
                raise ValueError('Expected five baselines')
            for key in ('p_mcnemar', 'p_wilcoxon'):
                for row, adjusted in zip(block, holm([row[key] for row in block])):
                    row[key + '_holm'] = float(adjusted)
            paired.extend(block)
            o = old[(old.family == family) & (old.seed == seed) & (old.set == OLD_SETS[axis])]
            versus_old.append(dict(family=family, seed=seed, axis=axis, a='diffusion_epoch110_selected',
                                   b=OLD_SETS[axis] + '_epoch10', **compare(main_s0, o)))
            seeds = g[(g.axis == axis) & (g.kind == 'main')].pivot_table(index='test_index', columns='offset', values='hit')
            rates = seeds.mean(0)
            variance.append(dict(family=family, seed=seed, axis=axis, hit_by_generation_seed=';'.join(f'{v:.2f}' for v in rates),
                                 hit_mean=float(rates.mean()), hit_sd=float(rates.std(ddof=1)),
                                 all_seeds_hit=float((seeds.min(1) == 1).mean())))
    pd.DataFrame(paired).to_csv(out / 'paired_tests_heldout.csv', index=False)
    pd.DataFrame(versus_old).to_csv(out / 'epoch110_vs_epoch10_heldout.csv', index=False)
    pd.DataFrame(variance).to_csv(out / 'seed_variance_heldout.csv', index=False)
    json_write(out / 'run.json', {'script': 'scripts/v4_review/heldout_b3_stats.py', 'script_sha256': digest(Path(__file__)),
                                  'inputs': {p: digest(B2 / p) for p in ['regrade_b3/per_slice.csv', 'regrade_old/per_slice.csv']}})
    (out / 'README.md').write_text(
        '# Held-out judge statistics for B3 (2026-10-16)\n\n'
        'All files made by `scripts/v4_review/heldout_b3_stats.py` from the per-slice re-grades in '
        '`reports/v4_heldout_judge/b2_local_20261016_01/regrade_{b3,old}/per_slice.csv` (hashes in `run.json`). '
        'Judges: the six B2 held-out judges (`orientation_robust` primary, `legacy_comparison` horizontal-flip-only), seeds 0-2.\n\n'
        '| File | Content |\n|---|---|\n'
        '| `paired_tests_heldout.csv` | Converged (epoch-110) edits at the validation-selected setting (generation seed offset 0) against each of the five baselines on the same 100 test slices: exact McNemar on hits, Wilcoxon on gains, Holm over the five baselines, exact sign test on the 16 per-patient mean hit differences (uncorrected). Same tests as `scripts/v4_convergence/analyze.py`. |\n'
        '| `epoch110_vs_epoch10_heldout.csv` | Same tests, converged selected edits against the epoch-10 edits (FS t=100 w=5; manufacturer t=300 w=5) on the same slices. Note the manufacturer settings differ (epoch 110: t=300 w=3). |\n'
        '| `seed_variance_heldout.csv` | Hit rate for each of the three generation seeds, their mean and SD, and the share of slices hit under all three. |\n',
        encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
