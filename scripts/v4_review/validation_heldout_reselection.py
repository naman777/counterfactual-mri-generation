"""Post-hoc: which B3 operating points would the held-out judges have chosen?

Re-grades the 24 saved validation edits of run b1_20261016_01 (100 validation
slices, 15 patients) with the frozen B2 held-out judges and applies the frozen
selection rule (core.choose_validation, protocol.json). Labelled post-hoc: it
was run after the test results were known. Nothing is regenerated.

python scripts/v4_review/validation_heldout_reselection.py --run-id val_heldout_reselection_20261016_01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v2_quickwins.judge_model import load_judge  # noqa: E402
from scripts.v4_convergence.core import AXES, choose_validation, digest, json_write, probabilities  # noqa: E402

B1 = ROOT / 'reports/v4_diffusion/b1_20261016_01'
B2 = ROOT / 'reports/v4_heldout_judge/b2_local_20261016_01'


def main(run_id):
    out = ROOT / 'reports/v4_diffusion' / run_id
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol = json.loads((B1 / 'protocol.json').read_text())
    models = {}
    for entry in json.loads((B2 / 'frozen_checkpoints.json').read_text()):
        path = B2 / entry['checkpoint']
        if digest(path) != entry['sha256']:
            raise ValueError('Frozen held-out checkpoint changed')
        models[(entry['family'], entry['seed'])] = load_judge(path, device)
    rows, inputs = [], []
    for folder in sorted((B1 / 'remote/full/validation').glob('val_*')):
        job = json.loads((folder / 'job.json').read_text())
        robust = json.loads((folder / 'result.json').read_text())
        with np.load(folder / 'images.npz', allow_pickle=False) as pack:
            cf, src_img = pack['cf'], pack['source_images']
            src = {'mfg': pack['source_mfg'].astype(int), 'fs': pack['source_fs'].astype(int)}
        inputs.append({'path': (folder / 'images.npz').relative_to(ROOT).as_posix(), 'sha256': digest(folder / 'images.npz')})
        key = AXES[job['axis']]
        other = 'fs' if key == 'mfg' else 'mfg'
        target = 1 - src[key]
        for (family, seed), model in models.items():
            before, after = probabilities(model, src_img), probabilities(model, cf)
            rows.append({'axis': job['axis'], 't': job['t'], 'w': job['w'], 'family': family, 'seed': seed,
                         'ssim': robust['ssim'], 'robust_judges_hit': robust['hit'],
                         'hit': float((after[key].argmax(1) == target).mean()),
                         'gain': float((after[key][np.arange(len(target)), target] - before[key][np.arange(len(target)), target]).mean()),
                         'stability': float((after[other].argmax(1) == before[other].argmax(1)).mean())})
        print('done', folder.name, flush=True)
    per_judge = pd.DataFrame(rows)
    per_judge.to_csv(out / 'per_judge.csv', index=False)
    grid = (per_judge.groupby(['family', 'axis', 't', 'w'], as_index=False)
            .agg(ssim=('ssim', 'first'), robust_judges_hit=('robust_judges_hit', 'first'),
                 hit=('hit', 'mean'), gain=('gain', 'mean'), stability=('stability', 'mean')))
    grid.to_csv(out / 'grid.csv', index=False)
    choice = {}
    for family, g in grid.groupby('family'):
        records = g.to_dict('records')
        choice[family] = {axis: choose_validation(records, axis, protocol) for axis in AXES}
    original = json.loads((B1 / 'remote/full/selected_operating_points.json').read_text())
    json_write(out / 'choice.json', {'post_hoc': True, 'rule': protocol['validation'], 'heldout_choice': choice,
                                     'prespecified_choice_robust_judges': {a: {k: original[a][k] for k in ('t', 'w', 'hit', 'ssim', 'stability', 'feasible')} for a in original}})
    json_write(out / 'run.json', {'script': 'scripts/v4_review/validation_heldout_reselection.py', 'script_sha256': digest(Path(__file__)),
                                  'inputs': inputs, 'frozen_checkpoints_sha256': digest(B2 / 'frozen_checkpoints.json')})
    (out / 'README.md').write_text(
        '# Post-hoc validation re-selection with the held-out judges (2026-10-16)\n\n'
        'Made by `scripts/v4_review/validation_heldout_reselection.py` (hashes in `run.json`). **Post-hoc**: run after the B3 test '
        'results were known; it does not change the reported operating points. Inputs: the 24 saved validation edits of run '
        '`b1_20261016_01` (100 validation slices from 15 patients; t in {50,100,200,300}, w in {1,3,5}, both axes). Judges: the six '
        'frozen B2 held-out judges. The frozen rule `core.choose_validation` with `protocol.json` is applied per judge family, with '
        'hit, gain and other-parameter stability from that family (mean of three seeds) and SSIM from the original run.\n\n'
        '| File | Content |\n|---|---|\n'
        '| `per_judge.csv` | Hit, gain and stability per setting and judge, with the original robust-judge hit and SSIM |\n'
        '| `grid.csv` | Mean over the three seeds of each family |\n'
        '| `choice.json` | Setting the rule selects for each family, next to the prespecified choice |\n',
        encoding='utf-8')
    print(json.dumps(choice, indent=1, default=float))
    print(grid[grid.family == 'orientation_robust'].round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
