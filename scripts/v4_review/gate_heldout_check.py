"""Grade every saved B1 gate sample with the B2 held-out judges, with and without a vertical flip.

Complements gate_orientation_check.py (robust judges, epochs 10-90) with all gates
(epochs 10-110) and the six held-out judges. Reads the local fetch mirror of B1;
nothing is generated or trained.

python scripts/v4_review/gate_heldout_check.py --run-id gate_heldout_check_20261016_01
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
from scripts.v4_convergence.core import digest, json_write, judges, probabilities  # noqa: E402
from scripts.v4_heldout.evaluate_heldout import wilson  # noqa: E402

GATES = ROOT / 'reports/v4_diffusion/b1_20261016_01/remote/full'
B2 = ROOT / 'reports/v4_heldout_judge/b2_local_20261016_01'


def main(run_id):
    out = ROOT / 'reports/v4_diffusion' / run_id
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = dict(judges(ROOT / 'reports/v4_diffusion/b1_20261016_01/inputs/assets', device=device, robust_only=True))
    for entry in json.loads((B2 / 'frozen_checkpoints.json').read_text()):
        path = B2 / entry['checkpoint']
        if digest(path) != entry['sha256']:
            raise ValueError('Frozen held-out checkpoint changed')
        models[f"heldout_{entry['family']}{entry['seed']}"] = load_judge(path, device)
    rows, inputs = [], []
    for meta_path in sorted(GATES.glob('gate_*/epoch_*_w*.json')):
        images_path = meta_path.with_name(meta_path.stem + '_images.npz')
        if not images_path.exists():
            raise FileNotFoundError(images_path)
        meta = json.loads(meta_path.read_text())
        with np.load(images_path, allow_pickle=False) as pack:
            images, cond = pack['images'], {k: pack[k].astype(int) for k in ('mfg', 'fs')}
        inputs.append({'path': images_path.relative_to(ROOT).as_posix(), 'sha256': digest(images_path)})
        for name, model in models.items():
            for transform, x in [('identity', images), ('vertical_flip', np.flip(images, -2))]:
                probs = probabilities(model, np.ascontiguousarray(x))
                for axis in ('mfg', 'fs'):
                    correct = (probs[axis].argmax(1) == cond[axis]).astype(float)
                    rows.append({'epoch': int(meta['epoch']), 'guidance': float(meta['guidance']), 'judge': name,
                                 'axis': axis, 'transform': transform, 'n': len(correct), 'correct': int(correct.sum()),
                                 'rate': float(correct.mean()), **wilson(correct)})
        print('done', meta_path.stem, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'summary_by_judge.csv', index=False)
    table['family'] = table.judge.str.replace(r'\d$', '', regex=True)
    (table.groupby(['epoch', 'guidance', 'family', 'axis', 'transform'], as_index=False)
     .agg(mean_rate=('rate', 'mean'), min_rate=('rate', 'min'), max_rate=('rate', 'max'))
     .to_csv(out / 'summary_by_family.csv', index=False))
    json_write(out / 'run.json', {'script': 'scripts/v4_review/gate_heldout_check.py', 'script_sha256': digest(Path(__file__)),
                                  'device': device, 'judges': list(models), 'inputs': inputs,
                                  'interval': 'Wilson 95% over the 100 synthetic gate samples; fixed noise seeds shared across epochs'})
    (out / 'README.md').write_text(
        '# B1 gate samples graded by the held-out judges (2026-10-16)\n\n'
        'All files made by `scripts/v4_review/gate_heldout_check.py`. Inputs: every saved t=999 gate sample set of run '
        '`b1_20261016_01` (epochs 10-110, w=3 and w=5; 100 samples each, 25 per vendor x field-strength cell), from the local '
        'fetch mirror `reports/v4_diffusion/b1_20261016_01/remote/full/gate_*/` (hashes in `run.json`). Judges: the three '
        'original robust judges and the six B2 held-out judges (`heldout_orientation_robust0-2`, `heldout_legacy_comparison0-2`). '
        'Each sample set is graded as generated and after a vertical flip, with the requested condition as the label.\n\n'
        '| File | Content |\n|---|---|\n'
        '| `summary_by_judge.csv` | Correct count, rate and Wilson 95% interval per gate, judge, axis and transform |\n'
        '| `summary_by_family.csv` | Mean, minimum and maximum over the judges of each family |\n'
        '| `run.json` | Script and input hashes, judges, device |\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    main(parser.parse_args().run_id)
