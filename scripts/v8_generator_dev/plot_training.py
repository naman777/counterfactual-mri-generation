"""Plot actual matched-epoch validation records, without extrapolated curves."""
import argparse
import json
import re

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def plot(source, run_id, live=False):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid plot ID')
    source = project_path(source)
    inputs, rows = {}, []
    if live:
        path = sorted(source.glob('snapshot_*.json'))[-1]
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        records = json.loads(path.read_text())['validation_records']
    else:
        records = []
        for path in sorted(source.glob('block_*/*_validation.json')):
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            records.append(dict(path=str(path), value=json.loads(path.read_text())))
    for record in records:
        name = record['path'].replace('\\', '/').rsplit('/', 1)[-1].removesuffix('_validation.json')
        rows.append(dict(variant=name, source_record=record['path'], **record['value']))
    frame = pd.DataFrame(rows)
    if set(frame.variant) != {'additive', 'scale_shift'}:
        raise ValueError('Both variants required')
    common = set.intersection(*[set(group.epoch) for _, group in frame.groupby('variant')])
    frame = frame[frame.epoch.isin(common)].sort_values(['variant', 'epoch'])
    if frame.empty:
        raise ValueError('No common epoch')
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new plot ID')
    output.mkdir(parents=True)
    csv_write(output / 'validation_curve.csv', frame)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for variant, color, label in [('additive', '#315a87', 'R0 · additive conditioning'),
                                   ('scale_shift', '#b5622c', 'R1 · adaptive normalisation')]:
        group = frame[frame.variant == variant]
        ax.plot(group.epoch, group.patient_macro_noise_mse, marker='o', markersize=4,
                linewidth=1.8, color=color, label=label)
    ax.set(xlabel='Completed common training epoch', ylabel='Patient-macro validation noise MSE',
           title='Canonical-generator training', yscale='log')
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(output / 'validation_curve.png', dpi=160)
    plt.close(fig)
    inputs['scripts/v8_generator_dev/plot_training.py'] = digest(__file__)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, latest_common_epoch=int(frame.epoch.max()),
                                              actual_validation_records_only=True, editing_efficacy_established=False))
    (output / 'README.md').write_text(
        '# Matched training-curve snapshot\n\nProducer: `python -m scripts.v8_generator_dev.plot_training`.\n'
        'validation_curve.csv retains each actual source record and noise losses. validation_curve.png '
        'plots only epochs available for both models; inputs.json binds records/code; complete.json '
        'records the latest common epoch. Noise prediction is separate from target recognition and '
        'anatomical preservation; no extrapolation or winner selection.\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    print(plot(args.source, args.run_id, args.live))
