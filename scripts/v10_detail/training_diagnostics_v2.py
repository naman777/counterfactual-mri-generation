"""Descriptive training-journal diagnostics, separate from editing efficacy."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def diagnose(training, checked, run_id):
    training, checked = map(project_path, (training, checked))
    status = json.loads((checked / 'complete.json').read_text())
    if not status['training_components_verified']:
        raise ValueError('Independent completed-training verification required')
    verify_inputs(json.loads((checked / 'inputs.json').read_text()))
    output = ROOT / 'reports/v10_detail' / run_id
    output.mkdir(parents=True, exist_ok=False)
    records = []
    inputs = {}
    for path in sorted(training.glob('block_*/*.jsonl')):
        records.extend(json.loads(s) for s in path.read_text().splitlines())
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    frame = pd.DataFrame(records)
    if len(frame) != 25020 or not frame.applied.all():
        raise ValueError('Full applied-update journals required')
    keys = ['training_seed', 'mode']
    summaries = []
    for key, part in frame.groupby(keys):
        part = part.sort_values('global_step')
        summaries.append(dict(zip(keys, key), applied_updates=len(part),
            total_amp_retries=int(part.amp_retries.sum()), updates_with_amp_retry=int(part.amp_retries.gt(0).sum()),
            max_amp_retries=int(part.amp_retries.max()), min_amp_scale=float(part.amp_scale_after.min()),
            max_gradient_norm=float(part.gradient_norm.max()),
            gradient_norm_p99=float(part.gradient_norm.quantile(.99)),
            mean_total_loss=float(part.total_loss.mean()), first_epoch_loss=float(part.loc[part.epoch_at_draw.eq(0), 'total_loss'].mean()),
            final_epoch_loss=float(part.loc[part.epoch_at_draw.eq(19), 'total_loss'].mean())))
    csv_write(output / 'per_copy.csv', pd.DataFrame(summaries))
    epochs = frame.groupby([*keys, 'epoch_at_draw'])[
        ['total_loss', 'noise_mse', 'coarse_structure_loss', 'gradient_norm', 'amp_retries']].mean().reset_index()
    csv_write(output / 'epoch_means.csv', epochs)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    modes = ('coarse_control', 'coarse_fine', 'learned_zero')
    colors = ('#496a8a', '#b45434', '#777777')
    metrics = ('total_loss', 'coarse_structure_loss', 'gradient_norm')
    for col, seed in enumerate(sorted(frame.training_seed.unique())):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for mode, color in zip(modes, colors):
                part = epochs[(epochs.training_seed == seed) & (epochs['mode'] == mode)]
                ax.plot(part.epoch_at_draw + 1, part[metric], label=mode, color=color)
            ax.set_title(f'Training seed {seed}\n{metric.replace("_", " ")}', fontsize=10)
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
            if row == 0:
                ax.legend(fontsize=7)
    fig.suptitle('All matched adapter training journals\nDescriptive epoch means; training loss does not establish editing efficacy or clinical preservation')
    fig.savefig(output / 'training_curves.png', dpi=150)
    plt.close(fig)
    for path in (project_path(__file__), checked / 'complete.json', checked / 'inputs.json'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, logged_updates=len(frame), copies=len(summaries),
        descriptive_training_only=True, efficacy_established=False, clinical_preservation='unknown', paid_gpu_calls=0))
    (output / 'README.md').write_text(
        'Producer: scripts.v10_detail.training_diagnostics_v2. per_copy.csv: loss, AMP-retry and gradient summaries for every copy; '
        'epoch_means.csv: all epoch means; training_curves.png: matched descriptive curves; inputs.json: immutable '
        'journals and prior full-training audit; complete.json: training-only scope. No validation selection, '
        'editing efficacy, between-mode superiority or clinical preservation is inferred from these losses.\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ('training', 'checked', 'run-id'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    diagnose(a.training, a.checked, a.run_id)
