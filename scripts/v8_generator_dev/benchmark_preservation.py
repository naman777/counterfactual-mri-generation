"""Train-calibrated automated structural screen on held-out controls and saved edits."""
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import source_native
from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.preservation_screen import METRICS, GRID_CONTROLS, DAMAGE_CONTROLS, measurements, controls, screen


def benchmark(fetched, verified, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new benchmark ID required')
    fetched, verified = map(project_path, (fetched, verified))
    if not json.loads((verified / 'complete.json').read_text())['saved_validation_verified']:
        raise ValueError('Saved validation must be independently recomputed first')
    verify_inputs(json.loads((verified / 'inputs.json').read_text()))
    receipt = json.loads((fetched / 'complete.json').read_text())
    homes = {model: project_path(path) for model, path in receipt['evaluations'].items()}
    config = json.loads((homes['additive'] / 'protocol.json').read_text())['config']
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Automated preservation benchmark\n\nProducer: `python -m scripts.v8_generator_dev.benchmark_preservation`.\n'
        'protocol.json freezes scope before measurements. calibration.csv and thresholds.json use training '
        'patients only. heldout_controls.csv/control_intervals.csv test grid-preserving transforms and '
        'injected damage on validation patients. edit_metrics.csv/edit_intervals.csv retain every fixed-point '
        'identity/null/target result and missingness. joint_judge_proxy.csv combines named classifier decisions '
        'with the proxy envelope, never verified anatomy. control_detection.png visualises all controls. '
        'RESULTS.md explains sensitivity and limits. inputs.json binds code/data/arrays. complete.json '
        'means automated benchmark completion; anatomy and lesion validation remain unknown.\n', encoding='utf-8')
    json_write(output / 'protocol.json', dict(view=config['view'], training_selection_seed=482946,
        one_slice_per_training_patient=True, validation_sources='All original frozen validation patients',
        metrics=list(METRICS), rank_radius=2, rank_patch_size=8, edge_sigma=1, edge_quantile=.8,
        threshold_rule='95th percentile across training-patient maxima over the six grid-preserving controls; any metric exceedance flags',
        grid_preserving_controls=list(GRID_CONTROLS), damage_controls=list(DAMAGE_CONTROLS),
        caution='Grid-preserving contrast/noise/blur controls are not guaranteed clinically harmless; synthetic damage is not real pathology ground truth',
        scoring_frame='source-native inverse geometry; no registration or resizing',
        image_size=128, threshold_retuning_on_validation=False, judge_gradients_used=False,
        test_pixels_used=False, clinical_anatomy_validated=False, lesion_preservation='unknown'))
    datasets = {split: CanonicalDataset(config['view'], split) for split in ('train', 'val')}
    inputs = {str((fetched / 'complete.json').relative_to(ROOT)): digest(fetched / 'complete.json'),
              str((verified / 'complete.json').relative_to(ROOT)): digest(verified / 'complete.json')}
    for name in ('benchmark_preservation.py', 'preservation_screen.py', 'test_preservation_screen.py'):
        path = ROOT / 'scripts/v8_generator_dev' / name
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    for name in ('scripts/v5_anatomy/metrics.py', 'scripts/v7_geometry/orientation.py',
                 'scripts/v8_generator_dev/data.py', 'scripts/v8_generator_dev/full_protocol.py'):
        inputs[name] = digest(ROOT / name)
    for split in ('train', 'val'):
        for path in (ROOT / config['view'] / 'splits' / f'{split}.csv',
                     ROOT / 'data/processed/ax_t1_128/splits' / f'{split}.csv'):
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    request_tables = {model: pd.read_csv(home / 'requests.csv') for model, home in homes.items()}
    indices = sorted(request_tables['additive'].dataset_index.unique())
    chosen = datasets['train'].frame.sample(frac=1, random_state=482946).drop_duplicates('patient_id')
    calibration, heldout, native_sources = [], [], {}
    for split, selection in [('train', chosen.index.tolist()), ('val', indices)]:
        dataset = datasets[split]
        for index in selection:
            row = dataset.frame.iloc[index]
            source = source_native(dataset.image_01(index)[None], dataset.plans[index])[0]
            inputs[project_path(row.source_image_path).relative_to(ROOT).as_posix()] = row.source_array_sha256
            if split == 'val':
                native_sources[index] = source
            transformed = controls(source, 482946 + int(index))
            names = GRID_CONTROLS if split == 'train' else ('identity', *GRID_CONTROLS, *DAMAGE_CONTROLS)
            for name in names:
                image = source if name == 'identity' else transformed[name]
                record = dict(patient_id=row.patient_id, dataset_index=int(index), split=split,
                    control=name, control_family='injected_damage' if name in DAMAGE_CONTROLS else 'grid_preserving',
                    **measurements(source, image))
                (calibration if split == 'train' else heldout).append(record)
        print('PRESERVATION CONTROLS', split, len(selection), 'patients', flush=True)
    calibration = pd.DataFrame(calibration)
    valid = calibration[calibration.screen_measurements_valid]
    if valid.patient_id.nunique() < 10:
        raise ValueError('Insufficient valid calibration patients')
    maxima = valid.groupby('patient_id')[list(METRICS)].max()
    thresholds = {name: max(1e-6, float(maxima[name].quantile(.95))) for name in METRICS}
    json_write(output / 'thresholds.json', dict(thresholds=thresholds, calibration_patients=len(maxima),
        invalid_calibration_rows=int((~calibration.screen_measurements_valid).sum()), calibration_split='train'))
    csv_write(output / 'calibration.csv', calibration)
    heldout = pd.DataFrame([dict(row, **screen(row, thresholds)) for row in heldout])
    csv_write(output / 'heldout_controls.csv', heldout)
    intervals = []
    for control, group in heldout.groupby('control', sort=False):
        for metric in ('screen_flag', *METRICS):
            intervals.append(dict(control=control, metric=metric,
                invalid_rows=int(group[metric].isna().sum()),
                **bootstrap_patients(group, metric, config['evaluation']['bootstrap_seed'])))
    intervals = pd.DataFrame(intervals)
    csv_write(output / 'control_intervals.csv', intervals)
    edit_rows, joint = [], []
    for model, home in homes.items():
        requests = request_tables[model]
        grading = read_grading_csv(home / 'per_image.csv')
        for filename in ('protocol.json', 'requests.csv', 'per_image.csv'):
            inputs[(home / filename).relative_to(ROOT).as_posix()] = digest(home / filename)
        for row in requests.itertuples():
            point = config['evaluation']['fixed_settings'][row.axis]
            if row.timestep != point['timestep'] or row.guidance != point['guidance']:
                continue
            path = home / row.array_file
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            with np.load(path, allow_pickle=False) as arrays:
                images = source_native(np.stack([arrays[key] for key in ('source', 'null', 'target')]),
                                       datasets['val'].plans[row.dataset_index])
            np.testing.assert_array_equal(images[0], native_sources[row.dataset_index])
            for j, control in enumerate(('identity', 'null', 'target')):
                measured = measurements(images[0], images[j])
                result = dict(model=model, patient_id=row.patient_id, case=row.case, axis=row.axis,
                    direction=row.direction, control=control, generation_seed=row.generation_seed,
                    condition_support_status=row.condition_support_status, **measured, **screen(measured, thresholds))
                edit_rows.append(result)
                for grade in grading[(grading.case == row.case) & (grading.variant == control)].itertuples():
                    joint.append(dict(model=model, patient_id=row.patient_id, case=row.case, axis=row.axis,
                        direction=row.direction, control=control, generation_seed=row.generation_seed,
                        condition_support_status=row.condition_support_status, judge=grade.judge, target_hit=grade.target_hit,
                        target_hit_and_proxy_envelope=(int(grade.target_hit and result['within_calibrated_envelope'])
                            if result['within_calibrated_envelope'] is not None else None), anatomy_validated=False))
        print('PRESERVATION EDITS', model, flush=True)
    edits = pd.DataFrame(edit_rows)
    csv_write(output / 'edit_metrics.csv', edits)
    csv_write(output / 'joint_judge_proxy.csv', pd.DataFrame(joint))
    summaries = []
    for (model, axis, control), group in edits.groupby(['model', 'axis', 'control']):
        for stratum, part in [('all_flagged_and_in_range', group),
            *[(f'support:{name}', value) for name, value in group.groupby('condition_support_status')],
            *[(f'direction:{name}', value) for name, value in group.groupby('direction')]]:
            for metric in ('screen_flag', 'within_calibrated_envelope', 'screen_score', 'ssim', *METRICS):
                summaries.append(dict(model=model, axis=axis, control=control, stratum=stratum, metric=metric,
                    invalid_rows=int(part[metric].isna().sum()),
                    **bootstrap_patients(part, metric, config['evaluation']['bootstrap_seed'])))
    csv_write(output / 'edit_intervals.csv', pd.DataFrame(summaries))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    points = intervals[intervals.metric == 'screen_flag']
    x = points['mean'].to_numpy()*100
    ax.errorbar(x, np.arange(len(points)), xerr=np.vstack([x-points.ci95_lo.to_numpy()*100, points.ci95_hi.to_numpy()*100-x]),
                fmt='o', capsize=3, color='#205e83')
    ax.set_yticks(np.arange(len(points)), points.control)
    ax.set_xlim(-3, 103)
    ax.invert_yaxis()
    ax.set_xlabel('Validation control flagged by structural screen, % (95% patient-bootstrap CI)')
    ax.set_title('Train-calibrated screen: all held-out synthetic controls\nFlags measure engineering sensitivity; clinical preservation is unknown')
    fig.savefig(output / 'control_detection.png', dpi=150)
    plt.close(fig)
    lines = ['# Automated preservation benchmark', '',
        'This supplies structural-consistency evidence without a reader. It does not establish breast/tumour '
        'preservation, physical scanner realism or clinical safety. No expert labels were fabricated.', '',
        'Cutoffs use training patients only, then remain fixed for every validation control and saved edit. '
        'All original frozen validation patients and all three fixed-point generation seeds are retained. '
        'The rank/edge measures supplement the earlier SSIM and threshold-mask proxies; these measures '
        'remain acquisition/visibility-sensitive and are not scanner-invariant anatomy representations.', '',
        '| Held-out control | Flagged, % [95% patient CI] |', '|---|---|']
    for row in points.itertuples():
        lines.append(f'| {row.control} | {100*row.mean:.1f} [{100*row.ci95_lo:.1f}, {100*row.ci95_hi:.1f}] |')
    lines += ['', 'Grid-position-preserving controls can still alter clinically relevant visibility. '
        'False flags on those controls and missed injected damage must both be reported. '
        'Small erased patches are artificial sensitivity tests, not annotated lesions. '
        'A within-envelope edit is only a proxy screen result, never a declaration that anatomy survived.', '',
        '`edit_intervals.csv` reports every model/axis/control/support/direction with patient intervals. '
        '`joint_judge_proxy.csv` retains named-judge hits alongside the proxy envelope; the joint outcome '
        'is not reviewed or clinically validated success. No winner or operating point is selected.', '',
        '![Synthetic-control sensitivity](control_detection.png)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, automated_benchmark_completed=True,
        train_calibration_patients=int(len(maxima)), validation_patients=len(indices),
        saved_edit_control_rows=len(edits), clinical_anatomy_validated=False, lesion_preservation='unknown',
        test_pixels_used=False, threshold_retuned_on_validation=False, expert_review_substituted=False))
    print('PRESERVATION BENCHMARK COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched', required=True)
    parser.add_argument('--verified', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    benchmark(args.fetched, args.verified, args.run_id)
