"""Render an evidence-linked interrupted-pair report, controls and unselected panels."""
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.failed_pair import allowance
from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def interval(row, scale=100, suffix='%', precision=1):
    return (f'{row["mean"]*scale:.{precision}f}{suffix} '
            f'[{row["ci95_lo"]*scale:.{precision}f}, {row["ci95_hi"]*scale:.{precision}f}]')


def report(analysis, fetched, verified, launch, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new report ID required')
    analysis, fetched, verified, launch = map(project_path, (analysis, fetched, verified, launch))
    if not json.loads((verified / 'complete.json').read_text())['saved_validation_verified']:
        raise ValueError('Saved-array grading verification must pass first')
    protocol = json.loads((analysis / 'protocol.json').read_text())
    config = protocol['config']
    retrieval = json.loads((fetched / 'complete.json').read_text())
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    estimates = pd.read_csv(analysis / 'primary_estimates.csv', keep_default_na=False)
    differences = pd.read_csv(analysis / 'paired_primary_differences.csv', keep_default_na=False)
    homes = {model: project_path(path) for model, path in retrieval['evaluations'].items()}
    tables = {model: read_grading_csv(home / 'per_image.csv') for model, home in homes.items()}
    requests = {model: pd.read_csv(home / 'requests.csv') for model, home in homes.items()}
    selections = {model: json.loads((home / 'selection.json').read_text()) for model, home in homes.items()}
    prior = json.loads((launch / 'allowance.json').read_text())
    process = json.loads((launch / 'process_result.json').read_text())
    if process['exit_code'] != 0:
        raise ValueError('Final validation did not succeed')
    budget = allowance([*prior['launch_seconds'], process['seconds']], prior['failure_seconds'])
    budget['scope'] = 'Final conservative ledger for the original US$6; no new allowance or exact invoice claim'
    budget['last_validation_launch_seconds'] = process['seconds']
    json_write(output / 'budget.json', budget)

    confirmation = []
    for model, table in tables.items():
        for axis, selection in selections[model].items():
            point = selection['selected']
            if point is None:
                continue
            frame = table[(table.axis == axis) & (table.timestep == point['timestep']) &
                          (table.guidance == point['guidance']) &
                          table.generation_seed.isin(config['evaluation']['confirmation_generation_seeds'])]
            for (judge, control), group in frame.groupby(['judge', 'variant']):
                for label, part in [('all_flagged_and_in_range', group),
                    ('selection_eligible_support', group[group.condition_support_status.isin(config['evaluation']['selection_support_statuses'])])]:
                    for metric in ('target_hit', 'paired_hit_delta', 'ssim', 'other_axis_stable'):
                        confirmation.append(dict(model=model, axis=axis, judge=judge, control=control,
                            stratum=label, metric=metric, timestep=point['timestep'], guidance=point['guidance'],
                            **bootstrap_patients(part, metric, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])))
    confirmation = pd.DataFrame(confirmation)
    csv_write(output / 'selected_confirmation_estimates.csv', confirmation)

    # Predeclared source ordering and directions only; never rank by output or judge score.
    panel_records = []
    fig, axes = plt.subplots(4, 7, figsize=(14, 9), constrained_layout=True)
    columns = ['Source', 'Additive null', 'Additive target', '|target-source|',
               'Scale-shift null', 'Scale-shift target', '|target-source|']
    for ax, label in zip(axes[0], columns):
        ax.set_title(label, fontsize=10)
    row_index = 0
    for axis in ('manufacturer', 'field_strength'):
        point = config['evaluation']['fixed_settings'][axis]
        candidates = requests['additive']
        candidates = candidates[(candidates.axis == axis) & (candidates.timestep == point['timestep']) &
            (candidates.guidance == point['guidance']) & (candidates.generation_seed == config['evaluation']['selection_generation_seed'])]
        for direction in ('0_to_1', '1_to_0'):
            first = candidates[candidates.direction == direction].sort_values('case').iloc[0]
            pair = requests['scale_shift']
            match = pair[(pair.dataset_index == first.dataset_index) & (pair.axis == axis) &
                (pair.timestep == point['timestep']) & (pair.guidance == point['guidance']) &
                (pair.generation_seed == first.generation_seed)].iloc[0]
            with np.load(homes['additive'] / first.array_file, allow_pickle=False) as arrays:
                source, null0, target0 = [arrays[key] for key in ('source', 'null', 'target')]
            with np.load(homes['scale_shift'] / match.array_file, allow_pickle=False) as arrays:
                null1, target1 = arrays['null'], arrays['target']
            images = [source, null0, target0, np.abs(target0-source), null1, target1, np.abs(target1-source)]
            for j, image in enumerate(images):
                axes[row_index, j].imshow(image, cmap='magma' if j in (3, 6) else 'gray', vmin=0, vmax=1)
                axes[row_index, j].set_xticks([])
                axes[row_index, j].set_yticks([])
            axes[row_index, 0].set_ylabel(f'{axis.replace("_", " ")} {direction}\n{first.patient_id}\n{first.condition_support_status}', fontsize=8)
            panel_records.append(dict(row=row_index, axis=axis, direction=direction,
                patient_id=first.patient_id, dataset_index=int(first.dataset_index),
                additive_case=int(first.case), scale_shift_case=int(match.case),
                support=first.condition_support_status, timestep=int(first.timestep),
                guidance=float(first.guidance), generation_seed=int(first.generation_seed)))
            row_index += 1
    fig.suptitle('Epoch 70 development examples: first frozen source per axis/direction; anatomy unassessed\n'
                 'Canonical arrays, common intensity range [0,1]; absolute differences share [0,1]', fontsize=12)
    fig.savefig(output / 'fixed_examples.png', dpi=160)
    plt.close(fig)
    csv_write(output / 'panel_selection.csv', pd.DataFrame(panel_records))
    forest = differences[(differences.stratum == 'all_flagged_and_in_range') &
                         (differences.control == 'target') & (differences.metric == 'paired_hit_delta')]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    y = np.arange(len(forest))
    ax.errorbar(forest['mean']*100, y,
        xerr=np.array([forest['mean']-forest.ci95_lo, forest.ci95_hi-forest['mean']])*100,
        fmt='o', color='#175d84', capsize=4)
    ax.set_yticks(y, [f'{row.axis.replace("_", " ")}, robust judge seed {row.judge[-1]}' for row in forest.itertuples()])
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Scale-shift minus additive target-minus-null hit difference (percentage points)')
    ax.set_title('Matched epoch-70 fixed settings: patient-bootstrap 95% intervals\n15 validation patients; 3 generation seeds averaged within patient')
    ax.invert_yaxis()
    fig.savefig(output / 'paired_differences.png', dpi=160)
    plt.close(fig)

    lines = ['# Epoch-70 generator development result', '',
        'Validation and saved-array verification completed. Training stopped on a nonfinite gradient; '
        'full training, convergence and anatomical preservation remain unestablished. No winning backbone is selected.', '',
        'The paired primary comparison uses the frozen manufacturer t=300, w=3 and field-strength t=100, w=5, '
        '15 validation patients, one slice each and three generation seeds. These development results include '
        'outside-range conditions; support and direction strata remain separate in the source CSVs. '
        'They are not directly comparable with the old paper\'s test results. All intervals below are '
        '95% equal-patient bootstrap intervals with generation seeds averaged within patient.', '',
        '## Frozen fixed-point target and null controls', '',
        '| Axis | Orientation-robust judge | Model | Target hit, % [95% CI] | Null hit, % [95% CI] | Target-minus-null, pp [95% CI] |',
        '|---|---|---|---|---|---|']
    def row_value(model, axis, judge, control, metric):
        return estimates[(estimates.model == model) & (estimates.axis == axis) & (estimates.judge == judge) &
            (estimates.control == control) & (estimates.metric == metric) &
            (estimates.stratum == 'all_flagged_and_in_range')].iloc[0]
    for axis in ('manufacturer', 'field_strength'):
        for seed in range(3):
            judge = f'heldout_orientation_robust_seed{seed}'
            for model in ('additive', 'scale_shift'):
                values = [interval(row_value(model, axis, judge, control, metric), suffix='')
                    for control, metric in [('target', 'target_hit'), ('null', 'target_hit'), ('target', 'paired_hit_delta')]]
                lines.append(f'| {axis} | seed {seed} | {model} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Interpretation and next work', '',
        '- Scale-and-shift has modest positive fixed-point field-strength contrasts and slightly negative manufacturer '
        'contrasts. All named-judge paired target-minus-null difference intervals include zero; this does not '
        'establish an architecture improvement or equivalence.',
        '- Field-strength setting selection remains infeasible: only one eligible 0-to-1 patient, '
        'below the frozen minimum of two. Do not lower that minimum or replace the fixed point using flagged cases.',
        '- Manufacturer provisional settings differ: additive t=300, w=5; scale-shift t=300, w=3. '
        'The additional additive confirmation cases explain 210 versus 180 total requests. '
        'The primary fixed-point sources/settings/seeds are fully matched. '
        '`selected_confirmation_estimates.csv` retains confirmation-only outcomes, including eligible-support strata; '
        'these reuse the validation patients and are not independent cohort replication.',
        '- Target SSIM and difference maps measure correspondence, not breast or tumour preservation. '
        'Independent preservation assessment, source-guide revision and adapter efficacy remain open.',
        '- Next numerical work is an attempt to reproduce the failed block from the last complete checkpoint '
        'under the original pinned runtime, retaining the failed draw and pre-step state if it recurs, followed by a matched '
        'precision-policy comparison. The historical failing draw was not saved. The separately prepared AMP retry policy is locally tested; it has not '
        'fixed or completed this historical training run. Any further paid training needs a new budget decision.',
        '- Improve field-strength support by auditing existing train/validation metadata and feasible fixed-EchoTime '
        'target cells before proposing a new cohort. New downloads or data access need separate approval.',
        '- Keep both epoch-70 checkpoints as interrupted development candidates. Do not switch the demo, paper '
        'numbers or final test protocol based on this comparison.', '',
        '## Correspondence and scope', '',
        '| Axis | Model | Target SSIM [95% CI] |', '|---|---|---|']
    for axis in ('manufacturer', 'field_strength'):
        for model in ('additive', 'scale_shift'):
            lines.append(f'| {axis} | {model} | {interval(row_value(model, axis, "heldout_orientation_robust_seed0", "target", "ssim"), scale=1, suffix="", precision=3)} |')
    lines += ['', 'SSIM is identical across judges because the same arrays are scored. Full precision is in the source CSV.', '',
        '![Matched paired differences](paired_differences.png)', '', '![Unselected fixed examples](fixed_examples.png)', '',
        'Panels use the first frozen request per axis/direction at the fixed point and selection seed, '
        'without filtering by image appearance or judge outcome. `panel_selection.csv` identifies each source and case.', '',
        '## Compute accounting', '',
        f'The conservative ledger estimates US${budget["estimated_prior_usd"]:.2f} including all dispatch/recovery '
        f'wall times, the existing rate margin and shared reserve, against the single approved US$6. '
        f'The rounded estimated remainder is US${budget["remaining_usd"]:.2f}; the exact invoice is unknown. '
        'No new cloud execution is launched by this report.', '', '## Evidence', '',
        f'- Paired primary estimates and differences: `{analysis.relative_to(ROOT).as_posix()}/`.',
        f'- Raw requests, controls, arrays and receipts: `{fetched.relative_to(ROOT).as_posix()}/`.',
        f'- All saved grades and frozen inputs verified: `{verified.relative_to(ROOT).as_posix()}/`.',
        '- `inputs.json` hashes every input used here; `budget.json` retains accounting assumptions.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'README.md').write_text(
        '# Interrupted generator result report\n\nProducer: `python -m scripts.v8_generator_dev.report_interrupted`.\n'
        'RESULTS.md interprets source CSVs; selected_confirmation_estimates.csv retains all provisional-setting '
        'confirmation-only judge/control/support estimates. fixed_examples.png and panel_selection.csv retain '
        'outcome-independent sample choices; paired_differences.png plots named-judge patient intervals. '
        'budget.json retains the original allowance ledger. inputs.json hashes evidence and script. '
        'complete.json means report creation, not completed training or anatomy approval.\n', encoding='utf-8')
    sources = [analysis / 'primary_estimates.csv', analysis / 'paired_primary_differences.csv', analysis / 'protocol.json',
               fetched / 'complete.json', verified / 'complete.json', launch / 'allowance.json', launch / 'process_result.json']
    for home in homes.values():
        sources += [home / name for name in ('requests.csv', 'per_image.csv', 'selection.json')]
        sources += list((home / 'arrays').glob('*.npz'))
    inputs = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    inputs['scripts/v8_generator_dev/report_interrupted.py'] = digest(__file__)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, report_created=True,
        common_epoch=70, winner_selected=False, full_training_completed=False, converged=False,
        anatomy_validated=False, test_pixels_used=False, new_cloud_execution_launched=False))
    print('REPORT CREATED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('analysis', 'fetched', 'verified', 'launch', 'run-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    report(args.analysis, args.fetched, args.verified, args.launch, args.run_id)
