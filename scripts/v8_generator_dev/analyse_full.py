"""Compare common-epoch R0/R1 validation at the frozen fixed operating points."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def primary_rows(frame, config):
    selected = []
    for axis, point in config['evaluation']['fixed_settings'].items():
        group = frame[(frame.axis == axis) & (frame.timestep == point['timestep']) &
                      np.isclose(frame.guidance, point['guidance'])].copy()
        expected = {config['evaluation']['selection_generation_seed'],
                    *config['evaluation']['confirmation_generation_seeds']}
        if set(group.generation_seed) != expected:
            raise ValueError('Fixed-point evaluation is missing a declared generation seed')
        keys = ['patient_id', 'generation_seed', 'judge', 'variant']
        if group.duplicated(keys).any():
            raise ValueError('Duplicate fixed-point grading rows')
        selected.append(group)
    return pd.concat(selected, ignore_index=True)


def compare_rows(additive, scale_shift, config):
    keys = ['patient_id', 'dataset_index', 'axis', 'direction', 'generation_seed', 'judge', 'variant',
            'timestep', 'guidance', 'condition_support_status']
    joined = additive.merge(scale_shift, on=keys, how='outer', suffixes=('_r0', '_r1'),
                              validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('R0/R1 fixed-point sources/settings/judges/support do not match')
    output = []
    metrics = ['target_hit', 'paired_hit_delta', 'ssim', 'mad', 'other_axis_stable']
    for (axis, judge, variant), group in joined.groupby(['axis', 'judge', 'variant']):
        strata = [('all_flagged_and_in_range', group)]
        strata += [(f'direction:{direction}', part) for direction, part in group.groupby('direction')]
        strata += [(f'support:{status}', part) for status, part in group.groupby('condition_support_status')]
        for stratum, part in strata:
            for metric in metrics:
                values = part[['patient_id']].copy()
                values['difference'] = part[f'{metric}_r1'] - part[f'{metric}_r0']
                result = bootstrap_patients(values, 'difference', config['evaluation']['bootstrap_seed'],
                                              config['evaluation']['bootstrap_resamples'])
                output.append(dict(axis=axis, judge=judge, control=variant, stratum=stratum,
                                   metric=metric, contrast='scale_shift_minus_additive', **result,
                                   interval='Patient bootstrap; average generation seeds within patient',
                                   anatomy_validated=False))
    return pd.DataFrame(output)


def analyse(training, additive, scale_shift, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid analysis ID')
    folders = [project_path(value) for value in (training, additive, scale_shift)]
    trained = json.loads((folders[0] / 'complete.json').read_text())
    if not trained['complete'] or trained['engineering']:
        raise ValueError('A completed non-engineering paired training result is required')
    if trained['progress']['additive'] != trained['progress']['scale_shift']:
        raise ValueError('Primary model training budgets do not match')
    config = json.loads((folders[0] / 'protocol.json').read_text())['config']
    frames, inputs = [], {}
    for variant, folder in zip(('additive', 'scale_shift'), folders[1:]):
        complete = json.loads((folder / 'complete.json').read_text())
        protocol = json.loads((folder / 'protocol.json').read_text())
        if not complete['complete'] or complete['engineering']:
            raise ValueError('Completed full development validation is required')
        if protocol['checkpoint'].replace('\\', '/') != trained['checkpoints'][variant].replace('\\', '/'):
            raise ValueError('Evaluation does not use the common-epoch primary checkpoint')
        if protocol['config'] != config:
            raise ValueError('Training/evaluation protocols differ')
        frames.append(primary_rows(read_grading_csv(folder / 'per_image.csv'), config))
    for folder in folders:
        for name in ('protocol.json', 'complete.json', 'per_image.csv', 'selection.json'):
            path = folder / name
            if path.exists():
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    source = ROOT / 'scripts/v8_generator_dev/analyse_full.py'
    inputs[source.relative_to(ROOT).as_posix()] = digest(source)
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new analysis ID')
    output.mkdir(parents=True)
    json_write(output / 'protocol.json', dict(training=str(folders[0].relative_to(ROOT)),
                                              evaluations=[str(p.relative_to(ROOT)) for p in folders[1:]],
                                              primary='Frozen common-epoch checkpoints and fixed operating points; all declared seeds',
                                              interpretation='Development comparison only, all ranges flagged; no clinical preservation or fresh-test claim'))
    table = compare_rows(*frames, config)
    csv_write(output / 'paired_primary_differences.csv', table)
    rates = []
    for variant, frame in zip(('additive', 'scale_shift'), frames):
        for (axis, judge, control), group in frame.groupby(['axis', 'judge', 'variant']):
            for stratum, part in [('all_flagged_and_in_range', group),
                                   *[(f'support:{name}', subset) for name, subset in group.groupby('condition_support_status')],
                                   *[(f'direction:{name}', subset) for name, subset in group.groupby('direction')]]:
                for metric in ('target_hit', 'paired_hit_delta', 'ssim', 'mad', 'other_axis_stable'):
                    rates.append(dict(model=variant, axis=axis, judge=judge, control=control,
                                      stratum=stratum, metric=metric,
                                      **bootstrap_patients(part, metric, config['evaluation']['bootstrap_seed'],
                                                           config['evaluation']['bootstrap_resamples']),
                                      anatomy_validated=False))
    csv_write(output / 'primary_estimates.csv', pd.DataFrame(rates))
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, common_progress=trained['progress']['additive'],
                                              fixed_points=config['evaluation']['fixed_settings'],
                                              named_judges=sorted(frames[0].judge.unique()),
                                              clinical_preservation='unknown', anatomy_validated=False,
                                              test_pixels_used=False, winner_selected=False,
                                              caveat='Classifier target hits and SSIM are not anatomy or physical scanner validation'))
    (output / 'README.md').write_text(
        '# Common-epoch fixed-point generator comparison\n\n'
        'Producer: `python -m scripts.v8_generator_dev.analyse_full`.\n'
        'protocol.json fixes the matched model/evaluation roles. primary_estimates.csv gives every '
        'model/judge/control rate or correspondence metric with patient-bootstrap CIs; '
        'paired_primary_differences.csv gives R1-minus-R0 contrasts on matched sources/seeds. '
        'All ranges are flagged and support/direction strata remain separate. inputs.json binds '
        'raw results/code; complete.json records scope and unknown preservation. '
        'No winner, causal scanner effect, fresh-test generalisation or anatomical validity is inferred.\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--additive', required=True)
    parser.add_argument('--scale-shift', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(analyse(args.training, args.additive, args.scale_shift, args.run_id))
