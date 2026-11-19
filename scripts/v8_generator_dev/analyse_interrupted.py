"""Analyse validated matched checkpoints without labelling interrupted training complete."""
import argparse
import json
import re

import pandas as pd

from scripts.v8_generator_dev.analyse_full import compare_rows, primary_rows
from scripts.v8_generator_dev.failed_pair import select_failed_pair
from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def analyse(training_fetch, validation_fetch, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new analysis ID required')
    training_fetch, validation_fetch = map(project_path, (training_fetch, validation_fetch))
    failure_path = training_fetch / 'root/failed.json'
    log_path = training_fetch / 'root/train.txt'
    selection = select_failed_pair(training_fetch / 'training', json.loads(failure_path.read_text()), log_path.read_text())
    training_protocol = training_fetch / 'training/protocol.json'
    config = json.loads(training_protocol.read_text())['config']
    fetched = json.loads((validation_fetch / 'complete.json').read_text())
    homes = list((validation_fetch / 'root').glob('*/protocol.json'))
    if len(homes) != 1:
        raise ValueError('Expected exactly one recovery protocol')
    recovery = json.loads(homes[0].read_text())
    completed = json.loads(homes[0].with_name('complete.json').read_text())
    if not fetched['validation_retrieved'] or not fetched['checkpoint_hashes_verified']:
        raise ValueError('Both validations and verified checkpoints required')
    if recovery['selection']['checkpoint_record'] != selection['checkpoint_record']:
        raise ValueError('Recovery did not select the original last matched pair')
    if completed['full_training_completed'] or completed['converged']:
        raise ValueError('Interrupted training cannot be promoted')
    inputs = {path.relative_to(ROOT).as_posix(): digest(path)
              for path in (failure_path, log_path, training_protocol, homes[0], homes[0].with_name('complete.json'))}
    frames = []
    for variant in ('additive', 'scale_shift'):
        folder = project_path(fetched['evaluations'][variant])
        protocol = json.loads((folder / 'protocol.json').read_text())
        complete = json.loads((folder / 'complete.json').read_text())
        record = selection['checkpoint_record']
        if not complete['complete'] or complete['engineering'] or protocol['config'] != config:
            raise ValueError('Incomplete or incompatible validation')
        if protocol['checkpoint'].replace('\\', '/') != record['checkpoints'][variant].replace('\\', '/'):
            raise ValueError('Evaluation used another checkpoint')
        if protocol['checkpoint_sha256'] != record['checkpoint_hashes'][variant]:
            raise ValueError('Evaluated checkpoint hash mismatch')
        if digest(project_path(fetched['checkpoints'][variant])) != record['checkpoint_hashes'][variant]:
            raise ValueError('Retrieved checkpoint hash mismatch')
        frames.append(primary_rows(read_grading_csv(folder / 'per_image.csv'), config))
        for filename in ('protocol.json', 'complete.json', 'per_image.csv', 'selection.json'):
            path = folder / filename
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    csv_write(output / 'paired_primary_differences.csv', compare_rows(*frames, config))
    rows = []
    for variant, frame in zip(('additive', 'scale_shift'), frames):
        for (axis, judge, control), group in frame.groupby(['axis', 'judge', 'variant']):
            strata = [('all_flagged_and_in_range', group)]
            strata += [(f'support:{name}', part) for name, part in group.groupby('condition_support_status')]
            strata += [(f'direction:{name}', part) for name, part in group.groupby('direction')]
            for stratum, part in strata:
                for metric in ('target_hit', 'paired_hit_delta', 'ssim', 'mad', 'other_axis_stable'):
                    rows.append(dict(model=variant, axis=axis, judge=judge, control=control,
                                     stratum=stratum, metric=metric,
                                     **bootstrap_patients(part, metric, config['evaluation']['bootstrap_seed'],
                                                          config['evaluation']['bootstrap_resamples'])))
    csv_write(output / 'primary_estimates.csv', pd.DataFrame(rows))
    json_write(output / 'protocol.json', dict(selection=selection, config=config,
                                              training_fetch=str(training_fetch.relative_to(ROOT)),
                                              validation_fetch=str(validation_fetch.relative_to(ROOT))))
    inputs['scripts/v8_generator_dev/analyse_interrupted.py'] = digest(__file__)
    inputs['scripts/v8_generator_dev/analyse_full.py'] = digest(ROOT / 'scripts/v8_generator_dev/analyse_full.py')
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, analysis_completed=True,
                                              full_training_completed=False, converged=False,
                                              common_epoch=selection['common_epoch'], winner_selected=False,
                                              anatomy_validated=False, test_pixels_used=False))
    (output / 'README.md').write_text(
        '# Numerically interrupted matched-pair comparison\n\nProducer: `python -m scripts.v8_generator_dev.analyse_interrupted`.\n'
        'protocol.json retains failure-based selection and frozen settings; inputs.json binds raw evidence. '
        'primary_estimates.csv gives each named judge/control/support/direction estimate with patient CIs. '
        'paired_primary_differences.csv gives matched scale-shift-minus-additive contrasts with CIs. '
        'complete.json means analysis completed, never completed training or validated anatomy.\n', encoding='utf-8')
    print('INTERRUPTED-PAIR ANALYSIS', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training-fetch', required=True)
    parser.add_argument('--validation-fetch', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    analyse(args.training_fetch, args.validation_fetch, args.run_id)
