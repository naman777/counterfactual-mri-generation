"""Retrieve interrupted-run checkpoints and optional completed validation, append-only."""
import argparse
import json
import re

from scripts.v8_generator_dev.fetch_full import download
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def fetch(original_run, recovery_id, report_id, checkpoints_only=False, checkpoint_fetch=None):
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (original_run, recovery_id, report_id)):
        raise ValueError('Safe new IDs required')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Interrupted-run evidence retrieval\n\nProducer: `python -m scripts.v8_generator_dev.fetch_failed_validation`.\n'
        'root/ retains original recovery receipts/logs; validation_*/ retains complete raw grading/arrays. '
        'fetch_manifest.json maps remote/local paths and hashes. complete.json describes retrieval only '
        'and retains incomplete training status. No source data or old evidence is replaced.\n', encoding='utf-8')
    remote = f'v8_generator_dev/{recovery_id}'
    records = [download(remote, output / 'root', directory=True)]
    home = output / 'root' / recovery_id
    protocol = json.loads((home / 'protocol.json').read_text())
    if protocol['original_run'] != original_run or protocol['recovery_id'] != recovery_id:
        raise ValueError('Recovery identity mismatch')
    selection = protocol['selection']
    project = f'v8_generator_dev/{original_run}/project'
    checkpoints = {}
    previous = None
    if checkpoint_fetch is not None:
        previous = json.loads((project_path(checkpoint_fetch) / 'complete.json').read_text())
        if not previous['checkpoint_hashes_verified'] or previous['common_epoch'] != selection['common_epoch']:
            raise ValueError('Existing checkpoint retrieval is not the selected matched pair')
    for variant, relative in selection['checkpoint_record']['checkpoints'].items():
        if previous is not None:
            destination = project_path(previous['checkpoints'][variant])
            records.append(dict(kind='reuse_verified_immutable_local_checkpoint',
                                local=str(destination.relative_to(ROOT)), sha256=digest(destination)))
        else:
            destination = ROOT / 'checkpoints/diffusion/ax_t1_128_v8_generator_dev' / report_id / f'{variant}.pt'
            records.append(download(f'{project}/{relative}', destination))
        if digest(destination) != selection['checkpoint_record']['checkpoint_hashes'][variant]:
            raise ValueError('Downloaded checkpoint hash mismatch')
        checkpoints[variant] = str(destination.relative_to(ROOT))
    evaluations = {}
    if not checkpoints_only:
        complete = json.loads((home / 'complete.json').read_text())
        if not complete['complete'] or set(complete['validation_completed']) != {'additive', 'scale_shift'}:
            raise ValueError('Both validations must complete before paired retrieval')
        for variant in ('additive', 'scale_shift'):
            record = download(f'{project}/reports/v8_generator_dev/{recovery_id}_validation_{variant}',
                              output / f'validation_{variant}', directory=True)
            records.append(record)
            evaluations[variant] = record['content_root']
    json_write(output / 'fetch_manifest.json', records)
    json_write(output / 'complete.json', dict(complete=True, checkpoints=checkpoints,
                                              evaluations=evaluations, checkpoint_hashes_verified=True,
                                              common_epoch=selection['common_epoch'],
                                              full_training_completed=False, converged=False,
                                              anatomy_validated=False, validation_retrieved=not checkpoints_only))
    print('FETCHED', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--original-run', required=True)
    parser.add_argument('--recovery-id', required=True)
    parser.add_argument('--report-id', required=True)
    parser.add_argument('--checkpoints-only', action='store_true')
    parser.add_argument('--checkpoint-fetch')
    args = parser.parse_args()
    fetch(args.original_run, args.recovery_id, args.report_id, args.checkpoints_only, args.checkpoint_fetch)
