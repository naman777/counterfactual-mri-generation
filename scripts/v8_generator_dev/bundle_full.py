"""Build a local, hash-verified train/val-only cloud input archive; no upload."""
import argparse
import hashlib
import json
import re
import zipfile

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs

PRICING = dict(source='https://modal.com/pricing', verified_date='2026-11-16',
               gpu='A10', gpu_usd_per_second=0.000306, physical_cpu_cores=2,
               cpu_usd_per_core_second=0.0000131, memory_gib=16,
               memory_usd_per_gib_second=0.00000222, margin=1.2,
               startup_cleanup_reserve_usd=0.50)


def conservative_rate():
    return (PRICING['gpu_usd_per_second'] + PRICING['physical_cpu_cores'] * PRICING['cpu_usd_per_core_second'] +
            PRICING['memory_gib'] * PRICING['memory_usd_per_gib_second']) * PRICING['margin']


def safe_archive_name(name):
    from pathlib import PurePosixPath
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unsafe archive path')
    return path


def bundle(prepared, run_id):
    import pandas as pd
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid bundle ID')
    prepared = project_path(prepared)
    if not json.loads((prepared / 'complete.json').read_text())['complete']:
        raise ValueError('Full protocol preparation is incomplete')
    protocol = json.loads((prepared / 'protocol.json').read_text())
    config = protocol['config']
    inputs = json.loads((prepared / 'inputs.json').read_text())
    verify_inputs(inputs)
    paths = {project_path(path) for path in inputs}
    paths.update(path for path in prepared.rglob('*') if path.is_file())
    # The remote worker derives a new explicitly resource-authorised protocol
    # from the original verified source preparation after budget approval.
    source = project_path(protocol['source_preparation'])
    paths.update(path for path in source.rglob('*') if path.is_file())
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new bundle ID')
    output.mkdir(parents=True)
    hashes, file_rows = {}, []
    archive = output / 'inputs.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            safe_archive_name(relative)
            if relative.startswith('data/raw/') or relative.endswith('/splits/test.csv'):
                raise ValueError('Raw DICOM/test index must not enter this development bundle')
            sha = digest(path)
            hashes[relative] = sha
            file_rows.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha))
            handle.write(path, relative)
        handle.writestr('_payload_manifest.json', json.dumps(hashes, indent=2))
    # Verify the actual archive bytes, not just its source file list.
    with zipfile.ZipFile(archive) as handle:
        for name, expected in hashes.items():
            if hashlib.sha256(handle.read(name)).hexdigest() != expected:
                raise ValueError('Archive member hash mismatch')
    csv_write(output / 'files.csv', pd.DataFrame(file_rows))
    json_write(output / 'inputs.json', hashes)
    training_seconds = 2 * config['resources']['max_variant_wall_seconds']
    validation_seconds = 2 * config['resources'].get('max_validation_wall_seconds', 1800)
    proposal = dict(
        pricing=PRICING, conservative_worker_usd_per_second=conservative_rate(),
        configured_training_seconds=training_seconds, configured_validation_seconds=validation_seconds,
        time_cap_cost_with_margin_usd=(training_seconds + validation_seconds) * conservative_rate(),
        startup_cleanup_reserve_usd=PRICING['startup_cleanup_reserve_usd'],
        proposed_allowance_usd=6.0, authorised_allowance_usd=None,
        worker_timeout_seconds=config['resources'].get('maximum_cloud_worker_seconds', 14400),
        meaning='Proposed private single-A10 job, one app/container, sequential variants; time caps may stop an incomplete run. Rate-based estimate is not a billing hard limit.',
        exclusions='No assumption about free credits. Storage/egress/platform accounting are not a proved exact bill.',
    )
    json_write(output / 'execution_proposal.json', proposal)
    verify_inputs(hashes)
    json_write(output / 'complete.json', dict(
        complete=True, archive=str(archive.relative_to(ROOT)), archive_sha256=digest(archive),
        compressed_bytes=archive.stat().st_size, uncompressed_bytes=sum(row['bytes'] for row in file_rows),
        payload_files=len(hashes), prepared=str(prepared.relative_to(ROOT)),
        source_preparation=protocol['source_preparation'], train_val_only=True,
        local_archive_verified=True, uploaded=False, cloud_launched=False,
    ))
    (output / 'README.md').write_text(
        '# Local cloud-input bundle and execution proposal\n\n'
        'Producer: `python -m scripts.v8_generator_dev.bundle_full`.\n'
        'inputs.zip contains only the frozen code/metadata/train-val arrays/judges and source-preparation evidence. '
        'Its _payload_manifest.json fingerprints every member. files.csv and inputs.json record the file/source hashes. '
        'execution_proposal.json records official pricing URL/date, calculation assumptions and proposed unapproved allowance. '
        'complete.json records archive size/hash and verification. The archive has not been uploaded and no cloud job was launched.\n', encoding='utf-8')
    print('LOCAL BUNDLE READY', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    bundle(args.prepared, args.run_id)
