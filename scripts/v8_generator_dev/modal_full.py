"""Private one-app/one-GPU workflow; launch only after explicit budget approval.

Set CAPSTONE_V8_BUNDLE to a verified local inputs.zip, then:
python -m modal run scripts/v8_generator_dev/modal_full.py --run-id <new_id> --budget-usd <approved cap>
No Modal calls occur merely by preparing the local bundle or running unit tests.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

app = modal.App('capstone-v8-generator-full')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow',
).env({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
if modal.is_local():
    value = os.environ.get('CAPSTONE_V8_BUNDLE')
    if value:
        bundle_path = Path(value).resolve()
        if not bundle_path.is_file():
            raise ValueError('CAPSTONE_V8_BUNDLE must name the verified local input archive')
        image = image.add_local_file(bundle_path, '/input/inputs.zip', copy=True)


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, timeout=14400,
              volumes={'/data': volume}, max_containers=1, retries=0, scaledown_window=2)
def workflow(run_id: str, budget_usd: float):
    import hashlib
    import subprocess
    import zipfile
    from pathlib import PurePosixPath
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id) or not 0.5 < budget_usd <= 6.0:
        raise ValueError('This prepared workflow accepts a new ID and an explicit approved allowance <= US$6')
    started = time.monotonic()
    # Same official rate/margin used in the reviewable local proposal. This
    # command-level timeout is an estimate-based guard, not a billing API cap.
    conservative_rate = (0.000306 + 2 * 0.0000131 + 16 * 0.00000222) * 1.2
    deadline = started + min(14000, (budget_usd - 0.50) / conservative_rate)
    volume.reload()
    home = Path('/data/v8_generator_dev') / run_id
    if home.exists():
        raise FileExistsError('Remote workflow IDs are append-only')
    project = home / 'project'
    project.mkdir(parents=True)
    with zipfile.ZipFile('/input/inputs.zip') as archive:
        manifest = json.loads(archive.read('_payload_manifest.json'))
        if set(archive.namelist()) != set(manifest) | {'_payload_manifest.json'}:
            raise ValueError('Archive payload does not match its manifest')
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('Input archive hash mismatch')
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(payload)
    sys.path.insert(0, str(project))
    from scripts.v8_generator_dev.io import json_write
    metadata = [path for path in project.glob('reports/v8_generator_dev/*/protocol.json')
                if json.loads(path.read_text()).get('config', {}).get('format') == 'generator_full_protocol_v1']
    if len(metadata) != 1:
        raise ValueError('Bundle must identify exactly one proposed full protocol')
    proposed = json.loads(metadata[0].read_text())
    overrides = json.loads((project / 'configs/generator_full_v2.json').read_text())
    overrides['resources']['monetary_budget_authorised'] = True
    overrides['resources']['authorised_allowance_usd'] = float(budget_usd)
    config_path = project / 'configs' / f'generator_authorised_{run_id}.json'
    json_write(config_path, overrides)
    json_write(home / 'authorisation.json', dict(
        run_id=run_id, budget_usd=budget_usd, authorisation='Explicit local entrypoint invocation after user approval',
        proposed_protocol=str(metadata[0].relative_to(project)),
        deadline_seconds=deadline - started, one_app=True, max_gpus=1,
        guard='Conservative rate/time estimate; not a proved platform billing hard cap',
    ))
    phase_results = []
    def call(name, module, arguments):
        remaining = int(deadline - time.monotonic())
        if remaining <= 30:
            raise TimeoutError('Approved allowance/time reserve exhausted')
        command = [sys.executable, '-m', module, *arguments]
        log = home / f'{name}.txt'
        try:
            with log.open('x') as handle:
                result = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT,
                                        timeout=remaining - 15,
                                        env=dict(os.environ, PYTHONPATH=str(project)))
        finally:
            volume.commit()
        phase_results.append(dict(phase=name, command=command, exit_code=result.returncode))
        json_write(home / f'{name}.json', phase_results[-1])
        volume.commit()
        if result.returncode:
            raise RuntimeError(f'Phase failed: {name}; logs retained')
    prepared_id, training_id = f'{run_id}_protocol', f'{run_id}_training'
    try:
        call('prepare', 'scripts.v8_generator_dev.prepare_full', [
            '--config', str(config_path.relative_to(project)),
            '--source-preparation', proposed['source_preparation'], '--run-id', prepared_id])
        call('tests', 'unittest', ['scripts.v8_generator_dev.test_pipeline', 'scripts.v8_generator_dev.test_full', '-v'])
        call('engineering', 'scripts.v8_generator_dev.full_train', [
            '--prepared', f'reports/v8_generator_dev/{prepared_id}', '--run-id', f'{run_id}_engineering',
            '--engineering-steps', '2'])
        call('train', 'scripts.v8_generator_dev.full_train', [
            '--prepared', f'reports/v8_generator_dev/{prepared_id}', '--run-id', training_id])
        complete_path = project / f'reports/v8_generator_dev/{training_id}/complete.json'
        if not complete_path.exists():
            raise RuntimeError('Training stopped at a resource cap; evaluation not promoted')
        trained = json.loads(complete_path.read_text())
        for variant, checkpoint in trained['checkpoints'].items():
            call(f'evaluate_{variant}', 'scripts.v8_generator_dev.evaluate_full', [
                '--prepared', f'reports/v8_generator_dev/{prepared_id}', '--checkpoint', checkpoint,
                '--run-id', f'{run_id}_validation_{variant}'])
        json_write(home / 'complete.json', dict(complete=True, phases=phase_results,
                                                budget_usd=budget_usd, seconds=time.monotonic() - started,
                                                anatomy_validated=False, fresh_test_evaluated=False))
    except Exception as error:
        json_write(home / 'failed.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
                                              phases=phase_results, seconds=time.monotonic() - started))
        raise
    finally:
        volume.commit()
    return dict(remote_folder=str(home), complete=True)


@app.local_entrypoint()
def main(run_id: str, budget_usd: float):
    if not os.environ.get('CAPSTONE_V8_BUNDLE'):
        raise ValueError('Set CAPSTONE_V8_BUNDLE only after reviewing the verified bundle and approved budget')
    print(workflow.remote(run_id, budget_usd))
