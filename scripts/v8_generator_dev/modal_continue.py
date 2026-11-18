"""Approved sequential continuation with local staging and immutable paired-block backup."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import modal

app = modal.App('capstone-v8-precision-continuation')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow',
).env({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
if modal.is_local():
    folder = Path(os.environ['CAPSTONE_V8_CONTINUATION']).resolve()
    plan = json.loads((folder / 'plan.json').read_text())
    image = image.add_local_file(Path(os.environ['CAPSTONE_V8_BUNDLE']), '/input/inputs.zip', copy=True)
    image = image.add_local_file(folder / 'plan.json', '/input/plan.json', copy=True)
    for name in plan['patch_hashes']:
        image = image.add_local_file(folder / 'code' / name, '/patches/'+name, copy=True)


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, max_containers=1,
              retries=0, scaledown_window=2, timeout=13500, volumes={'/data': volume})
def workflow(approved_usd: float):
    if approved_usd != 6.0:
        raise ValueError('Explicit new US$6 approval required for this substantial continuation')
    started = time.monotonic()
    plan = json.loads(Path('/input/plan.json').read_text())
    run_id = plan['run_id']
    if not run_id.replace('_', '').isalnum():
        raise ValueError('Safe new run ID required')
    deadline = started + min(13200, (approved_usd-.50)/plan['rate_with_margin'])
    volume.reload()
    home = Path('/data/v8_generator_dev') / run_id
    home.mkdir(exist_ok=False)
    project = home / 'project'
    project.mkdir()
    staged = Path('/tmp') / run_id
    staged.mkdir(exist_ok=False)
    original = Path('/data/v8_generator_dev/full_20261116_04/project')

    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()

    def write(path, value):
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    def persist_file(source, destination):
        if destination.exists():
            if sha(source) != sha(destination):
                raise ValueError('Refuse to replace persisted evidence: '+str(destination))
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)

    def persist_tree(source, destination):
        # Completion markers are copied last, after every payload file.
        files = sorted((path for path in source.rglob('*') if path.is_file()),
                       key=lambda path: (path.name == 'complete.json', str(path)))
        for path in files:
            persist_file(path, destination / path.relative_to(source))

    phases, backed_up = [], set()
    training_relative = f'reports/v8_generator_dev/{run_id}_training'
    training = staged / training_relative

    def backup_pairs():
        for path in sorted(training.glob('block_*/complete.json')):
            if path.parent.name in backed_up:
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if record['progress']['additive'] != record['progress']['scale_shift']:
                raise ValueError('Refuse to publish an unmatched pair')
            for variant, relative in record['checkpoints'].items():
                source = staged / relative.replace('\\', '/')
                if sha(source) != record['checkpoint_hashes'][variant]:
                    raise ValueError('Pair checkpoint changed before backup')
                persist_file(source, project / relative.replace('\\', '/'))
            persist_tree(path.parent, project / training_relative / path.parent.name)
            backed_up.add(path.parent.name)
            epoch = record['progress']['additive']['epoch']
            write(home / f'paired_epoch_{epoch:04d}.json', dict(epoch=epoch, backed_up=True,
                checkpoint_hashes=record['checkpoint_hashes'], full_training_completed=False))
            volume.commit()

    def call(name, module, arguments, monitor=False):
        remaining = int(deadline-time.monotonic())
        if remaining <= 60:
            raise TimeoutError('Continuation allowance reserve exhausted')
        command = [sys.executable, '-m', module, *arguments]
        with (home / (name+'.txt')).open('x') as handle:
            process = subprocess.Popen(command, cwd=staged, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH=str(staged), PYTHONUNBUFFERED='1'))
            try:
                while process.poll() is None:
                    if time.monotonic() > deadline-30:
                        raise TimeoutError('Continuation compute deadline reached')
                    if monitor:
                        backup_pairs()
                    volume.commit()
                    time.sleep(5)
                if monitor:
                    backup_pairs()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        record = dict(phase=name, command=command, exit_code=process.returncode)
        phases.append(record)
        write(home / (name+'.json'), record)
        volume.commit()
        if process.returncode:
            raise RuntimeError('Continuation phase failed: '+name)

    try:
        if sha(Path('/input/inputs.zip')) != plan['archive_sha256']:
            raise ValueError('Original approved archive changed')
        with zipfile.ZipFile('/input/inputs.zip') as archive:
            manifest = json.loads(archive.read('_payload_manifest.json'))
            if set(archive.namelist()) != set(manifest) | {'_payload_manifest.json'}:
                raise ValueError('Archive membership mismatch')
            for name, expected in manifest.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive name')
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError('Archive hash mismatch')
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        prepared_relative = 'reports/v8_generator_dev/full_20261116_04_protocol'
        source_prepared = original / prepared_relative
        shutil.copytree(source_prepared, staged / prepared_relative)
        required = json.loads((source_prepared / 'inputs.json').read_text())
        required.update({relative: plan['selection']['checkpoint_record']['checkpoint_hashes'][variant]
            for variant, relative in plan['selection']['checkpoint_record']['checkpoints'].items()})
        for relative, expected in required.items():
            destination = staged / relative.replace('\\', '/')
            if not destination.resolve().is_relative_to(staged):
                raise ValueError('Frozen input escapes project')
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / relative.replace('\\', '/'), destination)
            if sha(destination) != expected:
                raise ValueError('Original frozen input mismatch: '+relative)
        original_code = staged / 'frozen_original_code'
        original_code.mkdir()
        for name, expected in plan['patch_hashes'].items():
            source = Path('/patches') / name
            if Path(name).name != name or sha(source) != expected:
                raise ValueError('Invalid continuation code patch')
            destination = staged / 'scripts/v8_generator_dev' / name
            if destination.exists():
                shutil.copyfile(destination, original_code / name)
            shutil.copyfile(source, destination)
        shutil.copyfile('/input/plan.json', staged / 'continuation_plan.json')
        write(home / 'authorisation.json', dict(new_approved_usd=approved_usd, original_allowance_reset=False,
            scope='One-A10 sequential epoch-70 matched continuation and frozen validation',
            deadline_seconds=deadline-started, exact_bill_known=False, plan=plan))
        write(home / 'staging.json', dict(complete=True, original_inputs_verified=len(required),
            explicit_code_patch_hashes=plan['patch_hashes'], seconds=time.monotonic()-started))
        volume.commit()
        call('materialise', 'scripts.v8_generator_dev.continuation', ['--plan', 'continuation_plan.json',
            '--run-id', run_id, '--budget-usd', str(approved_usd)])
        new_prepared = f'reports/v8_generator_dev/{run_id}_protocol'
        persist_tree(staged / new_prepared, project / new_prepared)
        resume = json.loads((staged / new_prepared / 'prior_pair.json').read_text())
        for relative in resume['checkpoints'].values():
            persist_file(staged / relative.replace('\\', '/'), project / relative.replace('\\', '/'))
        call('runtime_smoke', 'scripts.v8_generator_dev.continuation_smoke', ['--prepared', new_prepared,
            '--run-id', run_id+'_runtime_smoke'])
        persist_tree(staged / f'reports/v8_generator_dev/{run_id}_runtime_smoke',
                     project / f'reports/v8_generator_dev/{run_id}_runtime_smoke')
        call('train', 'scripts.v8_generator_dev.full_train', ['--prepared', new_prepared,
            '--run-id', run_id+'_training', '--resume-pair', new_prepared+'/prior_pair.json'], monitor=True)
        persist_tree(training, project / training_relative)
        if not (training / 'complete.json').exists():
            raise RuntimeError('Incomplete resource stop; retain last matched pair without promoting completion')
        trained = json.loads((training / 'complete.json').read_text())
        for variant, checkpoint in trained['checkpoints'].items():
            evaluation = f'reports/v8_generator_dev/{run_id}_validation_{variant}'
            call('evaluate_'+variant, 'scripts.v8_generator_dev.evaluate_full', ['--prepared', new_prepared,
                '--checkpoint', checkpoint, '--run-id', run_id+'_validation_'+variant])
            persist_tree(staged / evaluation, project / evaluation)
        call('analyse', 'scripts.v8_generator_dev.analyse_full', ['--training', training_relative,
            '--additive', f'reports/v8_generator_dev/{run_id}_validation_additive',
            '--scale-shift', f'reports/v8_generator_dev/{run_id}_validation_scale_shift',
            '--run-id', run_id+'_comparison'])
        persist_tree(staged / f'reports/v8_generator_dev/{run_id}_comparison',
                     project / f'reports/v8_generator_dev/{run_id}_comparison')
        write(home / 'complete.json', dict(complete=True, phases=phases, seconds=time.monotonic()-started,
            new_allowance_usd=approved_usd, training_stop_reason=trained['stop_reason'],
            full_training_completed=True, efficacy_established=False, anatomy_validated=False,
            fresh_test_evaluated=False))
    except Exception as error:
        backup_error = None
        try:
            if training.exists():
                backup_pairs()
                persist_tree(training, project / training_relative)
        except Exception as copy_error:
            backup_error = str(copy_error)
        write(home / 'failed.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
            phases=phases, seconds=time.monotonic()-started, backed_up_pairs=sorted(backed_up),
            backup_error=backup_error, full_training_completed=False, anatomy_validated=False))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main(approved_usd: float):
    print(workflow.remote(approved_usd))
