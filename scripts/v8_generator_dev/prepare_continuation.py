"""Freeze a reviewable continuation proposal; never infer a new spending approval."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile

from scripts.v8_generator_dev.continuation import proposed_config
from scripts.v8_generator_dev.failed_pair import select_failed_pair, RATE_WITH_MARGIN
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def prepare(run_id, report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+', value) for value in (run_id, report_id)):
        raise ValueError('Safe new run/report IDs required')
    base = ROOT / 'reports/v8_generator_dev'
    migration = base / 'precision_resume_preparation_20261117_01'
    verify_inputs(json.loads((migration / 'inputs.json').read_text()))
    diagnostic = base / 'diagnostic_analysis_20261117_01'
    if not json.loads((diagnostic / 'complete.json').read_text())['controlled_retry_gate_passed']:
        raise ValueError('Recovered-failure control gate must pass')
    training = base / 'cloud_resume_fetch_20261116_01'
    failure = json.loads((training / 'root/failed.json').read_text())
    selection = select_failed_pair(training / 'training', failure, (training / 'root/train.txt').read_text())
    config = json.loads((training / 'training/protocol.json').read_text())['config']
    bundle = json.loads((base / 'full_bundle_20261116_01/complete.json').read_text())
    archive = project_path(bundle['archive'])
    if digest(archive) != bundle['archive_sha256']:
        raise ValueError('Original approved archive changed')
    output = base / report_id
    output.mkdir(exist_ok=False)
    (output / 'README.md').write_text(
        '# Matched continuation proposal\n\nProducer: `python -m scripts.v8_generator_dev.prepare_continuation`.\n'
        'plan.json freezes parent/config/code/price assumptions. PROPOSAL.md explains the bounded job. '
        'tests.txt/tests.json retain local checks; code/ snapshots every runtime patch and the Modal runner. '
        'inputs.json hashes original evidence and source. complete.json records local readiness, '
        'never a new approval, launched training or improved anatomy.\n', encoding='utf-8')
    names = set(('precision_step.py', 'prepare_precision_resume.py', 'numerical_replay.py',
                 'continuation.py', 'continuation_smoke.py', 'modal_continue.py'))
    # Include every changed scientific source from the old package, rather than
    # assuming only the engine itself changed during the precision repair.
    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read('_payload_manifest.json'))
    for name, expected in manifest.items():
        if name.startswith('scripts/v8_generator_dev/') and name.endswith('.py'):
            if digest(ROOT / name) != expected:
                names.add(name.rsplit('/', 1)[-1])
    (output / 'code').mkdir()
    for name in sorted(names):
        shutil.copyfile(ROOT / 'scripts/v8_generator_dev' / name, output / 'code' / name)
    modules = ['test_pipeline', 'test_full', 'test_analysis', 'test_guides', 'test_resource_limited',
        'test_verification', 'test_failed_pair', 'test_precision_step', 'test_numerical_replay',
        'test_precision_resume', 'test_preservation_screen', 'test_preservation_screen_v2', 'test_continuation']
    command = [sys.executable, '-m', 'unittest', *['scripts.v8_generator_dev.'+name for name in modules], '-v']
    with (output / 'tests.txt').open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
    json_write(output / 'tests.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError('Continuation preflight failed; keep failed preparation')
    validation = base / 'failed_pair_results_20261117_01/root/failed_pair_validation_20261117_02/complete.json'
    previous_validation_seconds = json.loads(validation.read_text())['seconds']
    estimated_train = failure['seconds'] * (config['full_training']['max_epochs']-70)/70
    estimated_seconds = estimated_train + previous_validation_seconds + 600
    plan = dict(run_id=run_id, selection=selection, proposed_config=proposed_config(config),
        proposed_new_allowance_usd=6, budget_approved=False, original_allowance_reset=False,
        archive=str(archive.relative_to(ROOT)), archive_sha256=bundle['archive_sha256'],
        patch_hashes={name: digest(output / 'code' / name) for name in sorted(names)},
        rate_with_margin=RATE_WITH_MARGIN, pricing_source='https://modal.com/pricing', pricing_verified_date='2026-11-17',
        gpu='A10', physical_cpu_cores=2, memory_gib=16, workers=1, apps_at_once=1,
        worker_timeout_seconds=13500, estimate_guard_seconds=(6-.5)/RATE_WITH_MARGIN,
        maximum_epoch_scenario_estimated_seconds=estimated_seconds,
        maximum_epoch_scenario_estimated_usd=estimated_seconds*RATE_WITH_MARGIN+.5,
        estimate_assumptions='Original training wall time scaled from 70 to remaining 130 epochs, completed paired edit-validation duration, 600s extra overhead, 20% rate margin and US$0.50 reserve; not a runtime/bill guarantee',
        partial_stop_policy='Publish only hash-verified completed matched blocks; resource/numerical interruption remains incomplete; no automatic retry',
        preservation_status='Automated proxy benchmark only; no clinical or lesion-preservation claim')
    json_write(output / 'plan.json', plan)
    lines = ['# Proposed matched generator continuation', '',
        'Request: a new US$6 allowance for one private sequential A10 app. No approval is inferred '
        'from the original US$6, whose remaining estimate stays separate.', '',
        '- Start both models at the last matched epoch 70, with raw/EMA weights, optimiser, scaler, '
        'RNG, cursor and history retained. Migrate under the original Torch/CUDA package runtime.',
        '- Change only bounded same-draw AMP retry as a scientific setting. Increase execution-time '
        'caps explicitly; keep learning rate, data, sampling, stopping, fixed EchoTime and evaluation frozen.',
        '- Run strict CUDA resume/full-batch software checks first and discard their updates. '
        'Then continue the matched pair until both meet the existing plateau rule or epoch 200.',
        '- Stage hash-verified inputs locally and back up every completed matched block to the volume. '
        'Run frozen validation and paired patient-CI analysis after completed training.',
        '- Keep numerical/resource interruptions incomplete. Publish no anatomical/clinical validity '
        'or model winner merely because training finishes.', '',
        f'The maximum-epoch scenario estimates {estimated_seconds/3600:.2f} hours and '
        f'US${plan["maximum_epoch_scenario_estimated_usd"]:.2f}, including the stated margin/reserve. '
        'Plateau stopping can shorten it. The US$6 allowance gives an estimate-based compute deadline '
        'of about 3.46 hours; this is not an exact billing cap or a promise of completion.', '',
        'The worker uses an existing approved private archive and original remote checkpoint pair; '
        'no new cohort is downloaded. Runtime code patches and all parent hashes are in plan.json/code/. '
        'Local preparation does not launch Modal.', '']
    (output / 'PROPOSAL.md').write_text('\n'.join(lines), encoding='utf-8')
    sources = [migration / 'complete.json', training / 'root/failed.json', training / 'root/train.txt',
        training / 'training/protocol.json', diagnostic / 'complete.json', validation, archive,
        ROOT / 'scripts/v8_generator_dev/prepare_continuation.py', ROOT / 'scripts/v8_generator_dev/test_continuation.py']
    sources += [ROOT / 'scripts/v8_generator_dev' / name for name in names]
    inputs = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, local_checks_passed=True,
        budget_approved=False, cloud_launched=False, full_training_completed=False, anatomy_validated=False))
    print('CONTINUATION PREPARED', output, flush=True)


def launch(prepared, report_id, approved_usd):
    if approved_usd != 6.0:
        raise ValueError('A new explicit user US$6 approval is required')
    prepared = project_path(prepared)
    verify_inputs(json.loads((prepared / 'inputs.json').read_text()))
    plan = json.loads((prepared / 'plan.json').read_text())
    if not json.loads((prepared / 'complete.json').read_text())['local_checks_passed']:
        raise ValueError('Passed local preflight required')
    if not re.fullmatch('[A-Za-z0-9_-]+', report_id):
        raise ValueError('Safe new launch ID required')
    import os
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1',
        CAPSTONE_V8_CONTINUATION=str(prepared), CAPSTONE_V8_BUNDLE=str(project_path(plan['archive'])))
    inventory = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--profile', 'namankundra7', '--json'],
        cwd=ROOT, capture_output=True, text=True, check=True, encoding='utf-8', env=environment)
    apps = json.loads(inventory.stdout)
    if any(row['state'] != 'stopped' for row in apps):
        raise ValueError('Another Modal app is active')
    output = ROOT / 'reports/v8_generator_dev' / report_id
    output.mkdir(exist_ok=False)
    json_write(output / 'approval.json', dict(new_user_approved_usd=approved_usd,
        source='Explicit user approval of the prepared continuation proposal', original_allowance_reset=False,
        run_id=plan['run_id'], prepared=str(prepared.relative_to(ROOT))))
    json_write(output / 'apps_before.json', apps)
    shutil.copyfile(prepared / 'plan.json', output / 'plan.json')
    command = [sys.executable, '-m', 'modal', 'run', '--profile', 'namankundra7',
        'scripts/v8_generator_dev/modal_continue.py', '--approved-usd', str(approved_usd)]
    json_write(output / 'command.json', dict(command=command))
    (output / 'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.prepare_continuation launch`.\n'
        'approval.json records the new allowance, apps_before.json isolates one app; plan.json and '
        'command.json freeze the scope. launch.log/process_result.json retain CLI output/runtime/exit. '
        'Scientific completion requires remote receipts, not CLI exit alone.\n', encoding='utf-8')
    started = time.monotonic()
    with (output / 'launch.log').open('x', encoding='utf-8') as handle:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    json_write(output / 'process_result.json', dict(exit_code=process.returncode, seconds=time.monotonic()-started,
        run_id=plan['run_id'], exact_bill_known=False))
    print('CONTINUATION CLI EXIT', process.returncode, flush=True)
    return process.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'launch'])
    parser.add_argument('--run-id')
    parser.add_argument('--report-id', required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--approved-usd', type=float)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run_id, args.report_id)
    else:
        raise SystemExit(launch(args.prepared, args.report_id, args.approved_usd))
