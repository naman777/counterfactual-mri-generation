"""Bind verified final parents to the explicitly approved, bounded adapter pilot."""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from scripts.v8_generator_dev.adapter_pilot import pilot_settings
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def offline_check(prepared,archive,plan):
    """Rebuild the exact staged package locally before any paid app exists."""
    with tempfile.TemporaryDirectory(prefix='adapter_package_',dir=ROOT) as temporary:
        staged=Path(temporary).resolve()
        if not staged.is_relative_to(ROOT):
            raise ValueError('Temporary package must remain within the workspace')
        with zipfile.ZipFile(archive) as bundle:
            manifest=json.loads(bundle.read('_payload_manifest.json'))
            for name,expected in manifest.items():
                relative=PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe original archive member')
                content=bundle.read(name)
                import hashlib
                if hashlib.sha256(content).hexdigest()!=expected:
                    raise ValueError('Archive content mismatch')
                destination=staged/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as handle:
                    handle.write(content)
        for name,expected in plan['patches'].items():
            source=prepared/'code'/name
            if digest(source)!=expected:
                raise ValueError('Pilot patch changed')
            destination=staged/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
        runtime_prepared=staged/plan['prepared_relative']
        runtime_prepared.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv'):
            shutil.copyfile(prepared/name,runtime_prepared/name)
        for name in plan['parent_paths'].values():
            destination=staged/name.replace('\\','/')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(project_path(name),destination)
        code=('import json; from scripts.v8_generator_dev.io import verify_inputs; '
            'from scripts.v8_generator_dev.adapter_pilot import pilot_settings; '
            'from scripts.v8_generator_dev.evaluate_adapter import evaluate; '
            'from scripts.v8_generator_dev.data import CanonicalDataset; '
            'from pathlib import Path; import sys,time; p=Path(sys.argv[1]); start=time.monotonic(); '
            'print("PACKAGE IMPORTS VERIFIED",flush=True); '
            'verify_inputs(json.loads((p/"inputs.json").read_text())); '
            'print("STAGED INPUT LEDGER VERIFIED",time.monotonic()-start,flush=True); '
            'v=json.loads((p/"protocol.json").read_text()); assert v["settings"]==pilot_settings(); '
            'a=CanonicalDataset(v["view"],"train"); print("TRAIN MANIFEST VERIFIED",time.monotonic()-start,flush=True); '
            'b=CanonicalDataset(v["view"],"val"); print("VALIDATION MANIFEST VERIFIED",time.monotonic()-start,flush=True); '
            'assert a.frame.patient_id.nunique()==72 and b.frame.patient_id.nunique()==15; '
            'print("OFFLINE STAGED PACKAGE VERIFIED",len(a),len(b))')
        command=[sys.executable,'-c',code,plan['prepared_relative']]
        with (prepared/'offline_package.txt').open('x',encoding='utf-8') as handle:
            result=subprocess.run(command,cwd=staged,env=dict(os.environ,PYTHONPATH=str(staged)),
                stdout=handle,stderr=subprocess.STDOUT,timeout=600)
        json_write(prepared/'offline_package.json',dict(exit_code=result.returncode,
            original_archive_verified=True,exact_staged_ledger_checked=True,new_gpu_app_created=False))
        if result.returncode:
            raise RuntimeError('Offline staged-package validation failed; no cloud launch permitted')


def prepare(fetched,postprocess_prefix,run_id,report_id,preflight):
    if not all(re.fullmatch('[A-Za-z0-9_-]+',value) for value in (postprocess_prefix,run_id,report_id)):
        raise ValueError('Safe new pilot identifiers required')
    fetched,preflight=project_path(fetched),project_path(preflight)
    base=ROOT/'reports/v8_generator_dev'
    approval_receipt=base/'adapter_approval_20261117_01/approval.json'
    approval=json.loads(approval_receipt.read_text())
    if approval['approved_additional_usd']!=8 or approval['automatic_paid_restart']:
        raise ValueError('Explicit bounded additional US$8 approval receipt required')
    final_inputs=base/(postprocess_prefix+'_inputs')
    verification=base/(postprocess_prefix+'_verification')
    receipt=json.loads((fetched/'complete.json').read_text())
    checked=json.loads((final_inputs/'checkpoint_checks.json').read_text())
    verified=json.loads((verification/'complete.json').read_text())
    recovered_components=(receipt.get('training_and_validation_success') is True and
        receipt.get('cloud_analysis_import_failed') is True and (fetched/'root/failed.json').is_file() and
        json.loads((base/(postprocess_prefix+'_comparison')/'complete.json').read_text())['complete'])
    if not (receipt['workflow_success'] or recovered_components) or not verified['saved_validation_verified']:
        raise ValueError('Current completed validation and independent saved-array checks required')
    if not all(all(values.values()) for values in checked.values()):
        raise ValueError('Final checkpoint checks failed')
    if not json.loads((preflight/'complete.json').read_text())['real_parent_updates_finite']:
        raise ValueError('Real-parent local adapter preflight required')
    verify_inputs(json.loads((preflight/'inputs.json').read_text()))
    evaluator_preflight=base/'adapter_evaluator_preflight_20261117_01'
    if not json.loads((evaluator_preflight/'complete.json').read_text())['absent_guide_parent_images_exact']:
        raise ValueError('All evaluator routes and absent-guide parent equality must pass')
    verify_inputs(json.loads((evaluator_preflight/'inputs.json').read_text()))
    trained=json.loads((fetched/'training/complete.json').read_text())
    if trained['progress']['additive']!=trained['progress']['scale_shift']:
        raise ValueError('Adapter parents must have equal primary training exposure')
    parents=trained['checkpoints']
    for backbone,path in parents.items():
        if digest(project_path(path))!=trained['checkpoint_hashes'][backbone]:
            raise ValueError('Verified final parent changed')
    protocol_path,=list((fetched/'protocol').glob('*/protocol.json'))
    original_protocol=json.loads(protocol_path.read_text())
    config=original_protocol['config']
    output=base/report_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.prepare_adapter_run. This is a local-only gated '
        'preparation; no paid app is created here. approval/plan/protocol/selection retain scope, '
        'parents and settings; code/ and modal_adapter.py snapshot runtime sources; inputs.json '
        'is the staged runtime ledger. offline_package.txt/json retain exact-package checking. '
        'readiness_inputs.json and complete.json exist only after readiness succeeds. A missing '
        'completion marker means incomplete preparation, never permission to launch.\n',encoding='utf-8')
    (output/'code').mkdir()
    proposal=base/'next_gpu_proposal_20261117_01'
    archive=base/'full_bundle_20261116_01/inputs.zip'
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('_payload_manifest.json'))
    patches={}
    for name,expected in manifest.items():
        if name.endswith('.py') and digest(ROOT/name)!=expected:
            patches[name]=digest(ROOT/name)
    for name in ('adapter_pilot.py','evaluate_adapter.py','soft_structure.py','precision_step.py'):
        relative='scripts/v8_generator_dev/'+name
        patches[relative]=digest(ROOT/relative)
    patches['scripts/v6_structure/model.py']=digest(ROOT/'scripts/v6_structure/model.py')
    # Include the full local import closure, including the older adapter's
    # lowpass helper and its own imports. A successful import in the broad
    # developer workspace does not prove the isolated cloud package is complete.
    pending=list(patches)
    visited=set()
    while pending:
        name=pending.pop()
        if name in visited:
            continue
        visited.add(name)
        tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if not module.startswith(('scripts.','src.')):
                    continue
                relative=module.replace('.','/')+'.py'
                local=ROOT/relative
                if not local.is_file():
                    continue
                expected=digest(local)
                if manifest.get(relative)!=expected:
                    patches[relative]=expected
                pending.append(relative)
    for name,expected in patches.items():
        destination=output/'code'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
        if digest(destination)!=expected:
            raise ValueError('Pilot source snapshot changed')
    shutil.copyfile(ROOT/'scripts/v8_generator_dev/modal_adapter.py',output/'modal_adapter.py')
    json_write(output/'approval.json',dict(additional_user_approved_usd=8,
        source='Explicit user reply: Approve additional US$8',previous_allowances_reset=False,
        source_receipt=str(approval_receipt.relative_to(ROOT)),source_receipt_sha256=digest(approval_receipt),
        scope_proposal=str(proposal.relative_to(ROOT)),one_app=True,one_a10=True,
        maximum_worker_seconds=14400,automatic_paid_restart=False))
    protocol=dict(format='soft_adapter_pilot_protocol_v1',run_id=run_id,settings=pilot_settings(),
        parents=parents,parent_hashes=trained['checkpoint_hashes'],parent_epoch=trained['progress']['additive']['epoch'],
        view=config['view'],approved_additional_usd=8,parent_config=config,
        evaluation='Existing fixed points, three generation seeds, all named judges; correct/absent/wrong guide controls',
        test_pixels_used=False,new_candidates_used=False,clinical_anatomy_validated=False)
    json_write(output/'protocol.json',protocol)
    shutil.copyfile(protocol_path.parent/'validation_selection.csv',output/'validation_selection.csv')
    selected=pd.read_csv(output/'validation_selection.csv')
    if len(selected)!=15 or selected.patient_id.nunique()!=15:
        raise ValueError('Original fifteen preselected development-validation patients required')
    inputs=dict(manifest)
    inputs.update(patches)
    inputs.update({path.replace('\\','/'):trained['checkpoint_hashes'][backbone] for backbone,path in parents.items()})
    json_write(output/'inputs.json',inputs)
    plan=dict(run_id=run_id,prepared_relative=str(output.relative_to(ROOT)).replace('\\','/'),
        archive=str(archive.relative_to(ROOT)),archive_sha256=digest(archive),patches=patches,
        source_parent_project='v8_generator_dev/precision_continuation_20261117_01/project',
        parent_paths=parents,parent_hashes=trained['checkpoint_hashes'],approved_additional_usd=8,
        price_with_margin=.000441264,reserve_usd=.5,maximum_worker_seconds=14400,
        maximum_training_seconds=8400,one_app=True,one_gpu=True,automatic_paid_restart=False)
    json_write(output/'plan.json',plan)
    offline_check(output,archive,plan)
    evidence=[approval_receipt,fetched/'complete.json',final_inputs/'checkpoint_checks.json',verification/'complete.json',
        preflight/'complete.json',preflight/'inputs.json',proposal/'PROPOSAL.md',output/'modal_adapter.py',
        ROOT/'scripts/v8_generator_dev/prepare_adapter_run.py',output/'offline_package.json',
        evaluator_preflight/'complete.json',evaluator_preflight/'inputs.json']
    evidence.extend(output/'code'/name for name in patches)
    json_write(output/'readiness_inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in evidence})
    json_write(output/'complete.json',dict(complete=True,approved_additional_usd=8,verified_parents=True,
        local_preflight_passed=True,cloud_launched=False,anatomy_validated=False))
    print('ADAPTER RUN PREPARED',output,flush=True)


def launch(prepared,report_id):
    prepared=project_path(prepared)
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    plan=json.loads((prepared/'plan.json').read_text())
    if plan['approved_additional_usd']!=8:
        raise ValueError('Separate explicit additional US$8 approval required')
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONUNBUFFERED='1',CAPSTONE_ADAPTER_PREPARED=str(prepared))
    inventory=subprocess.run([sys.executable,'-m','modal','app','list','--profile','namankundra7','--json'],
        cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    apps=json.loads(inventory.stdout)
    if any(app['state']!='stopped' for app in apps):
        raise ValueError('Another Modal app is active; do not launch in parallel')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    for name in ('approval.json','plan.json'):
        shutil.copyfile(prepared/name,output/name)
    json_write(output/'apps_before.json',apps)
    command=[sys.executable,'-m','modal','run','--profile','namankundra7',
        str(prepared/'modal_adapter.py'),'--approved-usd','8']
    json_write(output/'command.json',dict(command=command))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.prepare_adapter_run launch. approval/plan/command '
        'record the separate bounded launch; apps_before excludes another active app. launch.log '
        'and process_result.json retain actual CLI output and wall time. Scientific completion '
        'requires remote receipts plus independent local verification. No paid auto-restart.\n',encoding='utf-8')
    start=time.monotonic()
    with (output/'launch.log').open('x',encoding='utf-8') as handle:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    json_write(output/'process_result.json',dict(exit_code=process.returncode,seconds=time.monotonic()-start,
        approved_additional_usd=8,exact_invoice_known=False))
    print('ADAPTER CLI EXIT',process.returncode,flush=True)
    return process.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['prepare','launch'])
    p.add_argument('--fetched')
    p.add_argument('--postprocess-prefix')
    p.add_argument('--run-id')
    p.add_argument('--report-id',required=True)
    p.add_argument('--preflight')
    p.add_argument('--prepared')
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.fetched,a.postprocess_prefix,a.run_id,a.report_id,a.preflight)
    else:
        raise SystemExit(launch(a.prepared,a.report_id))
