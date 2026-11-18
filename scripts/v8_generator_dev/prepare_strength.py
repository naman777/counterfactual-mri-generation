"""Prepare a reviewable inference-only sweep; a new allowance is never inferred."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.prepare_adapter_run import offline_check


def prepare(run_id,report_id):
    if not all(value.replace('_','').isalnum() for value in (run_id,report_id)):
        raise ValueError('Safe new preparation identifiers required')
    base=ROOT/'reports/v8_generator_dev'
    analysis=base/'adapter_analysis_20261117_02'
    preflight=base/'strength_preflight_20261117_01'
    if not json.loads((analysis/'complete.json').read_text())['saved_validation_verified']:
        raise ValueError('Completed independently verified source-guide pilot required')
    if not json.loads((preflight/'complete.json').read_text())['endpoint_checks_passed']:
        raise ValueError('Actual trained-model endpoint and midpoint preflight required')
    verify_inputs(json.loads((preflight/'inputs.json').read_text()))
    previous=base/'adapter_preparation_20261117_02'
    old_plan=json.loads((previous/'plan.json').read_text())
    protocol=json.loads((previous/'protocol.json').read_text())
    trained=json.loads((base/'soft_adapter_pilot_20261117_01_training/complete.json').read_text())
    adapters={backbone:trained['copies'][backbone+'__soft_source_guide']['checkpoint'] for backbone in ('additive','scale_shift')}
    output=base/report_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.prepare_strength. PROPOSAL.md/plan.json freeze the '
        'new inference-only scope and proposed US$3 allowance, never approval. protocol/selection/'
        'inputs retain old learned weights/data/settings and new alpha values. code/ and '
        'modal_strength.py snapshot runtime sources; offline_package.txt/json retain package checks. '
        'readiness_inputs.json and complete.json require passed readiness. No paid job launches here.\n',encoding='utf-8')
    shutil.copytree(previous/'code',output/'code')
    patches=dict(old_plan['patches'])
    name='scripts/v8_generator_dev/evaluate_strength.py'
    shutil.copyfile(ROOT/name,output/'code'/name)
    patches[name]=digest(ROOT/name)
    for name,expected in patches.items():
        if digest(output/'code'/name)!=expected:
            raise ValueError('Frozen source snapshot mismatch')
    shutil.copyfile(ROOT/'scripts/v8_generator_dev/modal_strength.py',output/'modal_strength.py')
    protocol.pop('approved_additional_usd',None)
    protocol.update(run_id=run_id,stage='prospective_validation_only_source_strength',
        source_pilot_allowance_usd=8,proposed_new_allowance_usd=3,new_budget_approved=False,
        strengths=[.25,.5,.75],training_ran=False,
        purpose='Test the source-correspondence/acquisition-editing trade-off without retraining or judge gradients')
    json_write(output/'protocol.json',protocol)
    shutil.copyfile(previous/'validation_selection.csv',output/'validation_selection.csv')
    inputs=dict(json.loads((previous/'inputs.json').read_text()))
    inputs['scripts/v8_generator_dev/evaluate_strength.py']=digest(ROOT/'scripts/v8_generator_dev/evaluate_strength.py')
    remote_weights={}
    parent_paths=dict(protocol['parents'])
    for backbone,path in protocol['parents'].items():
        remote_weights[path.replace('\\','/')]=dict(project='v8_generator_dev/precision_continuation_20261117_01/project',
            sha256=protocol['parent_hashes'][backbone])
    for backbone,path in adapters.items():
        expected=trained['copies'][backbone+'__soft_source_guide']['sha256']
        if digest(project_path(path))!=expected:
            raise ValueError('Verified trained adapter changed')
        inputs[path.replace('\\','/')]=expected
        parent_paths[backbone+'_adapter']=path
        remote_weights[path.replace('\\','/')]=dict(project='v8_generator_dev/soft_adapter_pilot_20261117_01/project',sha256=expected)
    json_write(output/'inputs.json',inputs)
    remote_complete=base/'adapter_results_20261117_01/root/complete.json'
    phases=json.loads(remote_complete.read_text())['phases']
    observed=[row['seconds'] for row in phases if row['phase'] in (
        'evaluate_additive__soft_source_guide__correct','evaluate_scale_shift__soft_source_guide__correct')]
    estimate=float(np.mean(observed)*6+300)
    plan=dict(run_id=run_id,prepared_relative=output.relative_to(ROOT).as_posix(),
        archive=old_plan['archive'],archive_sha256=old_plan['archive_sha256'],patches=patches,
        parent_paths=parent_paths,adapter_checkpoints=adapters,remote_weights=remote_weights,
        strengths=[.25,.5,.75],proposed_additional_usd=3,budget_approved=False,
        previous_allowances_reset=False,maximum_worker_seconds=4500,rate_with_margin=.000441264,
        reserve_usd=.5,estimated_worker_seconds=estimate,estimated_usd=estimate*.000441264+.5,
        original_fixed_points=True,train_steps=0,validation_patients=15,generation_seeds=3,
        cases=540,named_judges=3,one_app=True,one_a10=True,automatic_paid_restart=False,
        current_weights_or_paper_modified=False,clinical_anatomy_validated=False)
    json_write(output/'plan.json',plan)
    text=['# Proposed source-guide strength sensitivity test','',
        'Request: a separate additional US$3 allowance for one private A10 app, validation only, '
        'after the verified twenty-epoch adapter pilot. Earlier allowances are not reset or reused.','',
        'Rationale: source-guided outputs have higher image correspondence, but full-strength '
        'guidance does not consistently retain acquisition editing. The learned zero-guide '
        'baseline can edit more successfully. Test a smaller guide contribution before another '
        'training run; classifier outcomes and SSIM do not establish clinical anatomy.','',
        '- Freeze both epoch-130 backbones, acquisition conditioners and trained twenty-epoch '
        'source-guide adapters. No optimiser, training updates, checkpoint selection or judge loss.',
        '- Predeclare residual guide strengths 0.25, 0.50, 0.75 on both backbones. Amplitude 0 '
        'is the existing parent; amplitude 1 is the existing full guide. Reuse their verified '
        'saved outputs rather than paying to regenerate them.',
        '- Keep both original fixed editing points, source EchoTime, fifteen preselected '
        'development-validation patients, three generation seeds and all three named judges. '
        'The six new passes contain 540 source/null/target cases. No new patient or test pixels.',
        '- Independently recompute saved grades; compare each strength with parent, full-guide '
        'and learned-zero-guide controls using patient CIs, SSIM/MAD and the unchanged proxy '
        'screen. Retain direction/support strata and negative trade-offs. No new clinical claim.',
        '- No automatic default/model promotion. These are further development measurements '
        'on reused patients, not independent confirmation. Existing condition-support eligibility '
        'rules remain unchanged; a guide strength cannot manufacture missing supported patients.',
        '- One A10, two CPU cores, 16 GiB; at most 75 worker minutes. Keep an estimated US$0.50 '
        'reserve and the prior 20%-margin rate US$0.000441264/s. No parallel app, paid restart, '
        'large download, new dataset admission, contact or publication. Partial failures stay incomplete.','',
        f'Measured-run projection: {estimate/60:.1f} worker minutes, estimated US${plan["estimated_usd"]:.2f}, '
        'from the previous two correct-guide phase durations scaled to six passes plus 300 seconds '
        'overhead and the stated margin/reserve. The 75-minute scenario estimates US$2.49; this '
        'is an estimate-based guard, not an exact invoice guarantee. Allow additional local retrieval '
        'and verification time.', '',
        'Readiness: bounded local tests verify finite fractional strengths, exact sampler endpoints '
        'and actual midpoint evaluator routes at two diffusion steps. These are software checks, '
        'not efficacy results. The complete staged package must pass before launch. Local '
        'preparation does not infer approval or start paid compute.','']
    (output/'PROPOSAL.md').write_text('\n'.join(text),encoding='utf-8')
    offline_check(output,project_path(plan['archive']),plan)
    evidence=[analysis/'complete.json',analysis/'comparisons.csv',preflight/'complete.json',
        preflight/'inputs.json',remote_complete,ROOT/'scripts/v8_generator_dev/prepare_strength.py',
        output/'modal_strength.py',output/'offline_package.json']
    evidence.extend(output/'code'/name for name in patches)
    json_write(output/'readiness_inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in evidence})
    json_write(output/'complete.json',dict(complete=True,readiness_passed=True,budget_approved=False,
        proposed_new_allowance_usd=3,cloud_launched=False,training_planned=False,anatomy_validated=False))
    print('STRENGTH SWEEP PREPARED',output,flush=True)


def launch(prepared,report_id,approved_usd):
    if approved_usd!=3:
        raise ValueError('Explicit separate additional US$3 approval required')
    prepared=project_path(prepared)
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    if not json.loads((prepared/'complete.json').read_text())['readiness_passed']:
        raise ValueError('Completed local readiness required')
    environment=dict(os.environ,CAPSTONE_STRENGTH_PREPARED=str(prepared),PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    inventory=subprocess.run([sys.executable,'-m','modal','app','list','--profile','namankundra7','--json'],
        cwd=ROOT,env=environment,capture_output=True,text=True,check=True)
    apps=json.loads(inventory.stdout)
    if any(app['state']!='stopped' for app in apps):
        raise ValueError('Another Modal app is active')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    json_write(output/'approval.json',dict(new_explicit_user_approved_usd=3,previous_allowances_reset=False,
        prepared=str(prepared.relative_to(ROOT)),scope='One-A10 validation-only strength sensitivity, maximum 75 worker minutes'))
    json_write(output/'apps_before.json',apps)
    shutil.copyfile(prepared/'plan.json',output/'plan.json')
    command=[sys.executable,'-m','modal','run','--profile','namankundra7',str(prepared/'modal_strength.py'),
        '--approved-usd','3']
    json_write(output/'command.json',dict(command=command))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.prepare_strength launch. approval/plan/command and '
        'apps_before retain the separate bounded launch. launch.log/process_result.json capture '
        'actual output/CLI wall time. No training, automatic paid restart or clinical promotion.\n',encoding='utf-8')
    start=time.monotonic()
    with (output/'launch.log').open('x',encoding='utf-8') as handle:
        result=subprocess.run(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
    json_write(output/'process_result.json',dict(exit_code=result.returncode,seconds=time.monotonic()-start,
        approved_additional_usd=3,exact_invoice_known=False))
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['prepare','launch'])
    p.add_argument('--run-id')
    p.add_argument('--report-id',required=True)
    p.add_argument('--prepared')
    p.add_argument('--approved-usd',type=float)
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.run_id,a.report_id)
    else:
        raise SystemExit(launch(a.prepared,a.report_id,a.approved_usd))
