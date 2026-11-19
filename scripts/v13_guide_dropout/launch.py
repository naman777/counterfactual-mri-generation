"""One authorised app only, after exact local readiness and remaining-budget guards."""
import argparse
import json
import os
import subprocess
import sys

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs


def launch(prepared,local_checks,preflight,followup,report_id):
    prepared,local_checks,preflight,followup=map(project_path,(prepared,local_checks,preflight,followup))
    ready=json.loads((prepared/'complete.json').read_text())
    if not ready['complete'] or ready['cloud_launched']:raise ValueError('Prepared unlaunched package required')
    for home in (prepared,followup):verify_inputs(json.loads((home/'readiness_inputs.json').read_text()))
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    for home in (local_checks,preflight):
        complete=json.loads((home/'complete.json').read_text())
        if not complete['complete'] or not complete['engineering_only']:raise ValueError('Completed local checks required')
        verify_inputs(json.loads((home/'inputs.json').read_text()))
    plan=json.loads((prepared/'plan.json').read_text())
    if plan['working_limit_usd']!=12 or plan['total_remaining_credit_ceiling_usd']!=13:
        raise ValueError('User-authorised remaining-credit identity changed')
    authority=ROOT/'reports/v12_modal_local/budget_20261118_01/authorisation.json'
    approved=json.loads(authority.read_text())
    if approved['conservative_working_limit_usd']!=12 or approved['total_new_modal_spending_ceiling_usd']!=13:
        raise ValueError('Stored user authority mismatch')
    env=dict(os.environ,CAPSTONE_DROPOUT_PREPARED=str(prepared),PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    def read(args):
        result=subprocess.run([sys.executable,'-m','modal',*args,'--json','--profile','namankundra7'],
            cwd=ROOT,env=env,capture_output=True,text=True,check=True,timeout=60)
        return json.loads(result.stdout)
    apps=read(['app','list'])
    if any(app['state']!='stopped' for app in apps):raise ValueError('Another workspace app is active')
    billing=read(['billing','summary','--for','2026-11'])
    if float(billing['metered_cost'])-float(plan['billing_baseline_metered_cost'])>.25:
        raise ValueError('Unexpected intervening usage; do not risk the credit reserve')
    if not report_id.replace('_','').isalnum():raise ValueError('Safe fresh launch report ID')
    output=ROOT/'reports/v13_guide_dropout'/report_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.launch. authority.json: '
        'recorded scope, total credit ceiling and working limit; apps_before.json and billing_before.json: '
        'live no-other-app/intervening-usage checks; command.json: exact frozen worker invocation; '
        'result.json: actual CLI termination. No automatic paid retry.\n',encoding='utf-8')
    json_write(output/'authority.json',dict(user_record=authority.relative_to(ROOT).as_posix(),sha256=digest(authority),
        working_limit_usd=12,total_remaining_ceiling_usd=13,reserve_outside_working_limit_usd=1,
        automatic_paid_retry=False,prepared=prepared.relative_to(ROOT).as_posix(),
        local_checks=local_checks.relative_to(ROOT).as_posix(),preflight=preflight.relative_to(ROOT).as_posix(),
        followup=followup.relative_to(ROOT).as_posix(),maximum_worker_seconds=plan['maximum_worker_seconds']))
    json_write(output/'apps_before.json',apps);json_write(output/'billing_before.json',billing)
    command=[sys.executable,'-m','modal','run','--profile','namankundra7',str(prepared/'modal_worker.py'),
             '--working-limit-usd','12']
    json_write(output/'command.json',dict(command=command))
    result=subprocess.run(command,cwd=ROOT,env=env)
    json_write(output/'result.json',dict(exit_code=result.returncode,automatic_paid_retry=False))
    if result.returncode:raise RuntimeError('Cloud workflow stopped with an error; retrieve evidence without paid restart')


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','local-checks','preflight','followup','report-id'):ap.add_argument('--'+name,required=True)
    launch(**vars(ap.parse_args()))
