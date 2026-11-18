"""Launch only after an explicit new allowance; never infer permission from leftovers."""
import argparse
import json
import os
import subprocess
import sys

from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path,verify_inputs


def launch(prepared,report_id,approved_usd):
    if approved_usd!=12:raise ValueError('Explicit separate additional US$12 user approval required')
    prepared=project_path(prepared)
    ready=json.loads((prepared/'complete.json').read_text())
    if not ready['local_package_prepared'] or ready['cloud_launched']:raise ValueError('Prepared unlaunched package required')
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    environment=dict(os.environ,CAPSTONE_DETAIL_PREPARED=str(prepared),PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    inventory=subprocess.run([sys.executable,'-m','modal','app','list','--profile','namankundra7','--json'],
        cwd=ROOT,env=environment,capture_output=True,text=True,check=True)
    apps=json.loads(inventory.stdout)
    if any(app['state']!='stopped' for app in apps):raise ValueError('Another workspace Modal app is active')
    if not report_id.replace('_','').isalnum():raise ValueError('Safe new launch report ID')
    output=ROOT/'reports/v10_detail'/report_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.launch. approval.json: new explicit allowance/scope; '
        'apps_before.json: no-other-active-app gate; command.json: exact frozen worker invocation; '
        'result.json: observed CLI termination. Failed execution is not automatically retried.\n',encoding='utf-8')
    json_write(output/'approval.json',dict(approved_additional_usd=12,prepared=prepared.relative_to(ROOT).as_posix(),
        proposal_sha256=digest(prepared/'PROPOSAL.md'),previous_allowances_reset=False,
        maximum_worker_seconds=23400,one_app=True,one_a10=True,automatic_paid_restart=False))
    json_write(output/'apps_before.json',apps)
    command=[sys.executable,'-m','modal','run','--profile','namankundra7',str(prepared/'modal_worker.py'),'--approved-usd','12']
    json_write(output/'command.json',dict(command=command))
    result=subprocess.run(command,cwd=ROOT,env=environment)
    json_write(output/'result.json',dict(exit_code=result.returncode,automatic_paid_restart=False))
    if result.returncode:raise RuntimeError('Cloud workflow failed; preserve terminal receipts and do not restart')


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True)
    ap.add_argument('--report-id',required=True);ap.add_argument('--approved-usd',required=True,type=float)
    a=ap.parse_args();launch(a.prepared,a.report_id,a.approved_usd)
