"""Test the training-record auditor against already persisted real cloud segments."""
import argparse
import hashlib
import json
import subprocess
import sys

import torch

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v13_guide_dropout.verify_training import audit_rows


def run(fetched,report_id):
    fetched=project_path(fetched);verify_inputs(json.loads((fetched/'inputs.json').read_text()))
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.audit_preflight. tests.txt/json: '
        'corrupt/missing/unapplied update checks; checks.json: actual persisted cloud segment/cursor/gate '
        'and checkpoint-chain equality for the first matched pair; inputs.json: bound bytes; '
        'complete.json: partial real-log auditor preflight, not completed training exposure.\n',encoding='utf-8')
    with (out/'tests.txt').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','unittest','scripts.v13_guide_dropout.test_training_audit','-v'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    json_write(out/'tests.json',dict(exit_code=result.returncode,tests_passed=1 if result.returncode==0 else None))
    if result.returncode:raise ValueError('Training auditor regression failed')
    home=ROOT/'reports/v13_guide_dropout/guide_dropout_20261118_01_training/segments'
    paths={mode:sorted(home.glob('seed583301__'+mode+'_step*/complete.json')) for mode in ('coarse_control','coarse_fine')}
    if any(not v for v in paths.values()):raise ValueError('A matched persisted cloud pair is required')
    common=min(max(json.loads(p.read_text())['global_step'] for p in v) for v in paths.values())
    checks=[];evidence={};chains=[]
    for mode,markers in paths.items():
        step=0;chain=hashlib.sha256(b'guide_dropout_actual_draw_chain_v1').hexdigest();counts=[0]*5
        for marker in markers:
            record=json.loads(marker.read_text())
            if record['global_step']>common:continue
            log=marker.parent/'updates.jsonl';rows=[json.loads(line) for line in log.read_text().splitlines()]
            step,chain,counts=audit_rows(rows,583301,mode,step,chain,counts)
            checkpoint=project_path(record['checkpoint'])
            if digest(checkpoint)!=record['sha256']:raise ValueError('Persisted checkpoint hash changed')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['step']!=step or state['gate_counts']!=counts or state['actual_draw_chain_sha256']!=chain:
                raise ValueError('Actual cloud segment and saved state disagree')
            for path in (marker,log,checkpoint):evidence[path.relative_to(ROOT).as_posix()]=digest(path)
            del state
        chains.append(chain);checks.append(dict(mode=mode,actual_updates_checked=step,counts=counts,chain=chain))
    if chains[0]!=chains[1]:raise ValueError('Real matched cloud streams differ')
    json_write(out/'checks.json',checks);json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,partial_actual_log_check=True,
        actual_updates_checked=sum(r['actual_updates_checked'] for r in checks),
        common_applied_step_in_first_seed=common,matched_cloud_streams=True,
        full_training_complete=False,no_paid_job_started=True))
    print('REAL CLOUD TRAINING AUDITOR PREFLIGHT PASS',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--fetched',required=True);ap.add_argument('--report-id',required=True)
    run(**vars(ap.parse_args()))
