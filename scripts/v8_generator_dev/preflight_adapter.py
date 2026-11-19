"""Bounded real-parent pilot integration; never authorises or launches cloud work."""
import argparse
import json
import subprocess
import sys

import torch

from scripts.v8_generator_dev.adapter_pilot import MODES, pilot_settings, train_copy
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import CachedCanonical, deterministic_runtime
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def run(report_id):
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.preflight_adapter. tests.txt retains pilot and '
        'CFG/guide controls; tests.json records actual exit. *_steps.jsonl and checks.json retain '
        'two actual adapter updates per guide/backbone on epoch-70 parents, with batch two '
        'and all frozen-weight audits. New engineering-only checkpoints are in checkpoints/. '
        'inputs.json binds dependencies. This local preflight neither trains the final parents '
        'nor establishes editing/anatomy efficacy or GPU spending approval.\n',encoding='utf-8')
    command=[sys.executable,'-m','unittest','scripts.v8_generator_dev.test_adapter_pilot',
        'scripts.v8_generator_dev.test_soft_structure','scripts.v8_generator_dev.test_precision_step','-v']
    with (output/'tests.txt').open('x',encoding='utf-8') as handle:
        tests=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    json_write(output/'tests.json',dict(command=command,exit_code=tests.returncode))
    if tests.returncode:
        raise RuntimeError('Local adapter/guide/precision tests failed')
    deterministic_runtime()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    receipt=ROOT/'reports/v8_generator_dev/failed_pair_checkpoints_20261116_01/complete.json'
    parents=json.loads(receipt.read_text())['checkpoints']
    view='data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    dataset=CachedCanonical(CanonicalDataset(view,'train'))
    settings=pilot_settings()
    settings['batch_size']=2  # Explicit engineering override; full pilot remains batch 32.
    ckdir=ROOT/'checkpoints/diffusion/ax_t1_128_v8_generator_dev'/report_id
    ckdir.mkdir(exist_ok=False)
    checks={}
    inputs={receipt.relative_to(ROOT).as_posix():digest(receipt)}
    for backbone,parent in parents.items():
        for mode in MODES:
            key=backbone+'__'+mode
            checks[key]=train_copy(parent,mode,dataset,settings,ckdir/(key+'.pt'),output/(key+'_steps.jsonl'),
                stop_step=2,device=device,engineering=True)
        inputs[project_path(parent).relative_to(ROOT).as_posix()]=digest(project_path(parent))
    for name in ('adapter_pilot.py','evaluate_adapter.py','soft_structure.py','preflight_adapter.py','test_adapter_pilot.py'):
        path=ROOT/'scripts/v8_generator_dev'/name
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for path in (ROOT/view/'splits/train.csv',ROOT/view/'splits/val.csv'):
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'checks.json',checks)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,local_tests_passed=True,real_parent_updates_finite=True,
        matched_engineering_steps_per_copy=2,engineering_batch_size=2,full_batch_tested=False,
        parent_epoch=70,final_parent_verification_still_required=True,anatomy_validated=False,cloud_launched=False))
    print('ADAPTER LOCAL PREFLIGHT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report-id',required=True)
    a=p.parse_args()
    run(a.report_id)
