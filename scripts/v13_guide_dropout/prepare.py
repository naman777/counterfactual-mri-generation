"""Freeze the exact six-copy package and current resource limits before paid work."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import PurePosixPath

from scripts.v8_generator_dev.io import ROOT,digest,json_write,verify_inputs
from scripts.v13_guide_dropout.protocol import make_protocol,canonical_hash


def prepare(report_id):
    if not report_id.replace('_','').isalnum():raise ValueError('Safe fresh ID')
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.prepare. protocol.json: '
        'single-factor science and ten gates; plan.json: bounded resource/credit plan; '
        'pricing.json and billing_*.json: current official CLI rates and billing observation; '
        'code/: exact dependency overlays; inputs.json: staged input hashes; '
        'offline_package.txt/json: exact staged import/data/protocol check; '
        'readiness_inputs.json: prepared artifact hashes; complete.json: package scope only. '
        'Target-runtime full-batch gate and successful local preflight are mandatory before efficacy.\n',encoding='utf-8')
    prior=ROOT/'reports/v10_detail/preparation_20261117_02'
    verify_inputs(json.loads((prior/'readiness_inputs.json').read_text()))
    oldplan=json.loads((prior/'plan.json').read_text())
    p=make_protocol(json.loads((prior/'protocol.json').read_text()))
    json_write(out/'protocol.json',p)
    for name in ('validation_selection.csv','mask_thresholds.json'):
        shutil.copyfile(prior/name,out/name)
    raw={}
    for what,command in [('billing_summary',['billing','summary','--for','2026-11']),('billing_rates',['billing','rates']),
                         ('apps_before',['app','list'])]:
        checked=subprocess.run([sys.executable,'-m','modal',*command,'--json','--profile','namankundra7'],
                               cwd=ROOT,capture_output=True,text=True,check=True,timeout=60)
        raw[what]=json.loads(checked.stdout);json_write(out/(what+'.json'),raw[what])
    if any(app['state']!='stopped' for app in raw['apps_before']):raise ValueError('Another app is active')
    rates=raw['billing_rates']
    rate=1.2*(max(float(rates['gpu_hour_cost_a10g'])/3600,.000306)+
              2*max(float(rates['cpu_hour_cost'])/3600,.0000131)+16*max(float(rates['mem_gib_hour_cost'])/3600,.00000222))
    json_write(out/'pricing.json',dict(checked_date_local='2026-11-18',source='https://modal.com/pricing',
        corroborating_source='modal billing rates --json',conservative_worker_rate_per_second=rate,
        margin=1.2,reserve_within_working_limit_usd=1.,worker_cap_seconds=23400,
        conservative_cap_estimate_usd=23400*rate+1.,actual_invoice_unknown=True))
    if 23400*rate+1>12:raise ValueError('Resource cap does not fit working allowance')
    archive=ROOT/oldplan['archive']
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('_payload_manifest.json'))
    patches={}
    for name,expected in oldplan['patches'].items():
        source=prior/'code'/name
        if digest(source)!=expected:raise ValueError('Historical prepared overlay changed')
        target=out/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        patches[name]=expected
    files=list((ROOT/'scripts/v13_guide_dropout').glob('*.py'))
    files += [ROOT/'scripts/v11_guide_audit/guide_dropout.py',ROOT/'scripts/v11_guide_audit/analyse.py']
    for source in files:
        ast.parse(source.read_text(encoding='utf-8-sig'))
        name=source.relative_to(ROOT).as_posix();target=out/'code'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);patches[name]=digest(source)
    reference=p['reference_protocol']
    inputs={**manifest,**patches,reference['parent']:reference['parent_sha256']}
    for name in ('protocol.json','validation_selection.csv','mask_thresholds.json'):
        inputs[(out/name).relative_to(ROOT).as_posix()]=digest(out/name)
    json_write(out/'inputs.json',inputs)
    plan=dict(format='bounded_remaining_credit_plan_v1',run_id='guide_dropout_20261118_01',
        prepared_relative=out.relative_to(ROOT).as_posix(),archive=oldplan['archive'],archive_sha256=digest(archive),
        patches=patches,parent=reference['parent'],parent_sha256=reference['parent_sha256'],
        source_parent_project=oldplan['source_parent_project'],protocol_sha256=canonical_hash(p),
        total_remaining_credit_ceiling_usd=13,working_limit_usd=12,unallocated_credit_buffer_usd=1,
        price_with_margin=rate,reserve_usd=1.,maximum_worker_seconds=23400,maximum_training_seconds=9000,
        maximum_full_batch_gate_seconds=900,one_app=True,one_gpu=True,automatic_paid_retry=False,
        billing_baseline_metered_cost=raw['billing_summary']['metered_cost'],
        authorised_instruction='okay i only have 13 dollar credits after that use my laptop',
        local_fallback='Complete cloud training first; unfinished profiles may be evaluated locally; no silent cross-runtime training resume')
    json_write(out/'plan.json',plan)
    shutil.copyfile(ROOT/'scripts/v13_guide_dropout/modal_worker.py',out/'modal_worker.py')
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='dropout_package_') as temporary:
        from pathlib import Path
        staged=Path(temporary)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(manifest)|{'_payload_manifest.json'}:
                raise ValueError('Archive membership changed')
            for name,expected in manifest.items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:raise ValueError('Unsafe member')
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive bytes changed')
                target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        for name in patches:
            target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out/'code'/name,target)
        target=staged/plan['parent'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/plan['parent'],target)
        dest=staged/plan['prepared_relative'];dest.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','mask_thresholds.json','plan.json'):
            shutil.copyfile(out/name,dest/name)
        command=('import json,sys; from pathlib import Path; '
            'from scripts.v8_generator_dev.io import verify_inputs; from scripts.v8_generator_dev.data import CanonicalDataset; '
            'from scripts.v13_guide_dropout.protocol import validate; from scripts.v13_guide_dropout.engine import Trainer; '
            'from scripts.v13_guide_dropout.evaluate import evaluate; from scripts.v13_guide_dropout.verify_saved import verify; '
            'p=Path(sys.argv[1]); verify_inputs(json.loads((p/"inputs.json").read_text())); '
            'v=json.loads((p/"protocol.json").read_text()); validate(v); '
            'a=CanonicalDataset(v["reference_protocol"]["view"],"train"); b=CanonicalDataset(v["reference_protocol"]["view"],"val"); '
            'assert len(a)==4434 and len(b)==987 and a.frame.patient_id.nunique()==72 and b.frame.patient_id.nunique()==15; '
            'print("EXACT DROPOUT PACKAGE, INPUTS, TRAIN/VAL AND IMPORTS PASS")')
        with (out/'offline_package.txt').open('x',encoding='utf-8') as log:
            checked=subprocess.run([sys.executable,'-c',command,plan['prepared_relative']],cwd=staged,
                env=dict(os.environ,PYTHONPATH=str(staged)),stdout=log,stderr=subprocess.STDOUT,timeout=600)
        json_write(out/'offline_package.json',dict(exit_code=checked.returncode,exact_staged_inputs=True))
        if checked.returncode:raise ValueError('Exact staged package failed')
    paths=[out/name for name in ('protocol.json','plan.json','inputs.json','pricing.json','billing_summary.json',
        'billing_rates.json','apps_before.json','validation_selection.csv','mask_thresholds.json','modal_worker.py','offline_package.json')]
    paths += [out/'code'/name for name in patches]
    json_write(out/'readiness_inputs.json',{f.relative_to(ROOT).as_posix():digest(f) for f in paths})
    json_write(out/'complete.json',dict(complete=True,exact_offline_package_passed=True,cloud_launched=False,
        local_preflight_required=True,intended_runtime_full_batch_gate_pending=True,
        authorised_total_ceiling_usd=13,working_limit_usd=12,efficacy_started=False))
    print('BOUNDED DROPOUT PACKAGE PREPARED',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--report-id',required=True)
    prepare(ap.parse_args().report_id)
