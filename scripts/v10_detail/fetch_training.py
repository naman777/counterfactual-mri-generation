"""Retrieve and audit completed training while evaluation continues; no success fiction."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch

from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path
from scripts.v10_detail.verify_training_logs import check_records
from scripts.v10_detail.protocol import copy_key,canonical_hash


def fetch(prepared,run_id,report_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    if not all(s.replace('_','').isalnum() for s in (run_id,report_id)):raise ValueError('Safe identifiers')
    os.environ['MODAL_PROFILE']='namankundra7'
    import modal
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    base='v10_detail/'+run_id;relative='reports/v10_detail/'+run_id+'_training'
    def read(path):return b''.join(volume.read_file(path))
    trained=json.loads(read(base+'/project/'+relative+'/complete.json'))
    phase=json.loads(read(base+'/train.json'))
    if phase['exit_code'] or trained['engineering'] or trained['common_step']!=2780 or trained['common_epoch']!=20:
        raise ValueError('Completed full-exposure training component required')
    expected={copy_key(seed,mode) for seed in p['training_seeds'] for mode in p['modes']}
    if set(trained['copies'])!=expected:raise ValueError('Nine completed copies required')
    output=ROOT/'reports/v10_detail'/report_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.fetch_training. phase_receipt.json: original training exit/time; '
        'manifest.json: original training reports/final checkpoints retrieved at fresh canonical paths with SHA256; '
        'verification.json: all actual update/journal/checkpoint checks; inputs.json: exact local files/checker code; '
        'complete.json: training-component verification only, explicitly not full workflow/evaluation success. '
        'No GPU app, restart or scientific/default promotion.\n',encoding='utf-8')
    json_write(output/'phase_receipt.json',phase)
    entries=volume.listdir(base+'/project/'+relative,recursive=True)
    tasks=[]
    for entry in entries:
        name=entry.path.lstrip('/')
        if '.' not in name.rsplit('/',1)[-1]:continue
        rel=name.removeprefix(base+'/project/')
        tasks.append((name,project_path(rel)))
    for copy in trained['copies'].values():
        rel=copy['checkpoint'].replace('\\','/');tasks.append((base+'/project/'+rel,project_path(rel)))
    def retrieve(task):
        remote,path=task;data=read(remote);sha=hashlib.sha256(data).hexdigest()
        if path.exists():
            if digest(path)!=sha:raise ValueError('Immutable training evidence collision')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(data)
        return dict(remote=remote,local=path.relative_to(ROOT).as_posix(),sha256=sha,bytes=len(data))
    with ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(retrieve,tasks))
    training=ROOT/relative;frame=pd.read_csv(project_path(p['view'])/'splits/train.csv')
    start=0;count=0
    for marker in sorted(training.glob('block_*/complete.json')):
        block=json.loads(marker.read_text());stop=block['common_step']
        if block['protocol_sha256']!=canonical_hash(p) or set(block['copies'])!=expected or stop-start!=695:
            raise ValueError('Common exposure/protocol/copy set drift')
        by_seed={}
        for key,copy in block['copies'].items():
            rows=[json.loads(s) for s in (marker.parent/(key+'.jsonl')).read_text().splitlines()]
            draws=check_records(rows,start,stop,frame,copy['training_seed'],copy['mode'])
            seed=copy['training_seed']
            if seed in by_seed and by_seed[seed]!=draws:raise ValueError('Actual recorded mode draws differ')
            by_seed[seed]=draws;count+=len(rows)
        start=stop
    if start!=2780 or count!=25020:raise ValueError('Incomplete journal exposure')
    for key,copy in trained['copies'].items():
        path=project_path(copy['checkpoint'])
        if digest(path)!=copy['sha256']:raise ValueError('Final checkpoint changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['protocol']!=p or ck['engineering'] or ck['step']!=2780 or ck['parent_sha256']!=p['parent_sha256']:
            raise ValueError('Final raw-state identity/exposure mismatch')
        if copy_key(ck['training_seed'],ck['mode'])!=key:raise ValueError('Final copy identity mismatch')
        if not all(torch.isfinite(v).all() for v in ck['adapter'].values()):raise ValueError('Nonfinite final adapter')
        if not all(torch.isfinite(v).all() for state in ck['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite optimiser state')
        del ck
    json_write(output/'manifest.json',files)
    evidence={r['local']:r['sha256'] for r in files}
    for path in (project_path(__file__),ROOT/'scripts/v10_detail/verify_training_logs.py',prepared/'protocol.json'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'verification.json',dict(passed=True,completed_training_copies=9,
        actual_logged_updates_checked=count,steps_per_copy=2780,all_final_states_finite=True,
        matched_recorded_mode_draws=True,source_membership_and_objective_checked=True,
        raw_noise_arrays_regenerated=False))
    json_write(output/'complete.json',dict(complete=True,training_components_verified=True,
        workflow_success=False,evaluation_completion_not_asserted=True,clinical_preservation='unknown',
        paid_gpu_calls=0,run_id=run_id))
    print('TRAINING COMPONENT VERIFIED',count,'actual updates; evaluation status separate',flush=True)
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','run-id','report-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();fetch(a.prepared,a.run_id,a.report_id)
