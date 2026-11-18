"""Bounded full-code local integration; never starts or grants a paid run."""
import argparse
import gc
import json
import subprocess
import sys

import numpy as np
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v10_detail.engine import Trainer
from scripts.v10_detail.evaluate import evaluate


def run(prepared,run_id):
    prepared=project_path(prepared)
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    p=json.loads((prepared/'protocol.json').read_text())
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.local_preflight. protocol.json: exact local scope; '
        'full_batch_stdout.txt/json: isolated local batch32 backward attempt, including failure if any; '
        'checks.json: new engine updates/evaluator routes and exact absent/zero controls; inputs.json: all consumed '
        'package/new snapshot/result hashes; complete.json: engineering scope. Referenced new reports retain '
        'all actual arrays and named-judge grades. No efficacy training or paid work.\n',encoding='utf-8')
    json_write(output/'protocol.json',dict(prepared=prepared.relative_to(ROOT).as_posix(),
        scope='One discarded local full-batch attempt, then batch2 full-code route checks',
        training_seed=p['training_seeds'][0],engineering=True,target_runtime_gate_not_waived=True))
    with (output/'full_batch_stdout.txt').open('x',encoding='utf-8') as log:
        full=subprocess.run([sys.executable,'-m','scripts.v10_detail.runtime_smoke','--prepared',str(prepared),
            '--run-id',run_id+'_batch32','--local-single'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    json_write(output/'full_batch_result.json',dict(exit_code=full.returncode,
        report=f'reports/v10_detail/{run_id}_batch32',target_runtime_gate_still_required=True))
    deterministic_runtime();train=CanonicalDataset(p['view'],'train');results={};reference=None
    checkpoints=ROOT/'checkpoints/diffusion/ax_t1_128_v10_detail'/run_id;checkpoints.mkdir(parents=True,exist_ok=False)
    evidence={path.relative_to(ROOT).as_posix():digest(path) for path in
        (prepared/'protocol.json',prepared/'readiness_inputs.json',ROOT/'scripts/v10_detail/local_preflight.py')}
    for mode in p['modes']:
        engine=Trainer(p,train,mode,p['training_seeds'][0],'cuda',engineering=True,batch_override=2)
        records=[engine.batch(),engine.batch()];engine.frozen_check()
        draws=[{k:r[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256')} for r in records]
        if reference is None:reference=draws
        elif reference!=draws:raise AssertionError('Full-code mode draws differ')
        snapshot=checkpoints/(mode+'.pt');engine.save(snapshot)
        evidence[snapshot.relative_to(ROOT).as_posix()]=digest(snapshot)
        del engine;gc.collect();torch.cuda.empty_cache()
        evaluations={}
        for axis,modes in p['controls'].items():
            for control in modes[mode]:
                identity=run_id+'_'+mode+'_'+axis+'_'+control
                home=evaluate(prepared,snapshot,identity,axis,control,'cuda',engineering=True)
                evaluations[axis+'__'+control]=home.relative_to(ROOT).as_posix()
                for name in ('protocol.json','requests.csv','per_image.csv','complete.json'):
                    evidence[(home/name).relative_to(ROOT).as_posix()]=digest(home/name)
        results[mode]=dict(records=records,checkpoint=snapshot.relative_to(ROOT).as_posix(),evaluations=evaluations)
    # New full evaluator's absent route must be independent of each trained adapter.
    absent=[]
    for mode in p['modes']:
        home=ROOT/results[mode]['evaluations']['manufacturer__absent']
        import pandas as pd
        requests=pd.read_csv(home/'requests.csv')
        images=[]
        for r in requests.itertuples():
            with np.load(home/r.array_file) as z:images.append((z['null'].copy(),z['target'].copy()))
        absent.append(images)
    for images in absent[1:]:
        for a,b in zip(absent[0],images):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    json_write(output/'checks.json',dict(modes=results,matched_actual_draws=True,
        absent_images_equal_across_trained_modes=True,full_batch_local_exit_code=full.returncode))
    verify_inputs(evidence);json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,engineering=True,modes=3,updates=6,
        evaluator_runs=sum(len(r['evaluations']) for r in results.values()),
        local_full_batch_backward_passed=full.returncode==0,pinned_a10_full_batch_pending=True,
        paid_compute=False,efficacy_started=False,clinical_preservation='unknown'))
    print('FULL CODE LOCAL PREFLIGHT COMPLETE',output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();run(a.prepared,a.run_id)
