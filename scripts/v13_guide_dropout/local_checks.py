"""Actual final-parent CUDA updates/resumes; six discarded batch-five copies only."""
import argparse
import gc
import json
import subprocess
import sys
import time

import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,json_write,digest
from scripts.v13_guide_dropout.engine import Trainer
from scripts.v13_guide_dropout.protocol import make_protocol,copy_key


def equal(a,b):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise ValueError('Resumed tensor differs')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Resumed keys differ')
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Resumed lengths differ')
        for x,y in zip(a,b):equal(x,y)
    elif a!=b:raise ValueError('Resumed state differs')


def run(run_id):
    output=ROOT/'reports/v13_guide_dropout'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.local_checks. protocol.json: '
        'new full-exposure design; tests.txt/json: CPU resume/failure/design tests; updates.jsonl: '
        'actual discarded final-parent CUDA draws; checks.json: each copy and resumed optimiser parity; '
        'inputs.json: source/parent/protocol hashes; complete.json: local engineering scope. '
        'Temporary engineering snapshots are discarded, not efficacy checkpoints.\n',encoding='utf-8')
    p=make_protocol(json.loads((ROOT/'reports/v10_detail/preparation_20261117_02/protocol.json').read_text()))
    json_write(output/'protocol.json',p)
    with (output/'tests.txt').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','unittest','scripts.v13_guide_dropout.test_training','-v'],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    json_write(output/'tests.json',dict(exit_code=result.returncode))
    if result.returncode:raise ValueError('Training regression checks failed')
    deterministic_runtime();torch.set_num_threads(2)
    train=CanonicalDataset(p['reference_protocol']['view'],'train')
    started=time.monotonic();checks=[];matched={}
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='dropout_engineering_') as temporary:
        from pathlib import Path
        with (output/'updates.jsonl').open('x',encoding='utf-8') as log:
            for seed in p['reference_protocol']['training_seeds']:
                for mode in p['modes']:
                    if time.monotonic()-started>300:raise TimeoutError('Local check cap')
                    torch.cuda.reset_peak_memory_stats()
                    engine=Trainer(p,train,mode,seed,'cuda',engineering=True,batch_override=5)
                    row1=engine.batch();log.write(json.dumps(row1)+'\n');log.flush()
                    snapshot=Path(temporary)/(copy_key(seed,mode)+'.pt');engine.save(snapshot)
                    row2=engine.batch();log.write(json.dumps(row2)+'\n');log.flush()
                    engine.frozen_check()
                    expected_adapter={k:v.detach().cpu().clone() for k,v in engine.model.adapter.state_dict().items()}
                    expected_optimizer=engine.optimizer.state_dict()
                    expected_counts=list(engine.gate_counts);expected_chain=engine.draw_chain
                    peak=torch.cuda.max_memory_allocated()
                    del engine;gc.collect();torch.cuda.empty_cache()
                    restored=Trainer(p,train,mode,seed,'cuda',engineering=True,batch_override=5)
                    restored.restore(snapshot);replay=restored.batch();restored.frozen_check()
                    equal(expected_adapter,{k:v.detach().cpu() for k,v in restored.model.adapter.state_dict().items()})
                    equal(expected_optimizer,restored.optimizer.state_dict())
                    if expected_counts!=restored.gate_counts or expected_chain!=restored.draw_chain:
                        raise ValueError('Gate/cursor resume differs')
                    keys=('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states','actual_draw_chain_sha256')
                    draws=[{k:r[k] for k in keys} for r in (row1,row2)]
                    if {k:row2[k] for k in keys}!={k:replay[k] for k in keys}:raise ValueError('Actual resumed draw differs')
                    if seed in matched and matched[seed]!=draws:raise ValueError('Mode draws differ')
                    matched[seed]=draws
                    checks.append(dict(key=copy_key(seed,mode),batch_size=5,applied_updates_in_retained_trajectory=2,
                        replayed_disposable_updates=1,exact_resume=True,peak_allocated_bytes=peak,
                        frozen_weights=True,gate_counts=restored.gate_counts,runtime=restored.runtime))
                    del restored,expected_adapter,expected_optimizer;gc.collect();torch.cuda.empty_cache()
    json_write(output/'checks.json',checks)
    paths=list((ROOT/'scripts/v13_guide_dropout').glob('*.py'))+[ROOT/p['reference_protocol']['parent'],output/'protocol.json']
    json_write(output/'inputs.json',{f.relative_to(ROOT).as_posix():digest(f) for f in paths})
    json_write(output/'complete.json',dict(complete=True,engineering_only=True,tests_passed=4,
        copies=len(checks),actual_disposable_updates=18,matched_actual_draws=True,
        full_exposure_resume_engine_tested=True,final_parent_cuda_engineering_resume_exact=True,
        efficacy_checkpoints_saved=False,intended_runtime_batch32_pending=True,
        paid_cloud_started=False,seconds=time.monotonic()-started))
    print('DROPOUT LOCAL CHECKS COMPLETE',output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run-id',required=True)
    run(ap.parse_args().run_id)
