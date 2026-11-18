"""Twelve disposable local updates test all candidate gate states; not efficacy."""
import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write
from scripts.v10_detail.engine import Trainer
from scripts.v11_guide_audit.guide_dropout import EngineeringGate,STATE_NAMES,STATE_PROBABILITIES


def run(run_id):
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    p=json.loads((ROOT/'reports/v10_detail/preparation_20261117_02/protocol.json').read_text())
    protocol=dict(format='guide_dropout_engineering_v1',base_protocol=p,updates_per_copy=2,batch_size=5,
        training_seeds=p['training_seeds'],modes=['coarse_control','coarse_fine'],state_names=STATE_NAMES,
        proposed_probabilities=STATE_PROBABILITIES,actual_engineering_policy='Force each of five states once per batch for coverage',
        no_efficacy_checkpoint=True,maximum_local_seconds=300,paid_gpu_calls=0)
    json_write(output/'protocol.json',protocol)
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.engineering. protocol.json: disposable local '
        'scope and forced all-state coverage (not efficacy/random-policy exposure); updates.csv: every actual batch/loss/precision/draw/gate record; '
        'tests.txt: gate semantics/RNG/analytical forward-response tests; inputs.json: code/parent/protocol identities; '
        'complete.json: matched draws, frozen weights, optimiser identity and finite applied updates. '
        'No efficacy weights saved. Full-exposure versioned checkpoint/resume/packaging and pinned-runtime checks remain required before training.\n',encoding='utf-8')
    started=time.monotonic();deterministic_runtime();torch.set_num_threads(2)
    dataset=CanonicalDataset(p['view'],'train');rows=[];copy_results=[]
    for seed in p['training_seeds']:
        reference=None
        for mode in protocol['modes']:
            if time.monotonic()-started>protocol['maximum_local_seconds']:raise TimeoutError('Engineering cap')
            engine=Trainer(p,dataset,mode,seed,'cuda',True,batch_override=5)
            engine.model=EngineeringGate(engine.model,seed,force_coverage=True)
            expected={id(v) for v in engine.model.adapter.parameters()}
            actual={id(v) for group in engine.optimizer.param_groups for v in group['params']}
            if expected!=actual:raise ValueError('Optimiser lost adapter identity')
            before=state_digest(engine.model.adapter);draws=[]
            for step in range(2):
                engine.model.step=step
                row=engine.batch()
                if sorted(engine.model.last_states)!=list(range(5)):raise ValueError('Missing forced gate state')
                row['gate_states']=engine.model.last_states;rows.append(row)
                draws.append({name:row[name] for name in ('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states')})
                if not row['applied'] or not all(np.isfinite(row[k]) for k in ('total_loss','gradient_norm','noise_mse','coarse_structure_loss')):
                    raise ValueError('Invalid engineering update')
            engine.frozen_check()
            if reference is not None and reference!=draws:raise ValueError('Mode draws/gates differ')
            reference=draws
            if state_digest(engine.model.adapter)==before:raise ValueError('Adapter never changed')
            copy_results.append(dict(seed=seed,mode=mode,updates=2,frozen_parent_and_conditioner=True,adapter_changed=True))
            del engine;gc.collect();torch.cuda.empty_cache()
    csv_write(output/'updates.csv',pd.DataFrame(rows))
    command=[sys.executable,'-m','unittest','scripts.v11_guide_audit.test_prototype','-v']
    checked=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (output/'tests.txt').write_text(checked.stdout+checked.stderr,encoding='utf-8')
    if checked.returncode:raise ValueError('Prototype regressions failed')
    paths=[ROOT/'scripts/v11_guide_audit'/name for name in ('engineering.py','guide_dropout.py','test_prototype.py','forward_probe.py')]
    paths += [ROOT/'scripts/v10_detail/engine.py',ROOT/'reports/v10_detail/preparation_20261117_02/protocol.json',ROOT/p['parent']]
    json_write(output/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(output/'complete.json',dict(complete=True,engineering_only=True,actual_updates=len(rows),copies=copy_results,
        matched_actual_draws=True,all_gate_states_exercised=True,tests_passed=3,local_seconds=time.monotonic()-started,
        full_batch_runtime_validated=False,full_exposure_resume_validated=False,efficacy_established=False,
        efficacy_checkpoint_saved=False,paid_gpu_calls=0,clinical_preservation='unknown'))
    print('DISPOSABLE GATED ENGINEERING COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--run-id',required=True);run(a.parse_args().run_id)
