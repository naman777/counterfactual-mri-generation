"""Discard two full-batch updates per copy; no smoke weights enter full training."""
import argparse
import gc
import json

import numpy as np
import torch

from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import multiscale_guide
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,json_write,project_path,verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v13_guide_dropout.engine import Trainer
from scripts.v13_guide_dropout.protocol import copy_key,validate


def smoke(prepared,run_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text());validate(p)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    deterministic_runtime();data=CanonicalDataset(p['reference_protocol']['view'],'train')
    output=ROOT/'reports/v13_guide_dropout'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.runtime_smoke. checks.json: '
        'six discarded full-batch backward copies, actual gates/draws, frozen weights and absent-parent equality; '
        'complete.json: target-runtime gate only. No smoke weights are reused.\n',encoding='utf-8')
    checks=[];matched={};states=set()
    for seed in p['reference_protocol']['training_seeds']:
        for mode in p['modes']:
            torch.cuda.reset_peak_memory_stats()
            engine=Trainer(p,data,mode,seed,'cuda',engineering=True)
            before=state_digest(engine.model.adapter);rows=[engine.batch(),engine.batch()];engine.frozen_check()
            if before==state_digest(engine.model.adapter):raise ValueError('Adapter did not change')
            draws=[{k:r[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states')} for r in rows]
            if seed in matched and matched[seed]!=draws:raise ValueError('Matched full-batch draws changed')
            matched[seed]=draws
            for row in rows:
                if row['actual_batch_size']!=32 or not row['applied']:raise ValueError('Full batch did not apply')
                states.update(row['gate_states'])
            item=data[0];image=item['image'][None].cuda();guide=multiscale_guide((image+1)/2,mode)
            args=(engine.c,engine.base,engine.scheduler,image,torch.tensor([0],device='cuda'),
                  torch.tensor([0],device='cuda'),torch.tensor([item['echo_time_norm']],device='cuda'),583390,2,3.)
            parent=sample_wiring(*args)
            absent=sample_wiring(engine.c,BoundStructureModel(engine.model.inner,guide,torch.zeros(1,device='cuda')),*args[2:])
            np.testing.assert_array_equal(parent,absent)
            checks.append(dict(key=copy_key(seed,mode),updates=rows,runtime=engine.runtime,
                frozen_parent=True,absent_parent_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del engine;gc.collect();torch.cuda.empty_cache()
            print('DROPOUT FULL BATCH PASS',copy_key(seed,mode),flush=True)
    if states!=set(range(5)):raise ValueError('All five random gate states not exercised')
    json_write(output/'checks.json',checks)
    json_write(output/'complete.json',dict(complete=True,copies=6,actual_discarded_updates=12,
        full_batch_backward=True,all_states_exercised=True,weights_reused=False,efficacy_started=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    smoke(**vars(ap.parse_args()))
