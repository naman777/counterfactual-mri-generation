"""Discarded full-batch backward/optimiser checks before any efficacy exposure."""
import argparse
import gc
import json
import time

import numpy as np
import torch

from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import multiscale_guide
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path,verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v10_detail.engine import Trainer
from scripts.v10_detail.protocol import copy_key


def smoke(prepared,run_id,local_single=False):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    deterministic_runtime();data=CanonicalDataset(p['view'],'train')
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.runtime_smoke. protocol.json: exact discarded '
        'full-batch check scope; checks.json: actual draws/loss/AMP/memory/frozen-parent/endpoints; inputs.json: '
        'bound evidence; failed.json retains a failed local/device check; complete.json only after success. '
        'No efficacy weights are saved or reused.\n',encoding='utf-8')
    json_write(output/'protocol.json',dict(scientific_protocol=p,local_single=local_single,
        batch_size=32,discarded_updates=True,required_device='cuda'))
    checks=[];matched={}
    try:
        seeds=p['training_seeds'][:1] if local_single else p['training_seeds']
        modes=['coarse_fine'] if local_single else p['modes']
        for seed in seeds:
            for mode in modes:
                torch.cuda.reset_peak_memory_stats()
                engine=Trainer(p,data,mode,seed,'cuda',engineering=True)
                initial=state_digest(engine.model.adapter);start=time.monotonic()
                result=engine.batch();engine.frozen_check()
                if state_digest(engine.model.adapter)==initial:raise AssertionError('Adapter did not update')
                draw={k:result[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256')}
                if seed in matched and matched[seed]!=draw:raise AssertionError('Full-batch mode draws differ')
                matched[seed]=draw
                item=data[0];image=item['image'][None].cuda();guide=multiscale_guide((image+1)/2,mode)
                args=(engine.c,engine.base,engine.scheduler,image,torch.tensor([0],device='cuda'),
                      torch.tensor([0],device='cuda'),torch.tensor([item['echo_time_norm']],device='cuda'),583390,2,3.)
                parent=sample_wiring(*args)
                absent=sample_wiring(engine.c,BoundStructureModel(engine.model,guide,torch.zeros(1,device='cuda')),
                                     *args[2:])
                np.testing.assert_array_equal(parent,absent)
                checks.append(dict(key=copy_key(seed,mode),update=result,runtime=engine.runtime,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
                    frozen_parent=True,adapter_updated=True,absent_parent_exact=True,actual_batch_size=result['actual_batch_size']))
                del engine;gc.collect();torch.cuda.empty_cache()
                print('FULL BATCH BACKWARD PASS',copy_key(seed,mode),flush=True)
    except Exception as error:
        json_write(output/'failed.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            completed_copies=checks,local_single=local_single,efficacy_started=False,automatic_retry=False))
        raise
    json_write(output/'checks.json',dict(copies=checks,discarded=True,full_batch_backward=True,local_single=local_single))
    inputs=dict(json.loads((prepared/'inputs.json').read_text()))
    inputs[(prepared/'protocol.json').relative_to(ROOT).as_posix()]=digest(prepared/'protocol.json')
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,full_batch_backward=True,local_single=local_single,
        required_cloud_runtime_smoke=not local_single,copies=len(checks),efficacy_started=False,weights_reused=False))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--local-single',action='store_true');a=ap.parse_args();smoke(a.prepared,a.run_id,a.local_single)
