"""Sequential replicated training with common-block publication and strict resumes."""
import argparse
import gc
import json
import math
import time

import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import CachedCanonical,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v10_detail.protocol import validate,canonical_hash,copy_key
from scripts.v10_detail.engine import Trainer


def train(prepared,run_id,engineering_steps=None,device='cuda'):
    prepared=project_path(prepared)
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    engineering=engineering_steps is not None
    if engineering and engineering_steps not in (1,2):raise ValueError('Engineering is bounded at one/two batches')
    if not engineering:
        approval=json.loads((prepared/'authorisation.json').read_text())
        if approval['approved_additional_usd']!=12 or approval['protocol_sha256']!=canonical_hash(p):
            raise ValueError('Explicit separate allowance/protocol authorisation required')
    if not run_id.replace('_','').isalnum():raise ValueError('Safe new ID required')
    deterministic_runtime()
    dataset=CachedCanonical(CanonicalDataset(p['view'],'train'))
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    checkpoints=ROOT/'checkpoints/diffusion/ax_t1_128_v10_detail'/run_id
    checkpoints.mkdir(parents=True,exist_ok=False)
    json_write(output/'protocol.json',dict(scientific_protocol=p,engineering=engineering))
    (output/'README.md').write_text('Producer: scripts.v10_detail.train. protocol.json fixes scope; '
        'block_*/ contains every actual draw/update log, copy audits and a common-block complete marker. '
        'inputs.json binds dependencies; complete.json requires all nine matched copies. '
        'Checkpoints are fresh files in the named v10_detail checkpoint folder. Failures retain incomplete evidence.\n',encoding='utf-8')
    batches=math.ceil(len(dataset)/p['training']['batch_size'])
    endpoints=[engineering_steps] if engineering else [batches*e for e in range(5,21,5)]
    previous={};last=0
    for number,until in enumerate(endpoints,1):
        block=output/f'block_{number:04d}';block.mkdir()
        copies={};draws_by_seed={}
        for seed in p['training_seeds']:
            for mode in p['modes']:
                key=copy_key(seed,mode)
                engine=Trainer(p,dataset,mode,seed,device,engineering)
                if key in previous:engine.restore(project_path(previous[key]))
                records=[]
                with (block/(key+'.jsonl')).open('x',encoding='utf-8') as log:
                    while engine.step<until:
                        record=engine.batch();records.append({k:record[k] for k in
                            ('sample_indices','timesteps','condition_dropped','noise_sha256')})
                        log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                        if engine.step%100==0:print('REPLICATED UPDATE',key,engine.step,flush=True)
                if seed in draws_by_seed and records!=draws_by_seed[seed]:raise AssertionError('Unmatched actual training draws')
                draws_by_seed[seed]=records
                destination=checkpoints/f'{key}_step{until:06d}.pt';engine.save(destination)
                result=dict(checkpoint=destination.relative_to(ROOT).as_posix(),sha256=digest(destination),
                    global_step=engine.step,epoch=engine.step//batches,training_seed=seed,mode=mode,
                    frozen_parent=True,engineering=engineering,updates=until-last)
                copies[key]=result;previous[key]=result['checkpoint']
                json_write(block/(key+'.json'),result)
                del engine;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        json_write(block/'complete.json',dict(copies=copies,common_step=until,common_epoch=until//batches,
            matched_draws_within_training_seed=True,engineering=engineering,protocol_sha256=canonical_hash(p)))
        last=until;print('REPLICATED COMMON BLOCK',until,flush=True)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    json_write(output/'inputs.json',dict(json.loads((prepared/'inputs.json').read_text()),
        **{(prepared/'protocol.json').relative_to(ROOT).as_posix():digest(prepared/'protocol.json')}))
    json_write(output/'complete.json',dict(complete=True,engineering=engineering,copies=copies,
        common_step=last,common_epoch=last//batches,training_seeds=p['training_seeds'],
        matched_draws_within_training_seed=True,clinical_preservation='unknown',efficacy_established=False))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--engineering-steps',type=int);ap.add_argument('--device',default='cuda')
    a=ap.parse_args();train(a.prepared,a.run_id,a.engineering_steps,a.device)
