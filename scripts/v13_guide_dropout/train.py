"""Six matched full-exposure copies, immutable segments and frequent recoverable snapshots."""
import argparse
import gc
import json
import math
import time

import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import CachedCanonical,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path,verify_inputs
from scripts.v13_guide_dropout.engine import Trainer
from scripts.v13_guide_dropout.protocol import validate,canonical_hash,copy_key


def train(prepared,run_id,device='cuda'):
    prepared=project_path(prepared)
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    authority=json.loads((prepared/'authorisation.json').read_text())
    if authority['working_limit_usd']!=12 or authority['protocol_sha256']!=canonical_hash(p):
        raise ValueError('Bound remaining-credit authority required')
    if not run_id.replace('_','').isalnum():raise ValueError('Safe fresh ID required')
    deterministic_runtime()
    reference=p['reference_protocol'];data=CachedCanonical(CanonicalDataset(reference['view'],'train'))
    if len(data)!=4434 or data.frame.patient_id.nunique()!=72:raise ValueError('Frozen training membership changed')
    output=ROOT/'reports/v13_guide_dropout'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v13_guide_dropout.train. protocol.json: full-exposure scope; '
        'segments/: every actual update/draw, immutable checkpoint markers and cumulative gate counts; '
        'block_*/complete.json: all six common-exposure copies; inputs.json: bound input ledger; '
        'complete.json: all copies at final exposure. Checkpoints are versioned under ax_t1_128_v13_guide_dropout. '
        'A completed individual segment is not a complete paired study.\n',encoding='utf-8')
    json_write(output/'protocol.json',p)
    ckhome=ROOT/'checkpoints/diffusion/ax_t1_128_v13_guide_dropout'/run_id
    ckhome.mkdir(parents=True,exist_ok=False);(output/'segments').mkdir()
    batches=math.ceil(len(data)/32);previous={};started=time.monotonic()
    for epoch in (5,10,15,20):
        until=epoch*batches;copies={};chains={}
        for seed in reference['training_seeds']:
            for mode in p['modes']:
                key=copy_key(seed,mode);engine=Trainer(p,data,mode,seed,device)
                if key in previous:engine.restore(project_path(previous[key]))
                while engine.step<until:
                    end=min(until,((engine.step//100)+1)*100)
                    segment=output/'segments'/f'{key}_step{end:06d}';segment.mkdir()
                    with (segment/'updates.jsonl').open('x',encoding='utf-8') as log:
                        while engine.step<end:
                            row=engine.batch();log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    destination=ckhome/f'{key}_step{end:06d}.pt';engine.save(destination)
                    marker=dict(checkpoint=destination.relative_to(ROOT).as_posix(),sha256=digest(destination),
                        global_step=engine.step,epoch=epoch if end==until else None,training_seed=seed,mode=mode,
                        actual_draw_chain_sha256=engine.draw_chain,gate_counts=engine.gate_counts,
                        engineering=False,protocol_sha256=canonical_hash(p))
                    json_write(segment/'complete.json',marker)
                    print('DROPOUT SEGMENT',key,engine.step,flush=True)
                if seed in chains and chains[seed]!=engine.draw_chain:raise ValueError('Matched actual draws/gates diverged')
                chains[seed]=engine.draw_chain;copies[key]=marker;previous[key]=marker['checkpoint']
                del engine;gc.collect();torch.cuda.empty_cache()
        block=output/f'block_{epoch:04d}';block.mkdir()
        json_write(block/'complete.json',dict(copies=copies,common_epoch=epoch,common_step=until,
            matched_draws_within_training_seed=True,engineering=False,protocol_sha256=canonical_hash(p)))
        print('DROPOUT COMMON EPOCH',epoch,flush=True)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    json_write(output/'inputs.json',json.loads((prepared/'inputs.json').read_text()))
    json_write(output/'complete.json',dict(complete=True,engineering=False,copies=copies,common_epoch=20,
        common_step=until,matched_draws_within_training_seed=True,seconds=time.monotonic()-started,
        clinical_preservation='unknown',efficacy_established=False))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    train(**vars(ap.parse_args()))
