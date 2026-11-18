"""Audit every applied update and segment checkpoint; never infer exposure from a cursor alone."""
import argparse
import hashlib
import json
import math

import pandas as pd
import torch

from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path
from scripts.v13_guide_dropout.engine import finite_nested
from scripts.v13_guide_dropout.protocol import validate,canonical_hash,copy_key,CHECKPOINT_FORMAT


DRAW_FIELDS=('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states')


def state_digest(state):
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        result.update(key.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def audit_rows(rows,seed,mode,start_step,start_chain,start_counts):
    chain=start_chain;counts=list(start_counts);cursor=start_step
    for row in rows:
        cursor+=1
        if row['global_step']!=cursor or row['training_seed']!=seed or row['mode']!=mode:
            raise ValueError('Missing/duplicate/changed actual update identity')
        if row['engineering'] or row['applied'] is not True or row['fixed_echo_time'] is not True:
            raise ValueError('Unapplied or engineering update in full-exposure log')
        expected_batch=min(32,4434-((cursor-1)%139)*32)
        if row['actual_batch_size']!=expected_batch or any(len(row[k])!=expected_batch for k in
            ('sample_indices','timesteps','condition_dropped','gate_states','patient_ids')):
            raise ValueError('Actual update batch/exposure changed')
        if row['epoch_at_draw']!=(cursor-1)//139:raise ValueError('Recorded epoch disagrees with actual cursor')
        if any(type(v) is not int or not 0<=v<4434 for v in row['sample_indices']):raise ValueError('Source index outside training split')
        if any(type(v) is not int or not 0<=v<1000 for v in row['timesteps']):raise ValueError('Invalid training timestep')
        if any(type(v) is not bool for v in row['condition_dropped']):raise ValueError('Invalid condition-drop record')
        if any(type(v) is not int or not 0<=v<5 for v in row['gate_states']):raise ValueError('Invalid actual gate state')
        for metric in ('total_loss','gradient_norm','noise_mse','coarse_structure_loss','seconds'):
            if not math.isfinite(row[metric]):raise ValueError('Nonfinite actual optimisation record')
        if not 0<=row['amp_retries']<=8:raise ValueError('Precision retry allowance exceeded')
        draw={k:row[k] for k in DRAW_FIELDS};chain=canonical_hash(dict(previous=chain,draw=draw))
        if chain!=row['actual_draw_chain_sha256']:raise ValueError('Actual draw-chain mismatch')
        for state in row['gate_states']:counts[state]+=1
    return cursor,chain,counts


def verify(prepared,training,run_id):
    prepared,training=map(project_path,(prepared,training))
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    final=json.loads((training/'complete.json').read_text())
    if not final['complete'] or final['engineering'] or final['common_epoch']!=20 or final['common_step']!=2780:
        raise ValueError('Final common exposure is incomplete')
    expected={copy_key(seed,mode) for seed in p['reference_protocol']['training_seeds'] for mode in p['modes']}
    if set(final['copies'])!=expected:raise ValueError('Missing/extra training copy')
    out=ROOT/'reports/v13_guide_dropout'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.verify_training. copies.csv: '
        'actual applied exposure/gate counts and matched draw-chain audit; inputs.json: every consumed '
        'log/checkpoint/marker/protocol/parent hash; complete.json: frozen-weight and finite-state '
        'checks distinct from inference/clinical success. Logged streams are compared and hash-chained; '
        'full raw noise arrays and every CUDA gate draw are not independently replayed.\n',encoding='utf-8')
    parent=project_path(p['reference_protocol']['parent'])
    if digest(parent)!=p['reference_protocol']['parent_sha256']:raise ValueError('Parent changed')
    payload=torch.load(parent,map_location='cpu',weights_only=False)
    frozen=dict(base=state_digest(payload['unet_state_dict']),conditioner=state_digest(payload['conditioning_state_dict']))
    del payload
    evidence={parent.relative_to(ROOT).as_posix():digest(parent)};summaries=[];chains={};runtime=None;seen=set()
    train_frame=pd.read_csv(ROOT/p['reference_protocol']['view']/'splits/train.csv')
    if len(train_frame)!=4434:raise ValueError('Training manifest length changed')
    for seed in p['reference_protocol']['training_seeds']:
        for mode in p['modes']:
            key=copy_key(seed,mode);step=0;chain=hashlib.sha256(b'guide_dropout_actual_draw_chain_v1').hexdigest();counts=[0]*5
            markers=sorted((training/'segments').glob(key+'_step*/complete.json'))
            if not markers:raise ValueError('No actual segment evidence')
            for marker in markers:
                record=json.loads(marker.read_text());log=marker.parent/'updates.jsonl'
                rows=[json.loads(line) for line in log.read_text().splitlines()]
                if not rows:raise ValueError('Empty completed segment')
                for row in rows:
                    if row['gate_policy_sha256']!=canonical_hash(p['guide_dropout']):raise ValueError('Gate policy mismatch')
                    if row['patient_ids']!=train_frame.iloc[row['sample_indices']].patient_id.tolist():
                        raise ValueError('Logged source index/patient identity mismatch')
                step,chain,counts=audit_rows(rows,seed,mode,step,chain,counts)
                checkpoint=project_path(record['checkpoint'])
                if digest(checkpoint)!=record['sha256']:raise ValueError('Segment checkpoint changed')
                ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                identity=dict(format=CHECKPOINT_FORMAT,protocol=p,protocol_sha256=canonical_hash(p),
                    parent_sha256=p['reference_protocol']['parent_sha256'],training_seed=seed,mode=mode,
                    engineering=False,step=step,total_steps=2780,batch_size=32,frozen=frozen,
                    actual_draw_chain_sha256=chain,gate_counts=counts)
                if any(ck.get(k)!=v for k,v in identity.items()):raise ValueError('Actual log/checkpoint identity mismatch')
                if record['global_step']!=step or record['gate_counts']!=counts or record['actual_draw_chain_sha256']!=chain:
                    raise ValueError('Segment marker disagrees with actual records')
                if not finite_nested(ck):raise ValueError('Nonfinite saved optimisation state')
                if runtime is not None and runtime!=ck['runtime']:raise ValueError('Mixed training runtime')
                runtime=ck['runtime']
                for path in (marker,log,checkpoint):evidence[path.relative_to(ROOT).as_posix()]=digest(path)
                seen.add(marker.parent.name);del ck
            if step!=2780 or sum(counts)!=4434*20 or any(v==0 for v in counts):raise ValueError('Actual final exposure incomplete')
            if seed in chains and chains[seed]!=chain:raise ValueError('Matched source/noise/drop/gate streams disagree')
            chains[seed]=chain
            if final['copies'][key]['checkpoint']!=record['checkpoint'] or final['copies'][key]['sha256']!=record['sha256']:
                raise ValueError('Final snapshot is not the final actual segment')
            summaries.append(dict(key=key,actual_applied_updates=step,samples=sum(counts),
                                  actual_draw_chain_sha256=chain,gate_counts=json.dumps(counts)))
    all_segments={path.parent.name for path in (training/'segments').glob('*/updates.jsonl')}
    if seen!=all_segments:raise ValueError('Unfinished or unexpected actual segment remains')
    for epoch in (5,10,15,20):
        path=training/f'block_{epoch:04d}/complete.json';block=json.loads(path.read_text())
        if block['common_epoch']!=epoch or block['common_step']!=139*epoch or set(block['copies'])!=expected:
            raise ValueError('Common matched block incomplete')
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    for path in (prepared/'protocol.json',training/'complete.json',project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    csv_write(out/'copies.csv',pd.DataFrame(summaries));json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,actual_applied_updates=16680,copies=6,
        all_segment_checkpoints_audited=True,actual_recorded_streams_matched=True,frozen_parent_verified=True,
        full_raw_noise_arrays_replayed=False,all_cuda_gate_draws_replayed=False,
        clinical_preservation='unknown',inference_verification_pending=True))
    print('ACTUAL DROPOUT TRAINING VERIFIED',out,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','training','run-id'):ap.add_argument('--'+name,required=True)
    verify(**vars(ap.parse_args()))
