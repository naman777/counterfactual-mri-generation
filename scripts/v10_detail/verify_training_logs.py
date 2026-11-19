"""Independent journal coverage, finite-update and matched-draw checks; no training."""
import argparse
import json
import math
import re

import pandas as pd

from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path
from scripts.v10_detail.protocol import copy_key,canonical_hash


def check_records(records,start,stop,frame,seed,mode,batch_size=32):
    batches=math.ceil(len(frame)/batch_size)
    if len(records)!=stop-start:raise ValueError('Missing or extra applied-update row')
    draws=[]
    for expected,row in zip(range(start+1,stop+1),records):
        if row['global_step']!=expected or row['training_seed']!=seed or row['mode']!=mode:
            raise ValueError('Journal cursor/copy identity mismatch')
        if not row['applied'] or row['engineering']:raise ValueError('Skipped or engineering update in efficacy log')
        if not row['fixed_echo_time']:raise ValueError('EchoTime changed')
        if not 0<=row['amp_retries']<=8:raise ValueError('Precision retry rule changed')
        for metric in ('total_loss','noise_mse','coarse_structure_loss','gradient_norm'):
            if not math.isfinite(row[metric]):raise ValueError('Nonfinite update metric')
        if not math.isclose(row['total_loss'],row['noise_mse']+.1*row['coarse_structure_loss'],rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('Reported objective components disagree')
        epoch,within=divmod(expected-1,batches)
        size=min(batch_size,len(frame)-within*batch_size)
        if row['epoch_at_draw']!=epoch or row['actual_batch_size']!=size:raise ValueError('Exposure/batch-size drift')
        if any(len(row[name])!=size for name in ('sample_indices','patient_ids','timesteps','condition_dropped')):
            raise ValueError('Draw-vector size mismatch')
        if any(type(i) is not int or not 0<=i<len(frame) for i in row['sample_indices']):raise ValueError('Source index invalid')
        if frame.iloc[row['sample_indices']].patient_id.tolist()!=row['patient_ids']:raise ValueError('Logged source patient mismatch')
        if any(type(t) is not int or not 0<=t<1000 for t in row['timesteps']):raise ValueError('Noise timestep invalid')
        if not re.fullmatch('[a-f0-9]{64}',row['noise_sha256']):raise ValueError('Missing noise identity')
        draws.append({name:row[name] for name in ('global_step','sample_indices','patient_ids','timesteps','condition_dropped','noise_sha256')})
    return draws


def verify(prepared,results,run_id):
    prepared,results=map(project_path,(prepared,results))
    protocol=json.loads((prepared/'protocol.json').read_text())
    receipt=json.loads((results/'complete.json').read_text())
    if not receipt['workflow_success']:raise ValueError('Incomplete workflow is not completed training')
    training=project_path(receipt['training'])
    frame=pd.read_csv(project_path(protocol['view'])/'splits/train.csv')
    batches=math.ceil(len(frame)/32);start=0;count=0
    manifest=project_path(protocol['view'])/'splits/train.csv'
    evidence={manifest.relative_to(ROOT).as_posix():digest(manifest),
              (prepared/'protocol.json').relative_to(ROOT).as_posix():digest(prepared/'protocol.json')}
    blocks=sorted(training.glob('block_*/complete.json'))
    if len(blocks)!=4:raise ValueError('All four common training blocks required')
    for marker in blocks:
        block=json.loads(marker.read_text());stop=block['common_step']
        if stop-start!=5*batches or len(block['copies'])!=9 or block['engineering']:
            raise ValueError('Common block exposure/copies mismatch')
        expected={copy_key(seed,mode) for seed in protocol['training_seeds'] for mode in protocol['modes']}
        if set(block['copies'])!=expected or block['protocol_sha256']!=canonical_hash(protocol):
            raise ValueError('Common block copy/protocol identity mismatch')
        matched={}
        for key,copy in block['copies'].items():
            if key!=copy_key(copy['training_seed'],copy['mode']):raise ValueError('Copy label mismatch')
            path=marker.parent/(key+'.jsonl');records=[json.loads(s) for s in path.read_text().splitlines()]
            draws=check_records(records,start,stop,frame,copy['training_seed'],copy['mode'])
            seed=copy['training_seed']
            if seed in matched and matched[seed]!=draws:raise ValueError('Modes received different actual logged draws')
            matched[seed]=draws;count+=len(records)
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        start=stop;evidence[marker.relative_to(ROOT).as_posix()]=digest(marker)
    if start!=2780 or count!=25020:raise ValueError('Final replicated exposure mismatch')
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    evidence[project_path(__file__).relative_to(ROOT).as_posix()]=digest(__file__)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,actual_logged_updates_checked=count,
        matched_draws_within_training_seed=True,source_membership_checked=True,finite_applied_updates=True,
        final_steps_per_copy=start,noise_streams_matched_by_recorded_hash=True,
        noise_arrays_independently_regenerated=False,clinical_preservation='unknown'))
    (output/'README.md').write_text('Producer: scripts.v10_detail.verify_training_logs. inputs.json binds all 36 journals/common markers '
        'and checker source; complete.json records every actual row/cursor/batch/source-patient/finite-update/fixed-EchoTime '
        'check and exact within-seed mode draw agreement. Recorded noise hashes are matched; raw noise tensors are not '
        'independently regenerated. No model-quality or clinical claim.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','run-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();verify(a.prepared,a.results,a.run_id)
