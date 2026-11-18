"""Check every new evaluator route on disposable real-parent engineering weights."""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v13_guide_dropout.engine import Trainer
from scripts.v13_guide_dropout.evaluate import evaluate
from scripts.v13_guide_dropout.protocol import evaluation_protocol,validate


def run(prepared,run_id):
    prepared=project_path(prepared)
    p=json.loads((prepared/'protocol.json').read_text());validate(p)
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    out=ROOT/'reports/v13_guide_dropout'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.local_preflight. checks.json: '
        'all fourteen two-source/two-step engineering evaluator routes, checkpoints and actual update draws; '
        'inputs.json: evidence hashes; complete.json: engineering-only scope. Referenced reports retain '
        'every image/judge grade. No full-exposure efficacy or paid launch.\n',encoding='utf-8')
    ckhome=ROOT/'checkpoints/diffusion/ax_t1_128_v13_guide_dropout'/run_id;ckhome.mkdir(parents=True,exist_ok=False)
    deterministic_runtime();data=CanonicalDataset(p['reference_protocol']['view'],'train')
    settings=evaluation_protocol(p);checks={};reference=None;absent=[]
    evidence={(prepared/'protocol.json').relative_to(ROOT).as_posix():digest(prepared/'protocol.json')}
    for mode in p['modes']:
        engine=Trainer(p,data,mode,583301,'cuda',engineering=True,batch_override=5)
        rows=[engine.batch(),engine.batch()];engine.frozen_check()
        draws=[{k:r[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states')} for r in rows]
        if reference is not None and draws!=reference:raise ValueError('Actual mode draws differ')
        reference=draws
        snapshot=ckhome/(mode+'.pt');engine.save(snapshot);evidence[snapshot.relative_to(ROOT).as_posix()]=digest(snapshot)
        del engine;gc.collect();torch.cuda.empty_cache()
        routes={}
        for axis,arms in settings['controls'].items():
            for control in arms[mode]:
                identity=run_id+'_'+mode+'_'+axis+'_'+control
                home=evaluate(prepared,snapshot,identity,axis,control,'cuda',engineering=True)
                routes[axis+'__'+control]=home.relative_to(ROOT).as_posix()
                for filename in ('protocol.json','requests.csv','per_image.csv','complete.json'):
                    evidence[(home/filename).relative_to(ROOT).as_posix()]=digest(home/filename)
        home=ROOT/routes['manufacturer__absent'];images=[]
        for r in pd.read_csv(home/'requests.csv').itertuples():
            with np.load(home/r.array_file,allow_pickle=False) as z:images.append((z['null'].copy(),z['target'].copy()))
        absent.append(images);checks[mode]=dict(updates=rows,checkpoint=snapshot.relative_to(ROOT).as_posix(),routes=routes)
    for a,b in zip(absent[0],absent[1]):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    verify_inputs(evidence);json_write(out/'checks.json',checks);json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,engineering_only=True,evaluator_routes=14,cases=28,
        named_grades=252,absent_images_equal_across_modes=True,pinned_runtime_batch32_pending=True,
        paid_cloud_started=False,efficacy_started=False))
    print('ALL DROPOUT EVALUATOR ROUTES PASS',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--prepared',required=True);ap.add_argument('--run-id',required=True)
    run(**vars(ap.parse_args()))
