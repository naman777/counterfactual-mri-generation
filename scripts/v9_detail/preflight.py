"""Actual final-parent bounded CUDA checks and evaluator integration, no efficacy."""
import argparse
import gc
import json
import time
import unittest
from contextlib import redirect_stdout,redirect_stderr

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import multiscale_guide
from scripts.v8_generator_dev.evaluate_full import source_native,predict_judge
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,project_path
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v9_detail.common import start,finish,VIEW,VAL_SELECTION
from scripts.v9_detail.engine import EngineeringTrainer,MODES
from scripts.v9_detail.evaluation import evaluate_case,GUIDE_CONTROLS,MASK_CONTROLS


def run(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.preflight','Actual final-parent local integration')
    protocol_path=ROOT/'reports/v8_generator_dev/adapter_preparation_20261117_02/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    inputs[protocol_path.relative_to(ROOT).as_posix()]=digest(protocol_path)
    dependencies=['adapter_pilot.py','full_train.py','model.py','precision_step.py','train_smoke.py',
                  'evaluate_full.py','full_protocol.py','test_adapter_pilot.py','test_full.py','test_pipeline.py']
    for name in dependencies:
        p=ROOT/'scripts/v8_generator_dev'/name;inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    for p in (ROOT/'src/models').glob('*.py'):inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    for p in (ROOT/'scripts/v2_quickwins/judge_model.py',ROOT/'scripts/v7_geometry/support.py'):
        inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    json_write(output/'protocol.json',dict(parent_protocol=str(protocol_path.relative_to(ROOT)),
        parent_epoch=130,modes=list(MODES),seed=583201,engineering_batch_size=2,updates_per_copy=2,
        full_batch_forward=32,full_batch_training_tested=False,
        full_batch_limit='Local 6GB GPU; batch32 forward-only probe is distinct from batch2 optimisation',
        evaluator_timestep=2,evaluator_guidance=3.,guide_controls=list(GUIDE_CONTROLS),mask_controls=list(MASK_CONTROLS),
        scope='Discarded local engineering updates; no efficacy or cloud run; no older budget reused'))
    suite=unittest.defaultTestLoader.loadTestsFromName('scripts.v9_detail.test_engine')
    with (output/'tests.txt').open('x',encoding='utf-8') as handle,redirect_stdout(handle),redirect_stderr(handle):
        tested=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    if not tested.wasSuccessful():raise AssertionError('Engine/evaluator unit checks failed')
    deterministic_runtime()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train,val=CanonicalDataset(VIEW,'train'),CanonicalDataset(VIEW,'val')
    indices=pd.read_csv(VAL_SELECTION).dataset_index.astype(int).tolist()[:2]
    if val.frame.iloc[indices].patient_id.duplicated().any():raise ValueError('Wrong guides require different patients')
    ckdir=ROOT/'checkpoints/diffusion/ax_t1_128_v9_detail'/run_id;ckdir.mkdir(parents=True,exist_ok=False)
    for backbone,parent in protocol['parents'].items():
        p=project_path(parent)
        if digest(p)!=protocol['parent_hashes'][backbone]:raise ValueError('Final parent hash changed')
        inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    results,records,grade_rows=[],[],[]
    matched_draws=None
    for backbone,parent in protocol['parents'].items():
        for mode in MODES:
            engine=EngineeringTrainer(parent,train,mode,device=device)
            # Hash every actual source consumed by optimiser draws after reading records.
            key=backbone+'__'+mode
            before=state_digest(engine.model.adapter)
            first=train[0]
            x=first['image'][None].to(device)
            guide=multiscale_guide((x+1)/2,mode)
            ts=torch.ones(1,device=device,dtype=torch.long)
            labels=torch.zeros(1,device=device,dtype=torch.long)
            with torch.no_grad():
                condition=engine.c(labels,labels,torch.zeros(1,device=device))
                zero_equal=torch.equal(engine.base(x,ts,condition),engine.model(x,ts,condition,guide))
            if not zero_equal:raise AssertionError('Zero-init differs from actual parent')
            forward32=dict(attempted=False,passed=False,reason='Checked once per backbone on coarse/fine copy only')
            if mode=='coarse_fine':
                forward32['attempted']=True
                try:
                    if device=='cuda':torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                        full_x=x.repeat(32,1,1,1)
                        full_guide=guide.repeat(32,1,1,1)
                        result=engine.model(full_x,ts.repeat(32),condition.repeat(32,1),full_guide)
                        forward32.update(passed=bool(torch.isfinite(result).all()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
                        del result,full_x,full_guide
                    if not forward32['passed']:raise FloatingPointError('Nonfinite batch32 forward')
                except torch.cuda.OutOfMemoryError as error:
                    forward32.update(passed=False,reason='Local batch32 forward OOM retained; no retry',error=str(error))
                    gc.collect();torch.cuda.empty_cache()
            started=time.perf_counter()
            draws=[]
            for _ in range(2):
                record=engine.batch();records.append(record)
                draws.append({k:record[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256')})
                for index in record['sample_indices']:
                    row=train.frame.iloc[index];p=project_path(row.source_image_path)
                    inputs[p.relative_to(ROOT).as_posix()]=row.source_array_sha256
            if matched_draws is None:matched_draws=draws
            elif draws!=matched_draws:raise AssertionError('Actual mode/backbone optimiser draws differ')
            updated=state_digest(engine.model.adapter)!=before
            if not updated:raise AssertionError('No actual adapter update')
            snapshot=ckdir/(key+'.pt');engine.save(snapshot)
            engine.restore(snapshot)
            # All guide routes and each mask route are tested on a fixed source.
            item=val[indices[0]];source=val.image_01(indices[0]);wrong=val.image_01(indices[1])
            for index in indices:
                p=project_path(val.frame.iloc[index].source_image_path)
                inputs[p.relative_to(ROOT).as_posix()]=digest(p)
            arrays={}
            for axis in ('manufacturer','field_strength'):
                seed=583241+indices[0]
                baseline,meta=evaluate_case(engine,item,source,axis,seed)
                for control in GUIDE_CONTROLS:
                    images,info=evaluate_case(engine,item,source,axis,seed,control,wrong_source=wrong)
                    if not np.isfinite(images['target']).all():raise AssertionError('Nonfinite guide route')
                    arrays[axis+'__guide_'+control]=images['target']
                    if control=='absent':
                        m,f=item['manufacturer_label'],item['field_strength_label']
                        tm,tf=(1-m,f) if axis=='manufacturer' else (m,1-f)
                        actual=sample_wiring(engine.c,engine.base,engine.scheduler,item['image'][None].to(device),
                            torch.tensor([tm],device=device),torch.tensor([tf],device=device),
                            torch.tensor([item['echo_time_norm']],device=device),seed,2,3.)[0,0]
                        np.testing.assert_array_equal(images['target'],actual)
                for control in MASK_CONTROLS:
                    images,info=evaluate_case(engine,item,source,axis,seed,mask_control=control,wrong_source=wrong)
                    if not np.isfinite(images['target']).all():raise AssertionError('Nonfinite mask route')
                    arrays[axis+'__mask_'+control]=images['target']
                    if control=='full':np.testing.assert_array_equal(images['target'],baseline['target'])
                    if control=='empty':np.testing.assert_array_equal(images['target'],((item['image']+1)/2)[0].numpy())
                # Frozen named judges see the source-native frame; no gradients, rates or efficacy assertions.
                native=source_native(np.stack([source,baseline['null'],baseline['target']]),val.plans[indices[0]])
                for j,path in enumerate(protocol['parent_config']['evaluation']['judge_paths']):
                    p=project_path(path);inputs[p.relative_to(ROOT).as_posix()]=digest(p)
                    judge=load_judge(p,device).eval().requires_grad_(False)
                    probabilities=predict_judge(judge,native,device)
                    for head,values in probabilities.items():
                        if not np.isfinite(values).all():raise AssertionError('Nonfinite engineering judge route')
                        grade_rows.append(dict(backbone=backbone,mode=mode,axis=axis,
                            judge=f'heldout_orientation_robust_seed{j}',head=head,
                            probabilities_json=json.dumps(values.tolist()),engineering=True))
                    del judge
            with (output/(key+'_arrays.npz')).open('xb') as handle:np.savez_compressed(handle,source=source,**arrays)
            inputs[snapshot.relative_to(ROOT).as_posix()]=digest(snapshot)
            results.append(dict(backbone=backbone,mode=mode,updates=engine.step,batch_size=2,
                zero_init_equal=zero_equal,adapter_updated=updated,frozen_parent=True,
                absent_equals_parent=True,full_mask_equals_unmasked=True,empty_mask_equals_source=True,
                forward32=forward32,elapsed_seconds=time.perf_counter()-started,
                checkpoint=str(snapshot.relative_to(ROOT)),checkpoint_sha256=digest(snapshot)))
            del engine;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            print('ACTUAL PARENT ENGINEERING PASS',key,flush=True)
    csv_write(output/'steps.csv',pd.DataFrame(records))
    csv_write(output/'judge_routes.csv',pd.DataFrame(grade_rows))
    json_write(output/'checks.json',dict(device=device,copies=results,matched_actual_draws=True,
        unit_tests=tested.testsRun,full_batch_training_tested=False))
    (output/'OUTPUTS.md').write_text('Producer: scripts.v9_detail.preflight. tests.txt: engine resume/rejection/control tests; '
        'steps.csv: actual discarded final-parent adapter updates and matched draws; checks.json: all identities/endpoints, '
        'local batch32 forward-only results and timing; *_arrays.npz: every engineering guide/mask target route; '
        'judge_routes.csv: finite source-native frozen named-judge probabilities, not efficacy hit rates. '
        'Engineering-only checkpoint files live at the new paths in checks.json.\n',encoding='utf-8')
    finish(output,inputs,copies=len(results),updates=len(records),device=device,
        full_batch_training_tested=False,cloud_ready=False,engineering_weights_only=True)
    print('ACTUAL PARENT PREFLIGHT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    run(p.parse_args().run_id)
