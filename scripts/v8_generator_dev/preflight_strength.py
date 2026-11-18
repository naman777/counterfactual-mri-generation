"""Local two-step amplitude checks; never starts paid compute or changes weights."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch

from scripts.v6_structure.model import BoundStructureModel, StructureConditionedUNet
from scripts.v8_generator_dev.adapter_pilot import guide_for
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_strength import evaluate
from scripts.v8_generator_dev.full_train import deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path
from scripts.v8_generator_dev.train_smoke import sample_wiring


def run(fetched,run_id):
    deterministic_runtime()
    fetched=project_path(fetched)
    receipt=json.loads((fetched/'complete.json').read_text())
    source_prepared=project_path(receipt['prepared'])
    protocol=json.loads((source_prepared/'protocol.json').read_text())
    trained=json.loads((project_path(receipt['training'])/'complete.json').read_text())
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.preflight_strength. prepared/ is an engineering-only '
        'fixture. endpoint_checks.csv checks alpha 0 against the direct frozen parent and alpha 1 '
        'against the established full-guide sampler, plus finite fractional strengths on actual '
        'trained checkpoints. Two separate midpoint evaluator reports exercise saved metadata, '
        'source/label/EchoTime/judge paths at t=2 on two preselected validation patients. '
        'inputs.json hashes sources/weights/code; complete.json is software readiness only.\n',encoding='utf-8')
    prepared=output/'prepared'
    prepared.mkdir()
    protocol.pop('approved_additional_usd',None)
    protocol.update(stage='engineering_only_strength_check',new_cloud_approval=False)
    json_write(prepared/'protocol.json',protocol)
    shutil.copyfile(source_prepared/'validation_selection.csv',prepared/'validation_selection.csv')
    inputs=dict(json.loads((source_prepared/'inputs.json').read_text()))
    for name in ('evaluate_strength.py','preflight_strength.py'):
        path=ROOT/'scripts/v8_generator_dev'/name
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(prepared/'inputs.json',inputs)
    (prepared/'README.md').write_text('Engineering-only source-strength fixture; no paid approval inferred from its parent pilot.\n')
    val=CanonicalDataset(protocol['view'],'val')
    index=int(pd.read_csv(prepared/'validation_selection.csv').dataset_index.iloc[0])
    item=val[index]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=item['image'][None].to(device)
    sampling_seed=protocol['parent_config']['evaluation']['selection_generation_seed']+index
    rows=[]
    evaluations={}
    for backbone in ('additive','scale_shift'):
        checkpoint=project_path(trained['copies'][backbone+'__soft_source_guide']['checkpoint'])
        before_hash=digest(checkpoint)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        parent=project_path(protocol['parents'][backbone])
        parent_before=digest(parent)
        c,base,scheduler,_=load_ema(parent,device)
        model=StructureConditionedUNet(base,ck['settings']['adapter_width']).to(device).eval().requires_grad_(False)
        model.adapter.load_state_dict(ck['adapter_state_dict'])
        guide=guide_for((source+1)/2,'soft_source_guide',0,augment=False)
        m=torch.tensor([1-item['manufacturer_label']],device=device)
        f=torch.tensor([item['field_strength_label']],device=device)
        echo=torch.tensor([item['echo_time_norm']],device=device)
        def sample(unet):
            return sample_wiring(c,unet,scheduler,source,m,f,echo,sampling_seed,2,3.)
        baseline=sample(base)
        full=sample(BoundStructureModel(model,guide,torch.ones(1,device=device)))
        for alpha in (0.,.25,.5,.75,1.):
            result=sample(BoundStructureModel(model,guide,torch.full((1,),alpha,device=device)))
            if not np.isfinite(result).all():
                raise AssertionError('Nonfinite source-strength inference')
            if alpha==0.:
                np.testing.assert_array_equal(result,baseline)
            if alpha==1.:
                np.testing.assert_array_equal(result,full)
            rows.append(dict(backbone=backbone,alpha=alpha,finite=True,
                parent_endpoint_exact=alpha!=0. or np.array_equal(result,baseline),
                full_guide_endpoint_exact=alpha!=1. or np.array_equal(result,full),
                mean_absolute_change_from_parent=float(np.abs(result-baseline).mean())))
        if digest(checkpoint)!=before_hash or digest(parent)!=parent_before:
            raise AssertionError('Frozen inference inputs changed')
        inputs[checkpoint.relative_to(ROOT).as_posix()]=before_hash
        evaluation_id=run_id+'_midpoint_'+backbone
        evaluate(prepared,checkpoint,evaluation_id,.5,device,engineering=True)
        home=ROOT/'reports/v8_generator_dev'/evaluation_id
        requests=pd.read_csv(home/'requests.csv')
        if not requests.guide_strength.eq(.5).all() or not requests.source_echo_time_norm.eq(requests.target_echo_time_norm).all():
            raise AssertionError('Midpoint evaluator metadata drift')
        evaluations[backbone]=str(home.relative_to(ROOT))
        print('SOURCE STRENGTH ENGINEERING',backbone,flush=True)
    csv_write(output/'endpoint_checks.csv',pd.DataFrame(rows))
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,endpoint_checks_passed=True,
        strengths=[0.,.25,.5,.75,1.],midpoint_evaluations=evaluations,engineering=True,
        full_operating_points_evaluated=False,cloud_launched=False,anatomy_validated=False))
    print('SOURCE STRENGTH PREFLIGHT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    run(a.fetched,a.run_id)
