"""Exercise every pilot evaluator route locally at two diffusion steps only."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_adapter import evaluate
from scripts.v8_generator_dev.full_train import load_ema
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path
from scripts.v8_generator_dev.train_smoke import sample_wiring


def run(preflight,run_id):
    preflight=project_path(preflight)
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    checks=json.loads((preflight/'checks.json').read_text())
    checkpoints={key:value['checkpoint'] for key,value in checks.items()}
    first=torch.load(project_path(next(iter(checkpoints.values()))),map_location='cpu',weights_only=False)
    parents={backbone:torch.load(project_path(checkpoints[backbone+'__soft_source_guide']),
        map_location='cpu',weights_only=False)['parent'] for backbone in ('additive','scale_shift')}
    prepared=output/'prepared'
    prepared.mkdir()
    view='data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    json_write(prepared/'protocol.json',dict(settings=first['settings'],view=view,parents=parents,
        engineering=True,anatomy_validated=False))
    source=ROOT/'reports/v8_generator_dev/cloud_resume_fetch_20261116_01/protocol/full_20261116_04_protocol/validation_selection.csv'
    if not source.exists():
        candidates=list((ROOT/'reports/v8_generator_dev/cloud_resume_fetch_20261116_01/protocol').glob('*/validation_selection.csv'))
        source,=candidates
    shutil.copyfile(source,prepared/'validation_selection.csv')
    inputs={project_path(value).relative_to(ROOT).as_posix():digest(project_path(value)) for value in checkpoints.values()}
    for name in ('evaluate_adapter.py','adapter_pilot.py','preflight_adapter_eval.py'):
        path=ROOT/'scripts/v8_generator_dev'/name
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(prepared/'inputs.json',inputs)
    (prepared/'README.md').write_text('Engineering-only evaluator-route fixture; epoch-70 parents and batch-two smoke adapters.\n')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    evaluations={}
    for key,path in checkpoints.items():
        controls=('correct','absent','wrong') if key.endswith('__soft_source_guide') else ('correct',)
        for control in controls:
            evaluation_id=run_id+'_'+key+'__'+control
            evaluate(prepared,path,evaluation_id,control,device,engineering=True)
            evaluations[key+'__'+control]='reports/v8_generator_dev/'+evaluation_id
    dataset=CanonicalDataset(view,'val')
    for backbone,parent in parents.items():
        home=project_path(evaluations[backbone+'__soft_source_guide__absent'])
        c,base,scheduler,_=load_ema(project_path(parent),device)
        for request in pd.read_csv(home/'requests.csv').itertuples():
            item=dataset[int(request.dataset_index)]
            image=item['image'][None].to(device)
            with np.load(home/request.array_file) as actual:
                for control,mfg,fs in [('null',item['manufacturer_label'],item['field_strength_label']),
                    ('target',request.target_manufacturer,request.target_field_strength)]:
                    expected=sample_wiring(c,base,scheduler,image,torch.tensor([mfg],device=device),
                        torch.tensor([fs],device=device),torch.tensor([item['echo_time_norm']],device=device),
                        int(request.actual_sampling_seed),2,3.)[0,0]
                    np.testing.assert_array_equal(actual[control],expected)
        del c,base,scheduler
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,evaluator_routes=8,cases=32,
        absent_guide_parent_images_exact=True,engineering=True,efficacy_established=False,
        anatomy_validated=False,evaluations=evaluations))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.preflight_adapter_eval. prepared/ is an engineering '
        'fixture only. Eight separately named evaluation reports exercise guide/zero/absent/wrong '
        'routes on two validation patients at t=2, not the scientific fixed points. '
        'complete.json also records exact absent-guide equality to separately sampled frozen '
        'parents. inputs.json binds all engineering adapters/code. No efficacy claim.\n',encoding='utf-8')
    print('ADAPTER EVALUATOR PREFLIGHT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--preflight',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    run(a.preflight,a.run_id)
