"""Local train/validation leakage and sensitivity audit of untrained finer guides."""
import argparse
import json
import platform
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import multiscale_guide
from scripts.v8_generator_dev.guide_audit import balanced_score, confidence
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.soft_structure import soft_structure

BASE=ROOT/'reports/v8_generator_dev'


def audit(run_id):
    if not run_id.replace('_','').isalnum():
        raise ValueError('Safe fresh ID required')
    torch.set_num_threads(2)
    output=BASE/run_id
    output.mkdir(exist_ok=False)
    selection=BASE/'canonical_guide_audit_20261116_01/patient_selection.csv'
    frame=pd.read_csv(selection)
    view='data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    datasets={s:CanonicalDataset(view,s) for s in ('train','val')}
    functions=dict(historical_coarse=soft_structure,
        matched_coarse=lambda x:multiscale_guide(x,'coarse_control'),
        coarse_fine=multiscale_guide)
    json_write(output/'protocol.json',dict(view=view,patient_selection=str(selection.relative_to(ROOT)),
        representations=list(functions),pooled_size=[8,8], feature_budget='Historical 192; matched coarse/fine both 384',
        probe='Fixed class-balanced ridge alpha10; train-only standardisation; no tuning',
        perturbations=['gamma .7','gamma 1.3','affine .8x+.04','translate8px'],
        role='Exploratory development weak-probe/sensitivity audit; no efficacy, invariance or anatomy claim',
        cloud_ready=False,clinical_preservation='unknown'))
    inputs={selection.relative_to(ROOT).as_posix():digest(selection)}
    sources=['scripts/v8_generator_dev/'+name for name in (
        'audit_detail_candidates.py','detail_candidates.py','soft_structure.py','data.py','io.py',
        'guide_audit.py','test_detail_candidates.py','test_detail_diagnosis.py','detail_diagnosis.py',
        'model.py','train_smoke.py','test_pipeline.py')]
    sources += ['scripts/v6_structure/model.py','scripts/v5_anatomy/metrics.py',
                'scripts/v8_generator_dev/preservation_screen.py','scripts/v7_geometry/orientation.py',
                'scripts/v5_anatomy/anchored_sampler.py']
    # Archive sources used for the new engineering results; historical modules remain unchanged.
    import shutil
    for name in sources:
        path=ROOT/name
        inputs[name]=digest(path)
        dest=output/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    for split in ('train','val'):
        for p in (project_path(view)/'splits'/f'{split}.csv',ROOT/'data/processed/ax_t1_128/splits'/f'{split}.csv'):
            inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    p=project_path(view)/'view.json'
    inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    features,sensitivities={},[]
    with torch.no_grad():
        for row in frame.itertuples():
            ds=datasets[row.split]
            if ds.frame.iloc[row.dataset_index].patient_id!=row.patient_id:
                raise ValueError('Saved selection mismatch')
            image=torch.from_numpy(ds.image_01(row.dataset_index).copy())[None,None]
            p=project_path(row.source_image_path)
            inputs[p.relative_to(ROOT).as_posix()]=digest(p)
            for name,function in functions.items():
                guide=function(image)
                features.setdefault(name,[]).append(F.adaptive_avg_pool2d(guide,(8,8)).flatten().numpy())
                values=[function(image**.7),function(image**1.3),function(image*.8+.04),
                        function(torch.roll(image,(8,8),(-2,-1)))]
                sensitivities.append(dict(patient_id=row.patient_id,split=row.split,representation=name,
                    **{label:float((guide-value).abs().mean()) for label,value in zip(
                        ('gamma07_l1','gamma13_l1','affine_l1','translation8_l1'),values)}))
    train=frame.split.eq('train').to_numpy()
    val=~train
    probes,predictions=[],[]
    for name,records in features.items():
        values=np.stack(records)
        mean,std=values[train].mean(0),values[train].std(0).clip(1e-4)
        x=np.column_stack([np.ones(len(values)),(values-mean)/std]).astype(np.float64)
        for axis in ('manufacturer','field_strength'):
            y=frame[axis].to_numpy()
            counts=np.bincount(y[train],minlength=2)
            weights=1/counts[y[train]]
            weights/=weights.mean()
            penalty=np.eye(x.shape[1])*10
            penalty[0,0]=0
            coef=np.linalg.solve(x[train].T@(weights[:,None]*x[train])+penalty,
                                 x[train].T@(weights*y[train]))
            score=x[val]@coef
            predicted=(score>=.5).astype(int)
            lo,hi=confidence(y[val],predicted)
            probes.append(dict(representation=name,axis=axis,feature_dimensions=values.shape[1],
                balanced_accuracy=balanced_score(y[val],predicted),ci95_lo=lo,ci95_hi=hi,
                train_patients=int(train.sum()),validation_patients=int(val.sum()),
                scanner_invariance_established=False))
            predictions.extend(dict(representation=name,axis=axis,patient_id=pid,truth=int(t),prediction=int(p),score=float(s))
                for pid,t,p,s in zip(frame[val].patient_id,y[val],predicted,score))
    csv_write(output/'patient_selection.csv',frame)
    csv_write(output/'sensitivities.csv',pd.DataFrame(sensitivities))
    csv_write(output/'leakage_probe.csv',pd.DataFrame(probes))
    csv_write(output/'probe_predictions.csv',pd.DataFrame(predictions))
    with (output/'features.npz').open('xb') as handle:
        np.savez_compressed(handle,**{k:np.stack(v) for k,v in features.items()})
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'scripts.v8_generator_dev.test_detail_candidates','scripts.v8_generator_dev.test_detail_diagnosis'])
    with (output/'engineering_tests.txt').open('x',encoding='utf-8') as handle,redirect_stdout(handle),redirect_stderr(handle):
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Local engineering tests failed; retain this output')
    json_write(output/'runtime.json',dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,device='cpu',
        known_environment_warning='Installed SciPy declares NumPy<2.3; local NumPy is newer. Existing warning retained; no environment parity/compatibility claim.'))
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,tests_run=result.testsRun,failures=0,
        train_patients=int(train.sum()),validation_patients=int(val.sum()),
        untrained_prototypes_only=True,cloud_ready=False,new_efficacy_checkpoint=False,
        full_batch_final_parent_smoke=False,paid_compute=False,test_pixels_used=False,
        clinical_preservation='unknown',scanner_invariance_established=False))
    (output/'README.md').write_text(
        '# Detail-candidate local preparation\n\nProducer: `python -m scripts.v8_generator_dev.audit_detail_candidates --run-id '+run_id+'`.\n\n'
        '`protocol.json`: fixed patient selection, weak probes and sensitivities. `patient_selection.csv`: all train/validation patients. '
        '`features.npz`, `leakage_probe.csv`, `probe_predictions.csv`: descriptors, patient CIs and every fixed-probe prediction. '
        '`sensitivities.csv`: all photometric/spatial responses. `engineering_tests.txt`: actual synthetic optimiser updates, '
        'CFG/absent/zero controls on both tiny backbones and full/empty/partial mask endpoints. '
        '`source_snapshot/`, `inputs.json`: archived code and source hashes. `runtime.json`: actual packages/warning. '
        '`complete.json`: limits. `EXPERIMENT_PLAN.md`: separately authored research proposal, not approved cloud work. '
        'The six-channel adapter requires retraining; existing trained adapters cannot consume it. '
        'Tiny discarded synthetic updates establish software wiring only. No trained finer-guide model or full-parent/full-batch evidence.\n',encoding='utf-8')
    print(pd.DataFrame(probes).to_string(index=False),flush=True)
    print('DETAIL CANDIDATE AUDIT COMPLETE',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    audit(parser.parse_args().run_id)
