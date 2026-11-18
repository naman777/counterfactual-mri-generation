"""Append-only report setup and equal-patient uncertainty for local audits."""
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, digest, json_write, verify_inputs

BASE=ROOT/'reports/v9_detail'
VIEW='data/processed/ax_t1_128_canonical/view_v1_20261116_01'
SELECTION=ROOT/'reports/v8_generator_dev/canonical_guide_audit_20261116_01/patient_selection.csv'
VAL_SELECTION=ROOT/'reports/v8_generator_dev/adapter_preparation_20261117_02/validation_selection.csv'


def start(run_id,producer,description):
    if not run_id.replace('_','').isalnum():
        raise ValueError('Use a safe fresh report ID')
    output=BASE/run_id
    output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('# '+description+'\n\nProducer: `python -m '+producer+' --run-id '+run_id+'`.\n'
        'protocol.json is written before measurements. inputs.json/source_snapshot bind consumed inputs/code; '
        'runtime.json records actual packages. complete.json is emitted only after successful checks. '
        'RESULTS.md interprets numerical tables. No test data, clinical labels or efficacy training.\n',encoding='utf-8')
    inputs={}
    files=list((ROOT/'scripts/v9_detail').glob('*.py'))+[
        ROOT/'scripts/v8_generator_dev'/name for name in ('data.py','io.py','detail_candidates.py',
        'soft_structure.py','preservation_screen.py','preservation_screen_v2.py','guide_audit.py')]
    files += [ROOT/'scripts/v5_anatomy/metrics.py',ROOT/'scripts/v6_structure/model.py',
              ROOT/'scripts/v7_geometry/orientation.py',SELECTION,VAL_SELECTION,ROOT/VIEW/'view.json']
    for split in ('train','val'):
        files.extend([ROOT/VIEW/'splits'/f'{split}.csv',ROOT/'data/processed/ax_t1_128/splits'/f'{split}.csv'])
    for path in files:
        relative=path.relative_to(ROOT).as_posix()
        inputs[relative]=digest(path)
        if path.suffix=='.py':
            dest=output/'source_snapshot'/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    json_write(output/'runtime.json',dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__),device='cpu',
        known_warning='Existing SciPy declares NumPy<2.3; local NumPy is newer. Compatibility/parity not established.'))
    return output,inputs


def sources(inputs):
    training=pd.read_csv(SELECTION)
    training=training[training.split.eq('train')].copy()
    validation=pd.read_csv(VAL_SELECTION)
    validation=validation.assign(manufacturer=validation.manufacturer_label,
                                 field_strength=validation.field_strength_label)
    columns=['split','patient_id','sop_uid','dataset_index','manufacturer','field_strength','source_image_path']
    frame=pd.concat([training[columns],validation[columns]],ignore_index=True)
    if frame.patient_id.duplicated().any():
        raise ValueError('One source per patient required')
    datasets={s:CanonicalDataset(VIEW,s) for s in ('train','val')}
    result=[]
    for row in frame.itertuples():
        ds=datasets[row.split]
        current=ds.frame.iloc[row.dataset_index]
        if current.patient_id!=row.patient_id:
            raise ValueError('Patient selection mismatch')
        image=ds.image_01(row.dataset_index)
        path=ROOT/str(row.source_image_path).replace('\\','/')
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        result.append((row,image))
    return frame,result


def interval(frame,metric,seed=583101,n=10000):
    values=frame.groupby('patient_id')[metric].mean().dropna().to_numpy()
    if not len(values):
        return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,patients_valid=0,
                    patients_total=frame.patient_id.nunique())
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(values),size=(n,len(values)))
    lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
    return dict(estimate=float(values.mean()),ci_low=float(lo),ci_high=float(hi),
                patients_valid=len(values),patients_total=frame.patient_id.nunique())


def finish(output,inputs,**scope):
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,test_pixels_used=False,
        efficacy_training=False,paid_compute=False,clinical_preservation='unknown',**scope))


def display(row):
    return f'{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]'
