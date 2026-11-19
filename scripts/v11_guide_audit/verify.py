"""Independently recompute every new image-change diagnostic and primary identity."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v9_detail.masks import noise_aware_mask


def verify(prepared,analysis,run_id):
    prepared,analysis=map(project_path,(prepared,analysis))
    verify_inputs(json.loads((analysis/'inputs.json').read_text()))
    p=json.loads((prepared/'protocol.json').read_text())
    data=pd.read_csv(analysis/'case_metrics.csv')
    contrasts=pd.read_csv(analysis/'contrast_intervals.csv')
    cases=pd.read_csv(analysis/'primary_cases.csv')
    assert len(data)==6885 and len(cases)==810
    assert not data.duplicated(['profile','case','judge']).any()
    assert contrasts[['axis','contrast']].drop_duplicates().shape[0]==27
    assert not contrasts.duplicated(['axis','contrast','judge','stratum','metric']).any()
    np.testing.assert_array_equal(data.target_hit-data.null_hit,data.paired_hit_delta)
    np.testing.assert_allclose(cases.target_hit_difference-cases.null_hit_difference,cases.paired_hit_delta_difference,atol=1e-14)
    old=ROOT/'reports/v10_detail/final_20261117_02_analysis/advancement.json'
    gates=json.loads(old.read_text())
    for g in gates['gates']:
        metric=('null_'+g['metric']) if g['control']=='null' else g['metric']
        row=contrasts[(contrasts.axis=='manufacturer')&(contrasts.contrast=='fine_minus_coarse__correct')&
            (contrasts.judge==g['judge'])&(contrasts.stratum=='all')&(contrasts.metric==metric)]
        assert len(row)==1
        np.testing.assert_allclose(row.estimate.iloc[0],g['estimate'],rtol=0,atol=1e-12)
    # Independent separable Torch convolution, not the SciPy producer calculation.
    torch.set_num_threads(2)
    grid=torch.arange(-3,4,dtype=torch.float64);kernel=torch.exp(-grid.square()/2);kernel/=kernel.sum()
    kx=kernel.reshape(1,1,1,7);ky=kernel.reshape(1,1,7,1)
    receipt=json.loads(project_path(p['previous_results']).read_text())
    masks={};count=0;maximum=0.; evidence={}
    for profile,relative in receipt['evaluations'].items():
        home=project_path(relative);requests=pd.read_csv(home/'requests.csv')
        changes=[];domains=[]
        for r in requests.itertuples():
            path=home/r.array_file
            assert digest(path)==r.array_sha256
            with np.load(path) as z:
                if r.dataset_index not in masks:
                    masks[r.dataset_index]=ndi.binary_erosion(noise_aware_mask(z['source'])[0],iterations=8,border_value=0)
                changes.append(z['target'].astype(np.float64)-z['null'].astype(np.float64));domains.append(masks[r.dataset_index])
            evidence[path.relative_to(ROOT).as_posix()]=r.array_sha256
        images=torch.from_numpy(np.stack(changes))[:,None]
        low=F.conv2d(F.conv2d(images,kx,padding=(0,3)),ky,padding=(3,0))[:,0].numpy()
        for i,r in enumerate(requests.itertuples()):
            saved=data[(data.profile==profile)&(data.case==r.case)]
            assert len(saved)==3
            mask=domains[i]
            for name,array in [('null_to_target_interior_mad',np.abs(changes[i])),
                               ('null_to_target_hf_mad',np.abs(changes[i]-low[i]))]:
                if mask.sum()<64:
                    assert saved[name].isna().all();continue
                expected=float(array[mask].mean())
                maximum=max(maximum,float(np.abs(saved[name]-expected).max()))
                np.testing.assert_allclose(saved[name],expected,rtol=0,atol=1e-10)
            count+=1
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','unittest','scripts.v11_guide_audit.test_analysis','-v']
    tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (output/'tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode:raise ValueError('Diagnostic regressions failed')
    for path in (analysis/'inputs.json',analysis/'case_metrics.csv',analysis/'contrast_intervals.csv',old,
                 project_path(__file__),ROOT/'scripts/v11_guide_audit/test_analysis.py'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,arrays_independently_checked=count,
        method='Independent float64 Torch separable convolution; source mask eroded8 excludes convolution image borders',
        maximum_new_metric_error=maximum,original_primary_point_estimates_equal=True,
        target_null_decomposition_equal=True,tests_passed=4,paid_gpu_calls=0,clinical_preservation='unknown'))
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.verify. complete.json: every new distance/HF '
        'diagnostic independently recomputed with Torch convolution, original primary-point and target/null identities; '
        'tests.txt: independent index-bootstrap/missingness/pairing/image checks; inputs.json: exact arrays/outputs/code. '
        'No saved output is regenerated or clinical/causal claim established.\n',encoding='utf-8')
    print('INDEPENDENT DIAGNOSIS CHECK',count,maximum,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for n in ('prepared','analysis','run-id'):a.add_argument('--'+n,required=True)
    q=a.parse_args();verify(q.prepared,q.analysis,q.run_id)
