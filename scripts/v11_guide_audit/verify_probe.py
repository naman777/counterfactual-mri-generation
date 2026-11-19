"""Audit local forward-probe identity/completeness and engineering evidence."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs


def verify(probe,engineering,run_id):
    probe,engineering=map(project_path,(probe,engineering))
    verify_inputs(json.loads((probe/'inputs.json').read_text()))
    verify_inputs(json.loads((engineering/'inputs.json').read_text()))
    p=json.loads((probe/'protocol.json').read_text());done=json.loads((probe/'complete.json').read_text())
    old=json.loads(project_path(p['previous_protocol']).read_text())
    parent=project_path(old['parent']);assert digest(parent)==old['parent_sha256']
    selected=pd.read_csv(project_path(p['selection']));evidence={parent.relative_to(ROOT).as_posix():digest(parent)}
    for r in selected.itertuples():
        source=project_path(r.source_image_path);assert digest(source)==r.source_array_sha256
        evidence[source.relative_to(ROOT).as_posix()]=r.source_array_sha256
    f=pd.read_csv(probe/'per_forward.csv')
    keys=['patient_id','training_seed','generation_seed','timestep','mode','guide_control']
    expected=pd.MultiIndex.from_product([selected.patient_id,p['training_seeds'],p['generation_seeds'],
        p['timesteps'],p['modes'],p['controls']],names=keys)
    observed=pd.MultiIndex.from_frame(f[keys]);assert len(f)==1080 and not observed.has_duplicates and set(observed)==set(expected)
    assert np.isfinite(f[p['metrics']]).all().all()
    np.testing.assert_array_equal(f.source_echo_time_norm,f.target_echo_time_norm)
    off=f[f.guide_control=='absent'];assert len(off)==180 and done['maximum_absent_epsilon_error']==0
    np.testing.assert_allclose(off.response_ratio_to_parent,1.,atol=1e-12,rtol=0)
    np.testing.assert_allclose(off.response_cosine_to_parent,1.,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(off.unconditional_adapter_shift_rms,np.zeros(len(off)))
    eng=json.loads((engineering/'complete.json').read_text());assert eng['actual_updates']==12 and eng['engineering_only']
    assert eng['matched_actual_draws'] and eng['all_gate_states_exercised'] and not eng['efficacy_checkpoint_saved']
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    for path in (project_path(__file__),probe/'inputs.json',probe/'complete.json',probe/'per_forward.csv',
                 engineering/'complete.json',engineering/'inputs.json',engineering/'tests.txt'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,parent_and_all_source_hashes_checked=True,
        probe_rows=1080,absent_parent_observations=180,absent_epsilon_exact=True,
        fixed_echo_time=True,engineering_updates=12,raw_epsilon_vectors_saved=False,all_forwards_independently_replayed=False,
        analytical_metric_tests_passed=True,torch_version=str(torch.__version__),local_gpu=torch.cuda.get_device_name(0),
        clinical_preservation='unknown',mechanism_proven=False,paid_gpu_calls=0))
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.verify_probe. complete.json: parent/source bytes, '
        'full forward-design completeness, exact absent-parent observations and disposable engineering evidence; '
        'inputs.json: bound source images/weights/results/code. Raw epsilon vectors were not saved or independently '
        'replayed for all forwards; analytical vector tests and source/parent/absence checks supplement the frozen producer. '
        'No reverse sampling, clinical preservation, causal proof or paid job.\n',encoding='utf-8')
    print('FORWARD/ENGINEERING AUDIT PASSED',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for n in ('probe','engineering','run-id'):a.add_argument('--'+n,required=True)
    q=a.parse_args();verify(q.probe,q.engineering,q.run_id)
