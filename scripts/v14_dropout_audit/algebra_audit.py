"""Check candidate fixed-input score identities on saved vectors, without new sampling."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v14_dropout_audit.shared_residual import parent_response_score


def run(probe,verified,run_id):
    probe,verified=map(project_path,(probe,verified));verify_inputs(json.loads((verified/'inputs.json').read_text()))
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.algebra_audit. tests.txt/json: '
        'analytical parent/zero-guidance/contrast identities; checks.json: candidate epsilon identity '
        'on all saved fixed-source points (not newly sampled MRIs); NEXT_LOCAL_PROTOTYPE.md: '
        'concrete unvalidated inference proposal; inputs.json: hashes; complete.json: algebra-only scope.\n',encoding='utf-8')
    with (out/'tests.txt').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','unittest','scripts.v14_dropout_audit.test_shared_residual','-v'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    json_write(out/'tests.json',dict(exit_code=result.returncode))
    if result.returncode:raise ValueError('Candidate algebra tests failed')
    rows=pd.read_csv(probe/'per_forward.csv',keep_default_na=False,na_values=[''])
    parents={};max_error=0.;absent_exact=0;zero_exact=0
    for row in rows.itertuples():
        if row.parent_file not in parents:
            with np.load(probe/row.parent_file,allow_pickle=False) as z:parents[row.parent_file]=torch.from_numpy(z['epsilon'].copy())
        parent=parents[row.parent_file]
        with np.load(probe/row.epsilon_file,allow_pickle=False) as z:guided=torch.from_numpy(z['epsilon'].copy())
        source=parent_response_score(guided[2],parent[0],parent[2],3.)
        target=parent_response_score(guided[2],parent[1],parent[2],3.)
        error=float(torch.max(torch.abs((target-source)-3*(parent[1]-parent[0]))));max_error=max(max_error,error)
        if error>1e-6:raise ValueError('Candidate fixed-x parent-contrast identity failed')
        zero=parent_response_score(guided[2],parent[1],parent[2],0.)
        if not torch.equal(zero,guided[2]):raise ValueError('Zero-guidance endpoint changed')
        zero_exact+=1
        if row.guide_control=='absent':
            expected=parent[2]+3*(parent[1]-parent[2])
            if not torch.equal(target,expected):raise ValueError('Absent-parent CFG endpoint changed')
            absent_exact+=1
    json_write(out/'checks.json',dict(saved_observations=len(rows),maximum_float32_contrast_roundoff=max_error,
        exact_zero_guidance_endpoints=zero_exact,exact_absent_parent_endpoints=absent_exact,
        fixed_guidance=3.,new_forward_calls=0,new_reverse_sampling=False))
    proposal='''# Next local prototype: shared guide residual with parent label response

This is a proposal derived after the completed negative outcome, not a trained
improvement or full sampling implementation. `shared_residual.py` implements
only the epsilon-combination primitive and its fixed-input algebra is checked.

At each noisy image x, evaluate the frozen parent's conditional and unconditional
epsilon, and the guided model's unconditional epsilon. Use

    epsilon = guided_unconditional + w * (parent_conditional - parent_unconditional)

The guide contributes the same residual for source and target labels. At a
fixed x and guide, their difference is exactly the frozen parent's conditional
contrast times w, up to floating-point roundoff. An absent guide recovers parent
CFG, and w=0 recovers the guided unconditional output. This differs from changing
only the unconditional guide: that change alone cannot restore a suppressed
conditional difference at fixed x.

The identity does not prove useful scanner editing. The parent's conditional
signal may be weak or spurious, particularly for exploratory field strength.
Reverse trajectories will differ; source texture/boundaries or lesions may change.
The guide can still leak scanner information. Existing negative outcomes stay.

Next implementation steps, all local before any efficacy claim:

1. Implement a distinct sampler preserving initial/reverse noise draws, source
   EchoTime and field strength for manufacturer requests. Use the existing fixed
   operating point, with no validation search over residual scales or guidance.
2. Exercise short-timestep engineering routes for every trained copy: absent
   guide equals parent CFG, zero guidance equals the ordinary guided sampler,
   zero input remains distinct from absence, all weights/conditioner stay frozen.
3. Benchmark the laptop on a declared fixed case, then freeze a bounded full
   paired validation matrix and resource/time plan. Keep all source patients,
   generation/training seeds, six guide controls, named judges, warnings and
   support strata; retain cached comparisons only after runtime/noise parity.
4. Evaluate any complete pilot against declared editing and source-correspondence
   safeguards, with paired CIs and clinical/independent-confirmation limits.

No full reverse-sampling pilot, retraining, new cloud spend or default change is
authorised or performed by this algebra check. No previous budget is reopened.
'''
    (out/'NEXT_LOCAL_PROTOTYPE.md').write_text(proposal,encoding='utf-8')
    paths=[probe/'per_forward.csv',verified/'complete.json',verified/'inputs.json',project_path(__file__),
           ROOT/'scripts/v14_dropout_audit/shared_residual.py',ROOT/'scripts/v14_dropout_audit/test_shared_residual.py']
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/'complete.json',dict(complete=True,engineering_algebra_only=True,saved_vector_checks=len(rows),
        full_sampler_implemented=False,new_reverse_sampling=False,paid_gpu_calls=0,efficacy_established=False))
    print('CANDIDATE ALGEBRA CHECKED; FULL SAMPLER REMAINS UNVALIDATED',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for name in ('probe','verified','run-id'):a.add_argument('--'+name,required=True)
    run(**vars(a.parse_args()))
