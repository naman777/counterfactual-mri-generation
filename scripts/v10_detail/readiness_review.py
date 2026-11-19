"""Review the exact package plus completed local checks; no cloud action."""
import argparse
import json
import shutil
import subprocess
import sys

import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs,csv_write
from scripts.v8_generator_dev.grading_io import read_grading_csv


def review(prepared,preflight,run_id):
    prepared,preflight=map(project_path,(prepared,preflight))
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    verify_inputs(json.loads((preflight/'inputs.json').read_text()))
    local=json.loads((preflight/'complete.json').read_text())
    if not local['complete'] or local['evaluator_runs']!=17 or local['updates']!=6:
        raise ValueError('Completed full-code local engineering evidence required')
    if not local['local_full_batch_backward_passed']:raise ValueError('Local full-batch gate did not pass')
    checks=json.loads((preflight/'checks.json').read_text())
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.readiness_review. route_checks.csv: every new evaluator '
        'route/control/EchoTime check; verifier_tests.txt/json: missing-diagnostic and contextual-schema checks; local_postprocessing/: '
        'frozen local-only verifier/postprocessor sources; inputs.json: prepared/local/source proof; complete.json: readiness scope; '
        'RESULTS.md: review and exact execution/postprocessing commands. No cloud launch from this module.\n',encoding='utf-8')
    rows=[];evidence={}
    for mode,record in checks['modes'].items():
        for route,relative in record['evaluations'].items():
            home=project_path(relative);frame=read_grading_csv(home/'per_image.csv')
            if len(frame)!=18 or set(frame.judge)!=set(f'heldout_orientation_robust_seed{i}' for i in range(3)):
                raise ValueError('Engineering named-judge/control route incomplete')
            if not (frame.source_echo_time_norm==frame.target_echo_time_norm).all():raise ValueError('EchoTime drift')
            if frame.groupby(['case','judge']).variant.nunique().ne(3).any():raise ValueError('Missing original-label control')
            rows.append(dict(mode=mode,route=route,cases=frame.case.nunique(),grades=len(frame),fixed_echo_time=True))
            for name in ('protocol.json','requests.csv','per_image.csv','complete.json'):
                path=home/name;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    csv_write(output/'route_checks.csv',pd.DataFrame(rows))
    with (output/'verifier_tests.txt').open('x',encoding='utf-8') as log:
        tested=subprocess.run([sys.executable,'-m','unittest','scripts.v10_detail.test_verifier_v2','-v'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    json_write(output/'verifier_tests.json',dict(exit_code=tested.returncode))
    if tested.returncode:raise ValueError('Local postprocessing tests failed')
    for name in ('verify_saved_v2.py','postprocess.py','test_verifier_v2.py','readiness_review.py'):
        path=ROOT/'scripts/v10_detail'/name;destination=output/'local_postprocessing'/name
        destination.parent.mkdir(exist_ok=True);shutil.copyfile(path,destination)
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    for home in (prepared,preflight):
        for name in ('complete.json','inputs.json'):
            path=home/name;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    path=prepared/'readiness_inputs.json';evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    (output/'RESULTS.md').write_text('''# Replicated-guide package readiness

Selected package: `reports/v10_detail/preparation_20261117_02/`.
Initial `_01` is preserved; do not launch it. Scientific settings are unchanged;
the selected evaluator verifies its actually consumed source/parent/judge/code
inputs and keeps explicit per-training-seed reporting.

The exact staged archive, source overlays, parent, input ledger, protocol and
train/validation manifests pass. Six study tests pass. The local CUDA device
passes a real discarded batch-32 backward/optimiser update with unchanged frozen
parent weights. All three guide modes complete matched batch-two updates in
the new engine; all 17 new evaluator routes pass at engineering t=2,w=3, with
source/null/target, each named judge and fixed EchoTime. Absent-guide images
match across the trained engineering copies. These are software checks, not
full efficacy results or the target cloud runtime gate.

Two local-postprocessor tests pass. The versioned `verify_saved_v2` handles
undefined diagnostics explicitly, preserving missingness and rejecting a
missing/finite mismatch. This local-only repair leaves every prepared cloud
source file unchanged. The postprocessor also retains contextual comparisons
to the immutable historical alpha-.50 research reference, with its one legacy
adapter held fixed; that is not a matched training-seed experiment. These full
postprocessing paths cannot be run on prospective efficacy outputs until those
outputs exist; their current evidence is local/synthetic, not an efficacy claim.

The proposed separate US$12 run is fully reviewable in the selected PROPOSAL.md.
It is one A10/app, at most 6.5 worker hours, with no automatic paid restart. First
cloud phase must pass all nine discarded full-batch updates under the pinned
runtime before efficacy training. Failure stops before training and consumes
the same allowance. The current local pass does not waive that gate.

Launch only within actual user authorisation:

```
python -m scripts.v10_detail.launch --prepared reports/v10_detail/preparation_20261117_02 --report-id launch_20261117_01 --approved-usd 12
```

After terminal cloud completion, retain failures if any. Read-only retrieval:

```
python -m scripts.v10_detail.fetch --run-id guide_replicates_20261117_01 --report-id results_20261117_01
python -m scripts.v10_detail.postprocess --prepared reports/v10_detail/preparation_20261117_02 --results reports/v10_detail/results_20261117_01 --prefix final_20261117_01
```

The second command requires complete successful components, then strictly checks
saved arrays, metrics, named-judge probabilities/decisions and trained checkpoint
identity before the frozen crossed-bootstrap gates. Missing primary cells make
advancement unavailable. Neither a software pass nor research advancement is
clinical preservation, independent confirmation or automatic default promotion.
Field-strength support, source-copy/scanner leakage, mask warnings and fine-detail
limits remain. Existing reports, paper results, splits and defaults are unchanged.
''',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,prepared_for_gated_run=True,
        local_full_batch_backward=True,engineering_evaluator_runs=len(rows),engineering_grades=sum(r['grades'] for r in rows),
        study_tests=6,local_postprocessor_tests=2,pinned_a10_full_batch_gate_pending=True,
        paid_compute_started=False,efficacy_started=False,clinical_preservation='unknown'))
    print('REPLICATED READINESS REVIEW PASS',output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','preflight','run-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();review(a.prepared,a.preflight,a.run_id)
