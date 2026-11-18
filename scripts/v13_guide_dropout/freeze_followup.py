"""Freeze local retrieval/regrading/analysis/reporting routes before the cloud launch."""
import argparse
import ast
import json
import subprocess
import sys

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs


def freeze(prepared,preflight,report_id):
    prepared,preflight=map(project_path,(prepared,preflight))
    if not json.loads((preflight/'complete.json').read_text())['complete']:
        raise ValueError('All local evaluator routes must pass')
    verify_inputs(json.loads((prepared/'readiness_inputs.json').read_text()))
    verify_inputs(json.loads((preflight/'inputs.json').read_text()))
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.freeze_followup. tests.txt/json: '
        'ten-gate/cached-baseline and invalid-cell tests; commands.json: exact no-new-cloud follow-up commands; '
        'readiness_inputs.json: exact local follow-up/protocol/preflight/old-baseline identities; '
        'complete.json: preparation only. Scientific outcomes and generated figure inspection remain pending.\n',encoding='utf-8')
    files=list((ROOT/'scripts/v13_guide_dropout').glob('*.py'))
    for f in files:ast.parse(f.read_text(encoding='utf-8-sig'))
    with (out/'tests.txt').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','unittest','scripts.v13_guide_dropout.test_analysis','-v'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    json_write(out/'tests.json',dict(exit_code=result.returncode,tests_passed=2 if result.returncode==0 else None))
    if result.returncode:raise ValueError('Prospective analysis tests failed')
    run='guide_dropout_20261118_01';rel=prepared.relative_to(ROOT).as_posix()
    result_path='reports/v13_guide_dropout/results_20261118_01'
    verified='reports/v13_guide_dropout/verified_20261118_01'
    analysis='reports/v13_guide_dropout/analysis_20261118_01'
    commands=dict(
        fetch=[sys.executable,'-m','scripts.v13_guide_dropout.fetch','--run-id',run,'--report-id','results_20261118_01'],
        verify=[sys.executable,'-m','scripts.v13_guide_dropout.verify_saved','--prepared',rel,
                '--results',result_path,'--run-id','verified_20261118_01'],
        analyse=[sys.executable,'-m','scripts.v13_guide_dropout.analyse','--prepared',rel,'--results',result_path,
                 '--verified',verified,'--run-id','analysis_20261118_01'],
        report=[sys.executable,'-m','scripts.v13_guide_dropout.report','--prepared',rel,'--results',result_path,
                '--analysis',analysis,'--run-id','report_20261118_01'])
    json_write(out/'commands.json',commands)
    paths=files+[prepared/'protocol.json',prepared/'readiness_inputs.json',preflight/'complete.json',preflight/'inputs.json',
        ROOT/'reports/v10_detail/results_20261117_01/complete.json',
        ROOT/'reports/v10_detail/final_20261117_02_verified/complete.json',
        ROOT/'scripts/v11_guide_audit/analyse.py',ROOT/'scripts/v10_detail/analyse.py',
        out/'tests.json',out/'commands.json']
    json_write(out/'readiness_inputs.json',{f.relative_to(ROOT).as_posix():digest(f) for f in paths})
    json_write(out/'complete.json',dict(complete=True,analysis_tests_passed=True,all_local_source_parsed=True,
        followup_commands_frozen=True,cloud_launched=False,paid_retry=False))
    print('LOCAL FOLLOWUP FROZEN',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','preflight','report-id'):ap.add_argument('--'+name,required=True)
    freeze(**vars(ap.parse_args()))
