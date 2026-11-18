"""Append a new local follow-up ledger requiring the actual-training auditor."""
import argparse
import ast
import json

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs


def freeze(prior,auditor,report_id):
    prior,auditor=map(project_path,(prior,auditor))
    verify_inputs(json.loads((prior/'readiness_inputs.json').read_text()))
    verify_inputs(json.loads((auditor/'inputs.json').read_text()))
    if not json.loads((auditor/'complete.json').read_text())['complete']:
        raise ValueError('Real cloud segment audit preflight required')
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.freeze_followup_v2. commands.json: '
        'ordered retrieval, actual-training audit, independent image/judge regrading, guarded ten-gate '
        'analysis and report commands; readiness_inputs.json: new local source/prior readiness/auditor hashes; '
        'complete.json: follow-up preparation only. No cloud runtime/science/budget change.\n',encoding='utf-8')
    commands=json.loads((prior/'commands.json').read_text())
    prepared=commands['verify'][commands['verify'].index('--prepared')+1]
    commands['verify_training']=[commands['fetch'][0],'-m','scripts.v13_guide_dropout.verify_training',
        '--prepared',prepared,'--training','reports/v13_guide_dropout/guide_dropout_20261118_01_training',
        '--run-id','training_verified_20261118_01']
    commands['analyse'][2]='scripts.v13_guide_dropout.analyse_v2'
    commands['analyse'] += ['--training-verified','reports/v13_guide_dropout/training_verified_20261118_01']
    ordered={name:commands[name] for name in ('fetch','verify_training','verify','analyse','report')}
    json_write(out/'commands.json',ordered)
    files=list((ROOT/'scripts/v13_guide_dropout').glob('*.py'))
    for path in files:ast.parse(path.read_text(encoding='utf-8-sig'))
    files += [prior/'readiness_inputs.json',auditor/'complete.json',auditor/'inputs.json',out/'commands.json']
    json_write(out/'readiness_inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in files})
    json_write(out/'complete.json',dict(complete=True,actual_training_audit_required=True,
        prior_followup_preserved=True,cloud_science_changed=False,new_paid_job_started=False))
    print('GUARDED LOCAL FOLLOWUP FROZEN',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prior','auditor','report-id'):ap.add_argument('--'+name,required=True)
    freeze(**vars(ap.parse_args()))
