"""Task-specific local monitor/retrieval/verification; never starts or restarts a paid app."""
import argparse
import json
import subprocess
import time

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v13_guide_dropout.monitor import snapshot


def run(run_id,followup,report_id):
    followup=project_path(followup)
    verify_inputs(json.loads((followup/'readiness_inputs.json').read_text()))
    commands=json.loads((followup/'commands.json').read_text())
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.lifecycle. protocol.json: '
        'read-only monitoring and frozen local follow-up; *_stdout.txt and phase_*.json: actual '
        'subprocess logs/exits; complete.json: local follow-up production, with visual inspection '
        'still pending; requires_local_evaluation.json: incomplete cloud matrix; failed.json: '
        'stopped local verification/error. No paid app invocation or retry occurs here.\n',encoding='utf-8')
    json_write(out/'protocol.json',dict(run_id=run_id,followup=followup.relative_to(ROOT).as_posix(),
        commands_sha256=digest(followup/'commands.json'),source_sha256=digest(__file__),
        paid_app_invocations=0,maximum_monitor_seconds=27000,visual_inspection_automated=False))
    started=time.monotonic();completed=[]

    def call(name,timeout):
        verify_inputs(json.loads((followup/'readiness_inputs.json').read_text()))
        with (out/(name+'_stdout.txt')).open('x',encoding='utf-8') as log:
            result=subprocess.run(commands[name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        record=dict(phase=name,exit_code=result.returncode,command=commands[name])
        json_write(out/('phase_'+name+'.json'),record);completed.append(record)
        if result.returncode:raise RuntimeError('Local follow-up failed: '+name)

    try:
        while True:
            state=snapshot(run_id,'monitor_20261118_01')
            if state['state'] in ('complete.json','stopped_for_budget.json','failed.json'):break
            if time.monotonic()-started>27000:raise TimeoutError('Local monitor limit; inspect provider before further action')
            time.sleep(60)
        call('fetch',1800)
        if state['state']!='complete.json' or not state['terminal']['workflow_success']:
            training=ROOT/'reports/v13_guide_dropout'/f'{run_id}_training/complete.json'
            training_complete=training.exists() and json.loads(training.read_text()).get('complete',False)
            if training_complete:call('verify_training',1800)
            json_write(out/'requires_local_evaluation.json',dict(complete=False,terminal=state['state'],
                full_cloud_training_complete=training_complete,partial_evidence_retrieved=True,
                automatic_paid_retry=False,hardware_parity_required_before_pooling_inference=True,
                cross_runtime_training_resume_validated=False))
            return
        call('verify_training',1800)
        call('verify',7200)
        call('analyse',7200)
        call('report',1800)
        json_write(out/'complete.json',dict(complete=True,local_followup_completed=True,phases=completed,
            report='reports/v13_guide_dropout/report_20261118_01/RESULTS.md',
            report_visual_inspection_pending=True,default_promoted=False,clinical_preservation='unknown'))
        print('LOCAL REPORT READY FOR VISUAL REVIEW',flush=True)
    except Exception as error:
        json_write(out/'failed.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            phases=completed,automatic_paid_retry=False))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('run-id','followup','report-id'):ap.add_argument('--'+name,required=True)
    run(**vars(ap.parse_args()))
