"""Read-only compact cloud snapshots; no GPU calls, launches or restarts."""
import argparse
import json
import os
import time
from datetime import datetime,timezone,timedelta

from scripts.v8_generator_dev.io import ROOT,json_write,digest


def snapshot(run_id,report_id):
    if not all(x.replace('_','').isalnum() for x in (run_id,report_id)):raise ValueError('Safe monitor IDs')
    os.environ['MODAL_PROFILE']='namankundra7'
    import modal
    output=ROOT/'reports/v10_detail'/report_id;output.mkdir(parents=True,exist_ok=True)
    if not (output/'protocol.json').exists():
        json_write(output/'protocol.json',dict(run_id=run_id,read_only=True,producer_sha256=digest(__file__)))
        (output/'README.md').write_text('Producer: scripts.v10_detail.monitor. protocol.json: exact run/read-only scope; '
            'snapshot_*.json: append-only observed terminal/phase/training/log state with IST times. '
            'No GPU invocation, restart, result grading or expenditure authorisation.\n',encoding='utf-8')
    elif json.loads((output/'protocol.json').read_text())['run_id']!=run_id:raise ValueError('Monitor identity changed')
    record=dict(time_ist=datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),run_id=run_id)
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False);home='v10_detail/'+run_id
    def read(path):return b''.join(volume.read_file(path))
    try:
        entries=volume.listdir(home)
        names={e.path.lstrip('/').rsplit('/',1)[-1] for e in entries}
        if 'complete.json' in names:
            root=json.loads(read(home+'/complete.json'));record.update(state='complete',seconds=root['seconds'],
                evaluations=len(root['evaluations']),strict_local_regrading_pending=True)
        elif 'failed.json' in names:
            record.update(state='failed',failure=json.loads(read(home+'/failed.json')))
        else:
            done=[n[:-5] for n in names if n.endswith('.json') and n not in ('authorisation.json',) and not n.startswith('common_epoch')]
            running=sorted(n for n in names if n.endswith('.txt') and n[:-4] not in done)
            record.update(state='running',completed_phases=done,active_phase=running,
                common_epochs=sorted(n for n in names if n.startswith('common_epoch_')))
            for name in running:
                lines=read(home+'/'+name).decode('utf-8',errors='replace').splitlines()
                record.setdefault('tails',{})[name]=lines[-3:]
    except modal.exception.NotFoundError:
        record.update(state='awaiting_worker_staging')
    json_write(output/f'snapshot_{time.time_ns()}.json',record)
    print(json.dumps(record,allow_nan=False),flush=True)
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run-id',required=True);ap.add_argument('--report-id',required=True)
    a=ap.parse_args();snapshot(a.run_id,a.report_id)
