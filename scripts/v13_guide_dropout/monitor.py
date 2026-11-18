"""Read-only compact durable progress snapshots."""
import argparse
import json
import os
import time
from datetime import datetime,timezone,timedelta

from scripts.v8_generator_dev.io import ROOT,json_write


def snapshot(run_id,report_id):
    if not all(x.replace('_','').isalnum() for x in (run_id,report_id)):raise ValueError('Safe identifiers')
    os.environ['MODAL_PROFILE']='namankundra7'
    import modal
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=True)
    if not (out/'README.md').exists():
        with (out/'README.md').open('x') as f:
            f.write('Producer: scripts.v13_guide_dropout.monitor. snapshot_*.json: append-only '
                    'phase/common-epoch/segment/terminal observations in IST. No GPU invocation or paid restart.\n')
    record=dict(run_id=run_id,time_ist=datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat())
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False);home='v13_guide_dropout/'+run_id
    def read(path):return b''.join(volume.read_file(path))
    try:
        entries=volume.listdir(home);names={e.path.rsplit('/',1)[-1] for e in entries}
        terminals=[n for n in ('complete.json','stopped_for_budget.json','failed.json') if n in names]
        if terminals:
            record.update(state=terminals[0],terminal=json.loads(read(home+'/'+terminals[0])))
        else:
            done=[n[:-5] for n in names if n.endswith('.json') and n!='authorisation.json' and not n.startswith('common_epoch')]
            running=sorted(n for n in names if n.endswith('.txt') and n[:-4] not in done)
            record.update(state='running',completed_phases=done,active_phases=running,
                common_epochs=sorted(n for n in names if n.startswith('common_epoch_')))
            for name in running:
                record.setdefault('tails',{})[name]=read(home+'/'+name).decode(errors='replace').splitlines()[-3:]
    except modal.exception.NotFoundError:record['state']='awaiting_worker_staging'
    json_write(out/f'snapshot_{time.time_ns()}.json',record)
    print(json.dumps(record,allow_nan=False),flush=True)
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run-id',required=True);ap.add_argument('--report-id',required=True)
    snapshot(**vars(ap.parse_args()))
