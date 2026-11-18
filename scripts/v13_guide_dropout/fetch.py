"""Read-only terminal/partial retrieval into append-only local paths."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path


def fetch(run_id,report_id,allow_running=False):
    if not all(x.replace('_','').isalnum() for x in (run_id,report_id)):raise ValueError('Safe identifiers required')
    os.environ['MODAL_PROFILE']='namankundra7'
    import modal
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote='v13_guide_dropout/'+run_id;entries=volume.listdir(remote,recursive=True)
    names={e.path.lstrip('/') for e in entries}
    terminals=[name for name in ('complete.json','stopped_for_budget.json','failed.json') if remote+'/'+name in names]
    if len(terminals)>1 or (not terminals and not allow_running):raise ValueError('One terminal receipt required')
    def content(name):return b''.join(volume.read_file(name))
    terminal=json.loads(content(remote+'/'+terminals[0])) if terminals else dict(workflow_success=False)
    out=ROOT/'reports/v13_guide_dropout'/report_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.fetch. root/: terminal and phase receipts; '
        'manifest.json/inputs.json: every local byte/hash; complete.json: retrieval success distinct from experiment success. '
        'Only finished immutable training segments are retrieved while running. No app creation/retry.\n',encoding='utf-8')
    if len(entries)>12000 or sum(getattr(e,'size',0) or 0 for e in entries)>3*1024**3:
        raise ValueError('Unexpected retrieval scope')
    # A running snapshot includes only immutable segment markers/checkpoints; mutable logs are omitted.
    immutable=set()
    if not terminals:
        for name in names:
            if '/segments/' in name and name.endswith('/complete.json'):
                record=json.loads(content(name));immutable.add(remote+'/project/'+record['checkpoint'])
                immutable.add(name);immutable.add(name.removesuffix('complete.json')+'updates.jsonl')
    tasks=[]
    for entry in entries:
        name=entry.path.lstrip('/');relative=name.removeprefix(remote+'/')
        if '.' not in relative.rsplit('/',1)[-1]:continue
        if not terminals and name not in immutable:continue
        if '/' not in relative:dest=out/'root'/relative
        elif relative.startswith('project/reports/v13_guide_dropout/'+run_id+'_') or relative.startswith('project/checkpoints/diffusion/ax_t1_128_v13_guide_dropout/'+run_id+'_'):
            dest=project_path(relative.removeprefix('project/'))
        else:continue
        tasks.append((name,dest))
    def retrieve(task):
        name,dest=task;data=content(name);sha=hashlib.sha256(data).hexdigest()
        if dest.exists():
            if digest(dest)!=sha:raise ValueError('Existing local artifact differs')
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(data)
        return dict(remote=name,local=dest.relative_to(ROOT).as_posix(),sha256=sha,bytes=len(data))
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(retrieve,tasks))
    json_write(out/'manifest.json',records);json_write(out/'inputs.json',{r['local']:r['sha256'] for r in records})
    json_write(out/'complete.json',dict(complete=True,workflow_success=bool(terminal.get('workflow_success',False)),
        running_snapshot=not bool(terminals),terminal=terminals[0] if terminals else None,
        training=terminal.get('training'),evaluations=terminal.get('evaluations',{}),prepared=terminal.get('prepared'),
        bytes=sum(r['bytes'] for r in records),strict_local_regrading_pending=True))
    print('READ ONLY RETRIEVAL COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run-id',required=True);ap.add_argument('--report-id',required=True)
    ap.add_argument('--allow-running',action='store_true');fetch(**vars(ap.parse_args()))
