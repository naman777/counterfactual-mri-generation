"""Read-only future result retrieval; no app creation, retry or overwritten evidence."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path


def fetch(run_id,report_id):
    if not all(x.replace('_','').isalnum() for x in (run_id,report_id)):raise ValueError('Safe new identifiers')
    os.environ['MODAL_PROFILE']='namankundra7'
    import modal
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote='v10_detail/'+run_id
    entries=volume.listdir(remote,recursive=True)
    names={e.path.lstrip('/') for e in entries}
    success=remote+'/complete.json' in names
    terminal=remote+('/complete.json' if success else '/failed.json')
    if terminal not in names:raise ValueError('Terminal receipt required')
    def content(name):return b''.join(volume.read_file(name))
    root=json.loads(content(terminal))
    output=ROOT/'reports/v10_detail'/report_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.fetch. root/: original terminal/phase receipts; '
        'manifest.json: downloaded file/hash/size map; inputs.json: downloaded evidence hashes; complete.json: retrieval '
        'scope and separate workflow success. Full training/evaluation and engineering snapshots retain fresh original paths.\n',encoding='utf-8')
    tasks=[]
    for entry in entries:
        name=entry.path.lstrip('/');relative=name.removeprefix(remote+'/')
        if '.' not in relative.rsplit('/',1)[-1]:continue
        if '/' not in relative:dest=output/'root'/relative
        elif relative.startswith('project/reports/v10_detail/'+run_id+'_') or relative.startswith('project/checkpoints/diffusion/ax_t1_128_v10_detail/'+run_id+'_'):
            dest=project_path(relative.removeprefix('project/'))
        else:continue
        tasks.append((name,dest))
    if len(tasks)>6000 or sum(getattr(e,'size',0) or 0 for e in entries)>3*1024**3:
        raise ValueError('Unexpected scope; inspect before retrieving')
    def retrieve(task):
        name,dest=task;data=content(name);sha=hashlib.sha256(data).hexdigest()
        if dest.exists():
            if digest(dest)!=sha:raise ValueError('Refuse to overwrite prior result')
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(data)
        return dict(remote=name,local=dest.relative_to(ROOT).as_posix(),sha256=sha,bytes=len(data))
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(retrieve,tasks))
    json_write(output/'manifest.json',records);json_write(output/'inputs.json',{r['local']:r['sha256'] for r in records})
    json_write(output/'complete.json',dict(complete=True,workflow_success=success,training=root.get('training'),
        evaluations=root.get('evaluations',{}),prepared=root.get('prepared'),bytes=sum(r['bytes'] for r in records),
        strict_local_regrading_pending=True,clinical_preservation='unknown'))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run-id',required=True);ap.add_argument('--report-id',required=True)
    a=ap.parse_args();fetch(a.run_id,a.report_id)
