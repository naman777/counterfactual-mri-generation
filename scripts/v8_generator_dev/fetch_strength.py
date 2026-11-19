"""Read-only retrieval for an inference-only sweep; no fictional training receipt."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

os.environ['MODAL_PROFILE']='namankundra7'
import modal

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def fetch(run_id,report_id):
    if not all(value.replace('_','').isalnum() for value in (run_id,report_id)):
        raise ValueError('Safe new retrieval identifiers required')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.fetch_strength. root/ retains original phase/terminal '
        'receipts; manifest.json maps every retrieved file/hash. Six evaluation reports retain '
        'their original new runtime-relative names. inputs.json binds all downloaded bytes. '
        'complete.json distinguishes retrieval/workflow completion and records no new training '
        'or weights. Original pilot references remain separate. No GPU app is created.\n',encoding='utf-8')
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote='v8_generator_dev/'+run_id
    def content(path):
        return b''.join(volume.read_file(path))
    entries=volume.listdir(remote,recursive=True)
    available={entry.path.lstrip('/') for entry in entries}
    success=remote+'/complete.json' in available
    terminal=remote+('/complete.json' if success else '/failed.json')
    if terminal not in available:
        raise ValueError('Terminal sweep receipt required')
    root=json.loads(content(terminal))
    if root['training_ran']:
        raise ValueError('This retrieval contract permits inference-only results')
    tasks=[]
    for entry in entries:
        name=entry.path.lstrip('/')
        if '.' not in name.rsplit('/',1)[-1]:
            continue
        relative=name.removeprefix(remote+'/')
        if '/' not in relative:
            destination=output/'root'/relative
        elif relative.startswith('project/reports/v8_generator_dev/'+run_id+'_validation_'):
            destination=project_path(relative.removeprefix('project/'))
        else:
            continue
        tasks.append((name,destination))
    if len(tasks)>3000:
        raise ValueError('Unexpected sweep retrieval scope')
    def retrieve(task):
        remote_path,destination=task
        data=content(remote_path)
        expected=hashlib.sha256(data).hexdigest()
        if destination.exists():
            if digest(destination)!=expected:
                raise ValueError('Refuse to replace differing existing sweep evidence')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(data)
        return dict(remote=remote_path,local=destination.relative_to(ROOT).as_posix(),sha256=expected,bytes=len(data))
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(retrieve,tasks))
    if success and len(root['evaluations'])!=6:
        raise ValueError('Six prospectively declared strength evaluations required')
    json_write(output/'manifest.json',records)
    json_write(output/'inputs.json',{row['local']:row['sha256'] for row in records})
    json_write(output/'complete.json',dict(complete=True,workflow_success=success,
        evaluations=root.get('evaluations',{}),prepared=root.get('prepared'),training_ran=False,
        new_checkpoint_written=False,anatomy_validated=False,run_id=run_id))
    print('STRENGTH RETRIEVAL COMPLETE',output,'workflow_success',success,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--report-id',required=True)
    a=p.parse_args()
    fetch(a.run_id,a.report_id)
