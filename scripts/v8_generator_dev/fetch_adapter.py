"""Read-only retrieval of the single approved pilot; never creates a GPU app."""
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
        'Producer: scripts.v8_generator_dev.fetch_adapter. root/ retains remote terminal/phase '
        'receipts; manifest.json maps every retrieved file, hash and byte count. Training/evaluation '
        'reports and final adapter checkpoints retain original new runtime-relative names locally. '
        'Only final adapters are fetched; parent checkpoints were already verified. inputs.json '
        'hashes downloaded outputs. complete.json distinguishes retrieval from scientific success.\n',encoding='utf-8')
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote='v8_generator_dev/'+run_id
    def content(name):
        return b''.join(volume.read_file(name))
    entries=volume.listdir(remote,recursive=True)
    available={entry.path.lstrip('/') for entry in entries}
    success=remote+'/complete.json' in available
    terminal=remote+('/complete.json' if success else '/failed.json')
    if terminal not in available:
        raise ValueError('A terminal pilot receipt is required before final retrieval')
    root=json.loads(content(terminal))
    training_relative=f'reports/v8_generator_dev/{run_id}_training'
    trained_path=remote+'/project/'+training_relative+'/complete.json'
    trained=json.loads(content(trained_path)) if trained_path in available else None
    checkpoint_names={copy['checkpoint'].replace('\\','/') for copy in trained['copies'].values()} if trained else set()
    tasks=[]
    for entry in entries:
        name=entry.path.lstrip('/')
        if name.rsplit('/',1)[-1].find('.')<0:
            continue
        remainder=name.removeprefix(remote+'/')
        if '/' not in remainder:
            destination=output/'root'/remainder
        elif remainder.startswith('project/reports/v8_generator_dev/'+run_id+'_'):
            destination=project_path(remainder.removeprefix('project/'))
        elif remainder.removeprefix('project/') in checkpoint_names:
            destination=project_path(remainder.removeprefix('project/'))
        else:
            continue
        tasks.append((name,destination))
    if len(tasks)>4000 or sum(getattr(entry,'size',0) or 0 for entry in entries)>3*1024**3:
        raise ValueError('Unexpected pilot retrieval size; inspect before downloading')
    def retrieve(task):
        name,destination=task
        data=content(name)
        expected=hashlib.sha256(data).hexdigest()
        if destination.exists():
            if digest(destination)!=expected:
                raise ValueError('Refuse to replace existing differing output: '+str(destination))
            reused=True
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(data)
            reused=False
        return dict(remote=name,local=destination.relative_to(ROOT).as_posix(),sha256=expected,
            bytes=len(data),identical_existing_file=reused)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(retrieve,tasks))
    if trained:
        for copy in trained['copies'].values():
            if digest(project_path(copy['checkpoint']))!=copy['sha256']:
                raise ValueError('Final adapter hash mismatch')
    json_write(output/'manifest.json',records)
    json_write(output/'inputs.json',{row['local']:row['sha256'] for row in records})
    json_write(output/'complete.json',dict(complete=True,workflow_success=success,
        training=training_relative,final_adapters_verified=bool(trained),run_id=run_id,
        evaluations=root.get('evaluations',{}),prepared=root.get('prepared'),
        bytes=sum(row['bytes'] for row in records),anatomy_validated=False))
    print('ADAPTER RETRIEVAL COMPLETE',output,'workflow_success',success,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--report-id',required=True)
    a=p.parse_args()
    fetch(a.run_id,a.report_id)
