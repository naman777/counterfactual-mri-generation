"""Retrieve completed training/validation after the cloud's analysis-import failure."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

os.environ['MODAL_PROFILE']='namankundra7'
import modal

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def recover(run_id,report_id):
    if not all(value.replace('_','').isalnum() for value in (run_id,report_id)):
        raise ValueError('Safe new recovery identifiers required')
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote='v8_generator_dev/'+run_id
    def read(path):
        return b''.join(volume.read_file(path))
    failed=json.loads(read(remote+'/failed.json'))
    phases={row['phase']:row['exit_code'] for row in failed['phases']}
    analysis_log=read(remote+'/analyse.txt').decode()
    if any(phases.get(phase)!=0 for phase in ('train','evaluate_additive','evaluate_scale_shift')):
        raise ValueError('Completed training and both validation phases are required')
    if phases.get('analyse')!=1 or 'No module named scripts.v8_generator_dev.analyse_full' not in analysis_log:
        raise ValueError('Recovery is limited to the retained analysis-import packaging failure')
    trained=json.loads(read(remote+f'/project/reports/v8_generator_dev/{run_id}_training/complete.json'))
    if trained['engineering'] or trained['progress']['additive']!=trained['progress']['scale_shift']:
        raise ValueError('Matched non-engineering final training receipt required')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.recover_continuation_results. root/ preserves the '
        'original cloud failure and all completed phase receipts/logs. training/ and protocol/ '
        'retain frozen completed training/protocol bytes; validation_*/ retains every saved '
        'array/request/grade. Final primary checkpoints retain original new paths. manifest.json '
        'maps bytes/hashes; inputs.json binds retrieval. complete.json keeps workflow_success '
        'false while recording separately verified training/validation completion. The missing '
        'cloud summary must be recomputed locally; no GPU restart or failure erasure occurs.\n',encoding='utf-8')
    entries=volume.listdir(remote,recursive=True)
    tasks=[]
    ckpaths={path.replace('\\','/') for path in trained['checkpoints'].values()}
    for entry in entries:
        name=entry.path.lstrip('/')
        if '.' not in name.rsplit('/',1)[-1]:
            continue
        relative=name.removeprefix(remote+'/')
        if '/' not in relative:
            destination=output/'root'/relative
        elif relative.startswith(f'project/reports/v8_generator_dev/{run_id}_training/'):
            tail=relative.split(run_id+'_training/',1)[1]
            if tail.endswith('_steps.json'):
                continue
            destination=output/'training'/tail
        elif relative.startswith(f'project/reports/v8_generator_dev/{run_id}_protocol/'):
            tail=relative.split(run_id+'_protocol/',1)[1]
            destination=output/'protocol'/(run_id+'_protocol')/tail
        elif any(relative.startswith(f'project/reports/v8_generator_dev/{run_id}_validation_{variant}/') for variant in ('additive','scale_shift')):
            variant=next(v for v in ('additive','scale_shift') if relative.startswith(f'project/reports/v8_generator_dev/{run_id}_validation_{v}/'))
            tail=relative.split(run_id+'_validation_'+variant+'/',1)[1]
            destination=output/('validation_'+variant)/(run_id+'_validation_'+variant)/tail
        elif relative.removeprefix('project/') in ckpaths:
            destination=project_path(relative.removeprefix('project/'))
        else:
            continue
        tasks.append((name,destination))
    def retrieve(task):
        name,destination=task
        data=read(name)
        sha=hashlib.sha256(data).hexdigest()
        if destination.exists():
            if digest(destination)!=sha:
                raise ValueError('Refuse to replace differing existing output')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(data)
        return dict(remote=name,local=destination.relative_to(ROOT).as_posix(),bytes=len(data),sha256=sha)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(retrieve,tasks))
    for variant,path in trained['checkpoints'].items():
        if digest(project_path(path))!=trained['checkpoint_hashes'][variant]:
            raise ValueError('Recovered final checkpoint hash mismatch')
    homes={variant:str((output/('validation_'+variant)/(run_id+'_validation_'+variant)).relative_to(ROOT))
        for variant in ('additive','scale_shift')}
    for home in homes.values():
        complete=json.loads((project_path(home)/'complete.json').read_text())
        if not complete['complete'] or complete['engineering'] or complete['patients']!=15:
            raise ValueError('Incomplete/nonprimary recovered validation')
    json_write(output/'manifest.json',records)
    json_write(output/'inputs.json',{row['local']:row['sha256'] for row in records})
    json_write(output/'complete.json',dict(complete=True,workflow_success=False,
        cloud_analysis_import_failed=True,training_and_validation_success=True,
        primary_checkpoint_hashes_verified=True,full_training_completed=True,
        training_folder=str((output/'training').relative_to(ROOT)),evaluation_folders=homes,
        local_analysis_still_required=True,anatomy_validated=False,new_gpu_app_created=False))
    print('CONTINUATION COMPONENTS RECOVERED',output,'cloud_analysis_failure_retained',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--report-id',required=True)
    a=p.parse_args()
    recover(a.run_id,a.report_id)
