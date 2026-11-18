"""Prospective one-A10 validation-only strength sweep; explicit separate allowance."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import modal

app=modal.App('capstone-v8-source-strength-validation')
volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1','torchvision==0.20.1','numpy==2.1.3','pandas==2.2.3',
    'scipy==1.14.1','scikit-image==0.24.0','pyyaml==6.0.2','pillow',
).env({'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
if modal.is_local():
    prepared=Path(os.environ['CAPSTONE_STRENGTH_PREPARED']).resolve()
    plan=json.loads((prepared/'plan.json').read_text())
    image=image.add_local_file(Path.cwd()/plan['archive'],'/input/inputs.zip',copy=True)
    image=image.add_local_dir(prepared,'/prepared',copy=True)


@app.function(image=image,gpu='A10G',cpu=2,memory=16384,max_containers=1,retries=0,
              scaledown_window=2,timeout=4620,volumes={'/data':volume})
def workflow(approved_usd: float):
    if approved_usd!=3:
        raise ValueError('Explicit separate additional US$3 approval required')
    start=time.monotonic()
    plan=json.loads(Path('/prepared/plan.json').read_text())
    deadline=start+min(4500,(approved_usd-.5)/plan['rate_with_margin'])
    run_id=plan['run_id']
    if not run_id.replace('_','').isalnum():
        raise ValueError('Safe new run ID required')
    volume.reload()
    home=Path('/data/v8_generator_dev')/run_id
    home.mkdir(exist_ok=False)
    project=home/'project'
    project.mkdir()
    staged=Path('/tmp')/run_id
    staged.mkdir(exist_ok=False)
    phases,evaluations=[],{}

    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):
                h.update(block)
        return h.hexdigest()

    def write(path,value):
        with path.open('x',encoding='utf-8') as handle:
            json.dump(value,handle,indent=2,allow_nan=False)

    def persist_tree(source,destination):
        for path in sorted((p for p in source.rglob('*') if p.is_file()),key=lambda p:(p.name=='complete.json',str(p))):
            target=destination/path.relative_to(source)
            if target.exists():
                if sha(target)!=sha(path):
                    raise ValueError('Refuse to overwrite persisted strength evidence')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst)

    def call(name,arguments):
        command=[sys.executable,'-m','scripts.v8_generator_dev.evaluate_strength',*arguments]
        phase_start=time.monotonic()
        with (home/(name+'.txt')).open('x',encoding='utf-8') as handle:
            process=subprocess.Popen(command,cwd=staged,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(staged),PYTHONUNBUFFERED='1'))
            try:
                while process.poll() is None:
                    if time.monotonic()>=deadline-30:
                        raise TimeoutError('Strength validation allowance/time guard reached')
                    volume.commit()
                    time.sleep(5)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                raise
        record=dict(phase=name,command=command,exit_code=process.returncode,seconds=time.monotonic()-phase_start)
        phases.append(record)
        write(home/(name+'.json'),record)
        volume.commit()
        if process.returncode:
            raise RuntimeError('Strength phase failed: '+name)

    try:
        if sha(Path('/input/inputs.zip'))!=plan['archive_sha256']:
            raise ValueError('Original archive hash changed')
        with zipfile.ZipFile('/input/inputs.zip') as archive:
            manifest=json.loads(archive.read('_payload_manifest.json'))
            if set(archive.namelist())!=set(manifest)|{'_payload_manifest.json'}:
                raise ValueError('Original archive membership changed')
            for name,expected in manifest.items():
                relative=PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive member')
                content=archive.read(name)
                if hashlib.sha256(content).hexdigest()!=expected:
                    raise ValueError('Original archive content changed')
                destination=staged/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as handle:
                    handle.write(content)
        for name,expected in plan['patches'].items():
            source=Path('/prepared/code')/name
            destination=staged/name
            if sha(source)!=expected or not destination.resolve().is_relative_to(staged):
                raise ValueError('Unsafe or changed explicit code patch')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
        for name,source_info in plan['remote_weights'].items():
            source=Path('/data')/source_info['project']/name
            if sha(source)!=source_info['sha256']:
                raise ValueError('Frozen learned weights changed')
            destination=staged/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as src,destination.open('xb') as dst:
                shutil.copyfileobj(src,dst)
        protocol_home=staged/plan['prepared_relative']
        protocol_home.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','plan.json'):
            shutil.copyfile(Path('/prepared')/name,protocol_home/name)
        (protocol_home/'README.md').write_text('Exact prospective strength protocol/input ledger; no training.\n')
        persist_tree(protocol_home,project/plan['prepared_relative'])
        write(home/'authorisation.json',dict(separate_additional_approved_usd=3,
            previous_allowances_reset=False,plan=plan,worker_deadline_seconds=deadline-start,training=False))
        write(home/'staging.json',dict(complete=True,seconds=time.monotonic()-start,
            frozen_weight_hashes={name:info['sha256'] for name,info in plan['remote_weights'].items()}))
        volume.commit()
        for backbone,checkpoint in plan['adapter_checkpoints'].items():
            for alpha in plan['strengths']:
                key=backbone+'_alpha'+str(int(alpha*100)).zfill(3)
                evaluation_id=run_id+'_validation_'+key
                relative='reports/v8_generator_dev/'+evaluation_id
                call('evaluate_'+key,['--prepared',plan['prepared_relative'],'--checkpoint',checkpoint,
                    '--run-id',evaluation_id,'--strength',str(alpha)])
                persist_tree(staged/relative,project/relative)
                evaluations[key]=relative
                volume.commit()
        write(home/'complete.json',dict(complete=True,phases=phases,seconds=time.monotonic()-start,
            approved_additional_usd=3,evaluations=evaluations,prepared=plan['prepared_relative'],
            training_ran=False,new_checkpoint_written=False,anatomy_validated=False,fresh_test=False))
    except Exception as error:
        for folder in (staged/'reports/v8_generator_dev').glob(run_id+'_validation_*'):
            persist_tree(folder,project/'reports/v8_generator_dev'/folder.name)
        write(home/'failed.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            phases=phases,seconds=time.monotonic()-start,training_ran=False,automatic_paid_restart=False))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main(approved_usd: float=0):
    print(workflow.remote(approved_usd),flush=True)
