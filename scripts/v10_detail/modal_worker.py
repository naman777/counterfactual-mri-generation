"""One prospective A10 app: full-batch gate, nine adapters, complete frozen evaluation."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path,PurePosixPath

import modal

app=modal.App('capstone-v10-replicated-detail')
volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1','torchvision==0.20.1','numpy==2.1.3','pandas==2.2.3',
    'scipy==1.14.1','scikit-image==0.24.0','pyyaml==6.0.2','pillow==11.0.0',
).env({'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
if modal.is_local():
    prepared=Path(os.environ['CAPSTONE_DETAIL_PREPARED']).resolve()
    plan=json.loads((prepared/'plan.json').read_text())
    image=image.add_local_file(Path.cwd()/plan['archive'],'/input/inputs.zip',copy=True)
    image=image.add_local_dir(prepared,'/prepared',copy=True)


@app.function(image=image,gpu='A10G',cpu=2,memory=16384,max_containers=1,retries=0,
              scaledown_window=2,timeout=23580,volumes={'/data':volume})
def workflow(approved_usd:float):
    if approved_usd!=12:raise ValueError('Explicit new US$12 allowance required')
    plan=json.loads(Path('/prepared/plan.json').read_text())
    started=time.monotonic()
    deadline=started+min(plan['maximum_worker_seconds'],(approved_usd-plan['reserve_usd'])/plan['price_with_margin'])
    run_id=plan['run_id']
    if not run_id.replace('_','').isalnum():raise ValueError('Safe new run ID')
    volume.reload();home=Path('/data/v10_detail')/run_id;home.mkdir(parents=True,exist_ok=False)
    staged=Path('/tmp')/run_id;staged.mkdir(exist_ok=False)
    project=home/'project';project.mkdir()
    phases=[];backed=set();training_relative=f'reports/v10_detail/{run_id}_training'
    training=staged/training_relative
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    def write(path,value):
        with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    def copy(source,destination):
        if destination.exists():
            if sha(source)!=sha(destination):raise ValueError('Immutable result collision: '+str(destination))
            return
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,destination.open('xb') as b:shutil.copyfileobj(a,b)
    def tree(source,dest):
        if not source.exists():return
        for f in sorted((p for p in source.rglob('*') if p.is_file()),key=lambda p:(p.name=='complete.json',str(p))):
            copy(f,dest/f.relative_to(source))
    def backup():
        for marker in sorted(training.glob('block_*/complete.json')):
            if marker.parent.name in backed:continue
            try:record=json.loads(marker.read_text())
            except json.JSONDecodeError:continue
            if len(record['copies'])!=9 or not record['matched_draws_within_training_seed']:
                raise ValueError('Nine complete matched copies required for common-block publication')
            for ck in record['copies'].values():
                relative=ck['checkpoint'].replace('\\','/')
                if sha(staged/relative)!=ck['sha256']:raise ValueError('Checkpoint hash drift')
                copy(staged/relative,project/relative)
            tree(marker.parent,project/training_relative/marker.parent.name)
            backed.add(marker.parent.name);write(home/f'common_epoch_{record["common_epoch"]:04d}.json',record);volume.commit()
    def call(name,module,args,cap=None,train_phase=False):
        beginning=time.monotonic();limit=min(deadline-60,beginning+cap if cap else deadline-60)
        command=[sys.executable,'-m',module,*args]
        with (home/(name+'.txt')).open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(command,cwd=staged,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(staged),PYTHONUNBUFFERED='1'))
            try:
                while proc.poll() is None:
                    if time.monotonic()>=limit:raise TimeoutError('New allowance/time guard reached: '+name)
                    if train_phase:backup()
                    volume.commit();time.sleep(5)
                if train_phase:backup()
            except BaseException:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        record=dict(phase=name,exit_code=proc.returncode,seconds=time.monotonic()-beginning,command=command)
        phases.append(record);write(home/(name+'.json'),record);volume.commit()
        if proc.returncode:raise RuntimeError('Phase failed: '+name)
    try:
        if sha(Path('/input/inputs.zip'))!=plan['archive_sha256']:raise ValueError('Original archive hash drift')
        with zipfile.ZipFile('/input/inputs.zip') as z:
            manifest=json.loads(z.read('_payload_manifest.json'))
            if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(manifest)|{'_payload_manifest.json'}:
                raise ValueError('Archive membership drift')
            for name,expected in manifest.items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:raise ValueError('Unsafe member')
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive content drift')
                target=staged/name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(data)
        for name,expected in plan['patches'].items():
            source=Path('/prepared/code')/name
            if sha(source)!=expected:raise ValueError('Frozen source/input overlay drift')
            dest=staged/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        source_parent=Path('/data')/plan['source_parent_project']/plan['parent']
        if sha(source_parent)!=plan['parent_sha256']:raise ValueError('Final parent drift')
        copy(source_parent,staged/plan['parent'])
        rel=plan['prepared_relative'];runtime_prepared=staged/rel;runtime_prepared.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','mask_thresholds.json','plan.json'):
            copy(Path('/prepared')/name,runtime_prepared/name)
        write(runtime_prepared/'authorisation.json',dict(approved_additional_usd=12,
            protocol_sha256=plan['protocol_sha256'],previous_allowances_reset=False))
        tree(runtime_prepared,project/rel)
        write(home/'authorisation.json',dict(approved_additional_usd=12,plan=plan,
            previous_allowances_reset=False,automatic_paid_restart=False))
        volume.commit()
        call('full_batch_gate','scripts.v10_detail.runtime_smoke',['--prepared',rel,'--run-id',run_id+'_runtime_smoke'],cap=900)
        smoke=staged/f'reports/v10_detail/{run_id}_runtime_smoke';tree(smoke,project/smoke.relative_to(staged))
        checked=json.loads((smoke/'complete.json').read_text())
        if checked['copies']!=9 or checked['local_single'] or not checked['full_batch_backward']:
            raise ValueError('Full target-runtime nine-copy backward gate required')
        call('train','scripts.v10_detail.train',['--prepared',rel,'--run-id',run_id+'_training'],
             cap=plan['maximum_training_seconds'],train_phase=True)
        tree(training,project/training_relative)
        trained=json.loads((training/'complete.json').read_text())
        if trained['engineering'] or trained['common_epoch']!=20 or len(trained['copies'])!=9:
            raise ValueError('Matched final exposure incomplete')
        protocol=json.loads((runtime_prepared/'protocol.json').read_text());evaluations={}
        for key,ck in trained['copies'].items():
            for axis,controls in protocol['controls'].items():
                for control in controls[ck['mode']]:
                    identity=key+'__'+axis+'__'+control;evaluation_id=run_id+'_eval_'+identity
                    relative=f'reports/v10_detail/{evaluation_id}'
                    call('eval_'+identity,'scripts.v10_detail.evaluate',['--prepared',rel,'--checkpoint',ck['checkpoint'],
                        '--run-id',evaluation_id,'--axis',axis,'--guide-control',control])
                    tree(staged/relative,project/relative);evaluations[identity]=relative
        write(home/'complete.json',dict(complete=True,workflow_success=True,phases=phases,
            training=training_relative,evaluations=evaluations,prepared=rel,seconds=time.monotonic()-started,
            approved_additional_usd=12,clinical_preservation='unknown',fresh_test=False,
            strict_local_regrading_pending=True))
    except Exception as error:
        try:
            backup();tree(training,project/training_relative)
            ckdir=staged/'checkpoints/diffusion/ax_t1_128_v10_detail'/f'{run_id}_training'
            tree(ckdir,project/ckdir.relative_to(staged))
            for folder in (staged/'reports/v10_detail').glob(run_id+'_*'):tree(folder,project/'reports/v10_detail'/folder.name)
            persistence_error=None
        except Exception as secondary:persistence_error=str(secondary)
        write(home/'failed.json',dict(complete=False,error_type=type(error).__name__,error=str(error),phases=phases,
            seconds=time.monotonic()-started,persistence_error=persistence_error,automatic_paid_restart=False))
        raise
    finally:volume.commit()
    return str(home)


@app.local_entrypoint()
def main(approved_usd:float=0):print(workflow.remote(approved_usd),flush=True)
