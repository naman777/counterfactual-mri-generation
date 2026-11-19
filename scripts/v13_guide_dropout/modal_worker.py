"""One bounded A10 app; persist training before spending the remainder on evaluation."""
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

app=modal.App('capstone-v13-guide-dropout')
volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1','torchvision==0.20.1','numpy==2.1.3','pandas==2.2.3',
    'scipy==1.14.1','scikit-image==0.24.0','pyyaml==6.0.2','pillow==11.0.0',
).env({'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
if modal.is_local():
    prepared=Path(os.environ['CAPSTONE_DROPOUT_PREPARED']).resolve()
    plan=json.loads((prepared/'plan.json').read_text())
    image=image.add_local_file(Path.cwd()/plan['archive'],'/input/inputs.zip',copy=True)
    image=image.add_local_dir(prepared,'/prepared',copy=True)


@app.function(image=image,gpu='A10G',cpu=2,memory=16384,max_containers=1,retries=0,
              scaledown_window=2,timeout=23580,volumes={'/data':volume})
def workflow(working_limit_usd:float):
    if working_limit_usd!=12:raise ValueError('Recorded remaining-credit working limit required')
    plan=json.loads(Path('/prepared/plan.json').read_text())
    if plan['working_limit_usd']!=12 or plan['total_remaining_credit_ceiling_usd']!=13:
        raise ValueError('Budget identity changed')
    started=time.monotonic()
    deadline=started+min(plan['maximum_worker_seconds'],(12-plan['reserve_usd'])/plan['price_with_margin'])
    run_id=plan['run_id']
    if not run_id.replace('_','').isalnum():raise ValueError('Safe run ID')
    volume.reload();home=Path('/data/v13_guide_dropout')/run_id;home.mkdir(parents=True,exist_ok=False)
    project=home/'project';project.mkdir()
    staged=Path('/tmp')/run_id;staged.mkdir(exist_ok=False)
    training_relative=f'reports/v13_guide_dropout/{run_id}_training';training=staged/training_relative
    phases=[];published=set();evaluations={}

    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()

    def write(path,value):
        with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

    def copy(source,dest):
        if dest.exists():
            if sha(source)!=sha(dest):raise ValueError('Immutable result collision: '+str(dest))
            return
        dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,dest.open('xb') as b:shutil.copyfileobj(a,b)

    def tree(source,dest):
        if not source.exists():return
        for f in sorted((p for p in source.rglob('*') if p.is_file()),key=lambda p:(p.name=='complete.json',str(p))):
            copy(f,dest/f.relative_to(source))

    def backup():
        for marker in sorted(training.glob('segments/*/complete.json')):
            identity=marker.parent.name
            if identity in published:continue
            record=json.loads(marker.read_text());relative=record['checkpoint']
            if record['engineering'] or record['protocol_sha256']!=plan['protocol_sha256']:
                raise ValueError('Segment identity mismatch')
            if sha(staged/relative)!=record['sha256']:raise ValueError('Segment checkpoint changed')
            copy(staged/relative,project/relative)
            tree(marker.parent,project/training_relative/'segments'/identity)
            published.add(identity)
        for marker in sorted(training.glob('block_*/complete.json')):
            dest=project/training_relative/marker.parent.name/'complete.json'
            if dest.exists():continue
            record=json.loads(marker.read_text())
            if len(record['copies'])!=6 or not record['matched_draws_within_training_seed']:
                raise ValueError('Six matched common-block copies required')
            copy(marker,dest)
            write(home/f'common_epoch_{record["common_epoch"]:04d}.json',record)
        volume.commit()

    def call(name,module,args,cap=None,training_phase=False):
        beginning=time.monotonic()
        limit=min(deadline-90,beginning+cap if cap else deadline-90)
        if limit-beginning<30:raise TimeoutError('Budget reserve reached before '+name)
        command=[sys.executable,'-u','-m',module,*args]
        with (home/(name+'.txt')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=staged,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(staged),PYTHONUNBUFFERED='1'))
            try:
                while child.poll() is None:
                    if time.monotonic()>=limit:raise TimeoutError('Budget/time guard reached: '+name)
                    if training_phase:backup()
                    else:volume.commit()
                    time.sleep(5)
                if training_phase:backup()
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        record=dict(phase=name,exit_code=child.returncode,seconds=time.monotonic()-beginning,command=command)
        phases.append(record);write(home/(name+'.json'),record);volume.commit()
        if child.returncode:raise RuntimeError('Phase failed: '+name)

    def persist():
        backup();tree(training,project/training_relative)
        ckdir=staged/'checkpoints/diffusion/ax_t1_128_v13_guide_dropout'/f'{run_id}_training'
        tree(ckdir,project/ckdir.relative_to(staged))
        parent=staged/'reports/v13_guide_dropout'
        if parent.exists():
            for folder in parent.glob(run_id+'_*'):tree(folder,project/'reports/v13_guide_dropout'/folder.name)

    try:
        write(home/'authorisation.json',dict(working_limit_usd=12,total_ceiling_usd=13,plan=plan,
            automatic_paid_retry=False,previous_allowances_reset=False))
        volume.commit()
        if sha(Path('/input/inputs.zip'))!=plan['archive_sha256']:raise ValueError('Archive changed')
        with zipfile.ZipFile('/input/inputs.zip') as z:
            manifest=json.loads(z.read('_payload_manifest.json'))
            if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(manifest)|{'_payload_manifest.json'}:
                raise ValueError('Archive membership changed')
            for name,expected in manifest.items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:raise ValueError('Unsafe member')
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Input bytes changed')
                target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        for name,expected in plan['patches'].items():
            source=Path('/prepared/code')/name
            if sha(source)!=expected:raise ValueError('Overlay changed')
            target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        parent=Path('/data')/plan['source_parent_project']/plan['parent']
        if sha(parent)!=plan['parent_sha256']:raise ValueError('Final parent changed')
        copy(parent,staged/plan['parent'])
        rel=plan['prepared_relative'];runtime=staged/rel;runtime.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','mask_thresholds.json','plan.json'):
            copy(Path('/prepared')/name,runtime/name)
        write(runtime/'authorisation.json',dict(working_limit_usd=12,protocol_sha256=plan['protocol_sha256']))
        tree(runtime,project/rel);volume.commit()
        call('full_batch_gate','scripts.v13_guide_dropout.runtime_smoke',
            ['--prepared',rel,'--run-id',run_id+'_runtime_smoke'],plan['maximum_full_batch_gate_seconds'])
        smoke=staged/f'reports/v13_guide_dropout/{run_id}_runtime_smoke'
        tree(smoke,project/smoke.relative_to(staged));gate=json.loads((smoke/'complete.json').read_text())
        if gate['copies']!=6 or not gate['full_batch_backward'] or not gate['all_states_exercised']:
            raise ValueError('Intended-runtime full-batch gate failed')
        call('train','scripts.v13_guide_dropout.train',['--prepared',rel,'--run-id',run_id+'_training'],
             plan['maximum_training_seconds'],True)
        persist();trained=json.loads((training/'complete.json').read_text())
        if trained['engineering'] or trained['common_epoch']!=20 or len(trained['copies'])!=6:
            raise ValueError('Full matched exposure incomplete')
        protocol=json.loads((runtime/'protocol.json').read_text())
        for key,ck in trained['copies'].items():
            for axis,arms in protocol['reference_protocol']['controls'].items():
                for control in arms[ck['mode']]:
                    identity=key+'__'+axis+'__'+control;evaluation_id=run_id+'_eval_'+identity
                    relative=f'reports/v13_guide_dropout/{evaluation_id}'
                    call('eval_'+identity,'scripts.v13_guide_dropout.evaluate',['--prepared',rel,'--checkpoint',ck['checkpoint'],
                        '--run-id',evaluation_id,'--axis',axis,'--guide-control',control])
                    tree(staged/relative,project/relative);evaluations[identity]=relative;volume.commit()
        if len(evaluations)!=42:raise ValueError('Declared matrix incomplete')
        write(home/'complete.json',dict(complete=True,workflow_success=True,phases=phases,training=training_relative,
            evaluations=evaluations,prepared=rel,seconds=time.monotonic()-started,working_limit_usd=12,
            strict_local_regrading_pending=True,clinical_preservation='unknown'))
    except Exception as error:
        try:persist();persistence_error=None
        except Exception as secondary:persistence_error=str(secondary)
        marker='stopped_for_budget.json' if isinstance(error,TimeoutError) else 'failed.json'
        write(home/marker,dict(complete=False,workflow_success=False,error_type=type(error).__name__,error=str(error),
            phases=phases,training=training_relative,evaluations=evaluations,prepared=plan['prepared_relative'],
            seconds=time.monotonic()-started,persistence_error=persistence_error,automatic_paid_retry=False,
            local_continuation_required=True))
        if not isinstance(error,TimeoutError):raise
    finally:volume.commit()
    return str(home)


@app.local_entrypoint()
def main(working_limit_usd:float=0):
    print(workflow.remote(working_limit_usd),flush=True)
