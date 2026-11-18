"""One explicitly approved, bounded A10 adapter pilot; no automatic paid restart."""
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

app=modal.App('capstone-v8-source-guide-pilot')
volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
image=modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1','torchvision==0.20.1','numpy==2.1.3','pandas==2.2.3',
    'scipy==1.14.1','scikit-image==0.24.0','pyyaml==6.0.2','pillow',
).env({'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
if modal.is_local():
    prepared=Path(os.environ['CAPSTONE_ADAPTER_PREPARED']).resolve()
    plan=json.loads((prepared/'plan.json').read_text())
    image=image.add_local_file(Path.cwd()/plan['archive'],'/input/inputs.zip',copy=True)
    image=image.add_local_dir(prepared,'/prepared',copy=True)


@app.function(image=image,gpu='A10G',cpu=2,memory=16384,max_containers=1,retries=0,
              scaledown_window=2,timeout=14520,volumes={'/data':volume})
def workflow(approved_usd: float):
    if approved_usd!=8:
        raise ValueError('Separate explicit additional US$8 approval required')
    started=time.monotonic()
    plan=json.loads(Path('/prepared/plan.json').read_text())
    if plan['approved_additional_usd']!=approved_usd:
        raise ValueError('Approved allowance/proposal mismatch')
    deadline=started+min(plan['maximum_worker_seconds'],(approved_usd-plan['reserve_usd'])/plan['price_with_margin'])
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
    training_relative=f'reports/v8_generator_dev/{run_id}_training'
    training=staged/training_relative
    phases,backed=[],set()

    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):
                h.update(block)
        return h.hexdigest()

    def write(path,value):
        with path.open('x',encoding='utf-8') as handle:
            json.dump(value,handle,indent=2,allow_nan=False)

    def persist_file(source,destination):
        if destination.exists():
            if sha(source)!=sha(destination):
                raise ValueError('Refuse to overwrite persisted evidence: '+str(destination))
            return
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,destination.open('xb') as dst:
            shutil.copyfileobj(src,dst)

    def persist_tree(source,destination):
        for path in sorted((p for p in source.rglob('*') if p.is_file()),key=lambda p:(p.name=='complete.json',str(p))):
            persist_file(path,destination/path.relative_to(source))

    def backup():
        for marker in sorted(training.glob('block_*/complete.json')):
            if marker.parent.name in backed:
                continue
            try:
                record=json.loads(marker.read_text())
            except json.JSONDecodeError:
                continue
            if len(record['copies'])!=4 or not record['matched_draws']:
                raise ValueError('All four matched copies required before block publication')
            for copy in record['copies'].values():
                relative=copy['checkpoint'].replace('\\','/')
                source=staged/relative
                if sha(source)!=copy['sha256']:
                    raise ValueError('Adapter checkpoint changed before backup')
                persist_file(source,project/relative)
            persist_tree(marker.parent,project/training_relative/marker.parent.name)
            backed.add(marker.parent.name)
            write(home/f'common_epoch_{record["common_epoch"]:04d}.json',record)
            volume.commit()

    def call(name,module,args,training_phase=False):
        phase_start=time.monotonic()
        command=[sys.executable,'-m',module,*args]
        with (home/(name+'.txt')).open('x',encoding='utf-8') as handle:
            process=subprocess.Popen(command,cwd=staged,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(staged),PYTHONUNBUFFERED='1'))
            try:
                while process.poll() is None:
                    if time.monotonic()>=deadline-45:
                        raise TimeoutError('Separate adapter allowance/worker guard reached')
                    if training_phase:
                        backup()
                        if time.monotonic()-phase_start>=plan['maximum_training_seconds']:
                            raise TimeoutError('Pilot training cap reached; reserve validation time, retain incomplete pilot')
                    volume.commit()
                    time.sleep(5)
                if training_phase:
                    backup()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        record=dict(phase=name,command=command,exit_code=process.returncode,seconds=time.monotonic()-phase_start)
        phases.append(record)
        write(home/(name+'.json'),record)
        volume.commit()
        if process.returncode:
            raise RuntimeError('Pilot phase failed: '+name)

    try:
        if sha(Path('/input/inputs.zip'))!=plan['archive_sha256']:
            raise ValueError('Approved original archive hash changed')
        with zipfile.ZipFile('/input/inputs.zip') as archive:
            manifest=json.loads(archive.read('_payload_manifest.json'))
            if set(archive.namelist())!=set(manifest)|{'_payload_manifest.json'}:
                raise ValueError('Original archive membership changed')
            for name,expected in manifest.items():
                relative=PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe archive path')
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
            if not destination.resolve().is_relative_to(staged) or sha(source)!=expected:
                raise ValueError('Unsafe or changed explicit source patch')
            destination.parent.mkdir(parents=True,exist_ok=True)
            # This worker-local source replacement is explicit and hash-bound;
            # original volume/archive bytes remain unchanged.
            shutil.copyfile(source,destination)
        prepared_relative=plan['prepared_relative']
        runtime_prepared=staged/prepared_relative
        runtime_prepared.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','approval.json','plan.json'):
            shutil.copyfile(Path('/prepared')/name,runtime_prepared/name)
        (runtime_prepared/'README.md').write_text('Exact approved pilot protocol and staged runtime input ledger.\n')
        original=Path('/data')/plan['source_parent_project']
        for backbone,name in plan['parent_paths'].items():
            relative=name.replace('\\','/')
            source=original/relative
            if sha(source)!=plan['parent_hashes'][backbone]:
                raise ValueError('Completed verified parent hash mismatch')
            destination=staged/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as src,destination.open('xb') as dst:
                shutil.copyfileobj(src,dst)
        persist_tree(runtime_prepared,project/prepared_relative)
        write(home/'authorisation.json',dict(separate_additional_approved_usd=8,previous_allowances_reset=False,
            plan=plan,maximum_worker_seconds=deadline-started,exact_invoice_known=False))
        write(home/'staging.json',dict(complete=True,seconds=time.monotonic()-started,
            source_archive_sha256=plan['archive_sha256'],parent_hashes=plan['parent_hashes'],test_pixels_used=False))
        volume.commit()
        # Actual batch-32 CUDA training on final parents; all four updates are discarded.
        call('runtime_smoke','scripts.v8_generator_dev.adapter_pilot',['--prepared',prepared_relative,
            '--run-id',run_id+'_runtime_smoke','--engineering-steps','1'])
        persist_tree(staged/f'reports/v8_generator_dev/{run_id}_runtime_smoke',
                     project/f'reports/v8_generator_dev/{run_id}_runtime_smoke')
        call('train','scripts.v8_generator_dev.adapter_pilot',['--prepared',prepared_relative,
            '--run-id',run_id+'_training'],training_phase=True)
        persist_tree(training,project/training_relative)
        trained=json.loads((training/'complete.json').read_text())
        if trained['engineering'] or trained['epoch']!=20 or len(trained['copies'])!=4:
            raise ValueError('All four full pilot copies at epoch twenty required')
        evaluations={}
        for key,copy in trained['copies'].items():
            controls=('correct','absent','wrong') if key.endswith('__soft_source_guide') else ('correct',)
            for control in controls:
                evaluation_id=run_id+'_validation_'+key+'__'+control
                evaluation_relative='reports/v8_generator_dev/'+evaluation_id
                call('evaluate_'+key+'__'+control,'scripts.v8_generator_dev.evaluate_adapter',
                    ['--prepared',prepared_relative,'--checkpoint',copy['checkpoint'],
                     '--run-id',evaluation_id,'--guide-control',control])
                persist_tree(staged/evaluation_relative,project/evaluation_relative)
                evaluations[key+'__'+control]=evaluation_relative
        write(home/'complete.json',dict(complete=True,phases=phases,seconds=time.monotonic()-started,
            approved_additional_usd=8,training=training_relative,evaluations=evaluations,
            prepared=prepared_relative,all_four_copies_epoch20=True,anatomy_validated=False,
            efficacy_established=False,fresh_test_evaluated=False))
    except Exception as error:
        backup_error=None
        try:
            if training.exists():
                backup()
                persist_tree(training,project/training_relative)
            for folder in (staged/'reports/v8_generator_dev').glob(run_id+'_validation_*'):
                persist_tree(folder,project/'reports/v8_generator_dev'/folder.name)
        except Exception as persistence_error:
            backup_error=str(persistence_error)
        write(home/'failed.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            phases=phases,seconds=time.monotonic()-started,backed_up_blocks=sorted(backed),
            backup_error=backup_error,anatomy_validated=False,automatic_paid_restart=False))
        raise
    finally:
        volume.commit()
    return str(home)


@app.local_entrypoint()
def main(approved_usd: float=0):
    print(workflow.remote(approved_usd),flush=True)
