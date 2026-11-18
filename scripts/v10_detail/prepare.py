"""Freeze a concrete unapproved package and verify its exact worker-local composition."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import PurePosixPath

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v10_detail.protocol import make_protocol,canonical_hash,validate


def prepare(report_id):
    if not report_id.replace('_','').isalnum():raise ValueError('Safe fresh report ID required')
    output=ROOT/'reports/v10_detail'/report_id;output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('Producer: scripts.v10_detail.prepare. protocol.json: frozen science; plan.json: '
        'resource/staging plan; code/: exact source/input overlays; inputs.json: full runtime ledger; validation_selection.csv: '
        'frozen sources; mask_thresholds.json: unchanged train warnings; offline_package.txt/json: exact staged import/input/data '
        'checks; tests.txt/json: regression checks; PROPOSAL.md: reviewable unapproved allowance; pricing.json: source/rate calculation; '
        'readiness_inputs.json: prepared artifact hashes; complete.json: preparation scope. No cloud work is started.\n',encoding='utf-8')
    prior=ROOT/'reports/v8_generator_dev/adapter_preparation_20261117_02'
    p=make_protocol(json.loads((prior/'protocol.json').read_text()));validate(p)
    json_write(output/'protocol.json',p)
    shutil.copyfile(prior/'validation_selection.csv',output/'validation_selection.csv')
    cutoff=ROOT/p['scope']['mask_warning_cutoffs'];shutil.copyfile(cutoff,output/'mask_thresholds.json')
    archive=ROOT/'reports/v8_generator_dev/full_bundle_20261116_01/inputs.zip'
    with zipfile.ZipFile(archive) as z:original=json.loads(z.read('_payload_manifest.json'))
    files=[]
    for folder in ('scripts/v10_detail','scripts/v9_detail','scripts/v8_generator_dev','src/models'):
        files.extend((ROOT/folder).glob('*.py'))
    files.extend(ROOT/name for name in ('scripts/v6_structure/model.py','scripts/v5_anatomy/metrics.py',
        'scripts/v5_anatomy/anchored_sampler.py','scripts/v7_geometry/orientation.py','scripts/v7_geometry/support.py',
        'scripts/v2_quickwins/judge_model.py','scripts/v4_convergence/core.py'))
    files.append(cutoff)
    patches={}
    for path in files:
        name=path.relative_to(ROOT).as_posix()
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'))
        target=output/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        patches[name]=digest(target)
    shutil.copyfile(ROOT/'scripts/v10_detail/modal_worker.py',output/'modal_worker.py')
    inputs={**original,**patches,p['parent']:p['parent_sha256']}
    inputs[(output/'protocol.json').relative_to(ROOT).as_posix()]=digest(output/'protocol.json')
    inputs[(output/'validation_selection.csv').relative_to(ROOT).as_posix()]=digest(output/'validation_selection.csv')
    inputs[(output/'mask_thresholds.json').relative_to(ROOT).as_posix()]=digest(output/'mask_thresholds.json')
    json_write(output/'inputs.json',inputs)
    rate=1.2*(.000306+2*.0000131+16*.00000222)
    plan=dict(run_id='guide_replicates_20261117_01',prepared_relative=output.relative_to(ROOT).as_posix(),
        archive=archive.relative_to(ROOT).as_posix(),archive_sha256=digest(archive),patches=patches,
        parent=p['parent'],parent_sha256=p['parent_sha256'],protocol_sha256=canonical_hash(p),
        source_parent_project='v8_generator_dev/precision_continuation_20261117_01/project',
        proposed_additional_usd=12,price_with_margin=rate,reserve_usd=1.,maximum_worker_seconds=23400,
        maximum_training_seconds=9000,maximum_full_batch_gate_seconds=900,
        one_app=True,one_gpu=True,automatic_paid_restart=False,budget_approved=False)
    json_write(output/'plan.json',plan)
    json_write(output/'pricing.json',dict(source='https://modal.com/pricing',checked_date='2026-11-17',
        a10_per_second=.000306,cpu_core_per_second=.0000131,memory_gib_per_second=.00000222,
        cpu_cores=2,memory_gib=16,margin=1.2,rate_with_margin=rate,reserve_usd=1.,
        maximum_worker_seconds=23400,conservative_cap_estimate_usd=rate*23400+1.,
        exact_invoice_known=False))
    # Check the composed source/archive, not just imports from the working tree.
    with tempfile.TemporaryDirectory(prefix='detail_package_',dir=ROOT) as temporary:
        staged=project_path(temporary)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(original)|{'_payload_manifest.json'}:
                raise ValueError('Original archive membership mismatch')
            for name,expected in original.items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive member')
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive content mismatch')
                dest=staged/name;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(data)
        for name in patches:
            dest=staged/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(output/'code'/name,dest)
        dest=staged/p['parent'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project_path(p['parent']),dest)
        runtime=staged/plan['prepared_relative'];runtime.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','inputs.json','validation_selection.csv','mask_thresholds.json'):
            shutil.copyfile(output/name,runtime/name)
        code=('import json,sys; from pathlib import Path; '
              'from scripts.v8_generator_dev.io import verify_inputs; '
              'from scripts.v8_generator_dev.data import CanonicalDataset; '
              'from scripts.v10_detail.engine import Trainer; from scripts.v10_detail.evaluate import evaluate; '
              'from scripts.v10_detail.verify import verify; from scripts.v10_detail.analyse import analyse; '
              'from scripts.v10_detail.protocol import validate; '
              'p=Path(sys.argv[1]); verify_inputs(json.loads((p/"inputs.json").read_text())); '
              'v=json.loads((p/"protocol.json").read_text()); validate(v); '
              'a=CanonicalDataset(v["view"],"train"); b=CanonicalDataset(v["view"],"val"); '
              'assert len(a)==4434 and len(b)==987; assert a.frame.patient_id.nunique()==72 and b.frame.patient_id.nunique()==15; '
              'print("EXACT PACKAGE IMPORTS, INPUTS, PROTOCOL AND TRAIN/VAL MANIFESTS PASS")')
        with (output/'offline_package.txt').open('x',encoding='utf-8') as log:
            check=subprocess.run([sys.executable,'-c',code,plan['prepared_relative']],cwd=staged,
                env=dict(os.environ,PYTHONPATH=str(staged)),stdout=log,stderr=subprocess.STDOUT,timeout=600)
        json_write(output/'offline_package.json',dict(exit_code=check.returncode,exact_staged_inputs=True,cloud_launched=False))
        if check.returncode:raise RuntimeError('Exact staged package failed; preserve this failed report')
    with (output/'tests.txt').open('x',encoding='utf-8') as log:
        tested=subprocess.run([sys.executable,'-m','unittest','scripts.v10_detail.test_study','-v'],cwd=ROOT,
                             stdout=log,stderr=subprocess.STDOUT,timeout=600)
    json_write(output/'tests.json',dict(exit_code=tested.returncode))
    if tested.returncode:raise RuntimeError('Study tests failed; no launch')
    proposal='''# Replicated finer-guide experiment: separate US$12 proposal

This package is prepared but not approved. No cloud job has been launched.

## Exact work

- One A10, one app/container, at most 6.5 worker hours; zero automatic retries
  or paid restarts. Previous allowances remain closed and are not transferred.
- First phase: at most 15 minutes for nine discarded full-batch-32 CUDA
  backward/optimiser updates under the pinned target runtime. Failure stops
  the app before efficacy training, retaining receipts and spent time.
- Nine newly initialised adapters on the frozen epoch-130 additive parent:
  coarse/coarse, coarse/fine and learned-zero, each at training seeds
  583301/583302/583303. All use the same capacity, data/noise/dropout draws
  within seed, optimiser, unchanged coarse consistency loss and 20 epochs.
- Retain matched blocks every five epochs; no validation checkpoint selection.
  Raw final adapters are primary. Interrupted/unmatched work stays incomplete.
- Manufacturer: all declared correct/zero/absent/wrong/coarse-erased/fine-erased
  guide controls for both source-guide arms; correct/absent for learned-zero.
  Field strength: correct guide only, explicitly exploratory, support unchanged.
  Every evaluation uses all 15 frozen development patients and three generation
  seeds: 2,295 prospective cases. This is reused development, not a fresh test.
- No mask sampling change: only source-fixed strict regional diagnostics and
  warnings. Keep all warnings and invalid regions, without favourable exclusion.
  The separate foreground-sampling ablation is outside this allocation.

## Frozen decision and limits

protocol.json records six simultaneous gates with Bonferroni-adjusted crossed
patient/training-replica bootstrap intervals: target interior MAD and HF
residual must improve, null interior MAD must not rise by .005, and each named
judge's target-minus-null editing change must exceed -.05. All three replicas'
point directions and original support/correspondence safeguards must pass.
The .005 and five-percentage-point tolerances are prospective research choices,
not clinical safety/equivalence margins. All other profiles, effects and failures
remain visible. Missing primary cells make advancement unavailable. Passing
only nominates a development candidate for later independent confirmation.

Full-mask and absent-guide software checks do not validate anatomy. Prior
source-copy/scanner leakage and tiny-erasure sensitivity limits remain. No
clinical preservation, causal scanner effect or automatic default promotion.
EchoTime, old splits, publication results and data admission stay unchanged.

## Cost and time

The separate proposed allowance is US$12, with a 23,400-second worker cap,
a 9,000-second training cap and a US$1 reserve for staging/billing uncertainty.
The guard uses the current official A10/2-core/16-GiB list prices with a 20%
margin: US$0.000441264/second. Full worker-cap accounting plus reserve is about
US$11.33; this is an estimate-based guard, not an exact invoice guarantee.
Source: https://modal.com/pricing (checked 17 November 2026); pricing.json
retains the calculation. Build/staging and failed attempts consume the same
new allowance. The app stops incomplete at a resource/error limit, never resets
the allowance. Conservative timing extrapolated from the saved adapter and
strength runs is roughly six hours, with additional local retrieval/regrading.

## Execution and recovery

The exact archive/code/parent/protocol/data/judge package has an offline staged
import/input check. Local engineering checks are reported separately; they do
not replace the first pinned-A10 full-batch gate. The launch command checks
there is no other active workspace app and verifies all readiness hashes.
Completed common blocks are hash-checked and backed up before publication.
Terminal failures and partial logs/checkpoints are retained, without an
automatic restart. After completion, read-only retrieval, strict local saved-
array/metric/named-judge verification and the frozen crossed analysis precede
any scientific conclusion. No further paid run, download, contact, publication,
new-patient admission or default change is authorised by this proposal.
'''
    (output/'PROPOSAL.md').write_text(proposal,encoding='utf-8')
    evidence={path.relative_to(ROOT).as_posix():digest(path) for path in
        [output/name for name in ('protocol.json','inputs.json','plan.json','validation_selection.csv',
         'mask_thresholds.json','modal_worker.py','offline_package.json','tests.json','PROPOSAL.md','pricing.json')]}
    evidence.update({(output/'code'/name).relative_to(ROOT).as_posix():value for name,value in patches.items()})
    json_write(output/'readiness_inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,local_package_prepared=True,
        exact_offline_package_passed=True,study_tests_passed=True,budget_approved=False,
        proposed_additional_usd=12,target_runtime_full_batch_gate_pending=True,cloud_launched=False,
        efficacy_started=False,clinical_preservation='unknown'))
    print('REPLICATED STUDY PREPARED',output,flush=True)
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--report-id',required=True)
    prepare(ap.parse_args().report_id)
