"""Recompute continued-model grading, compare epoch 70, and apply the frozen proxy screen."""
import argparse
import json
import re
import shutil
import zipfile

import pandas as pd
import torch

from scripts.v8_generator_dev.audit_failed_pair import finite_state
from scripts.v8_generator_dev.analyse_full import primary_rows, compare_rows, analyse as analyse_pair
from scripts.v8_generator_dev.benchmark_preservation_v2 import benchmark
from scripts.v8_generator_dev.explain_preservation import explain
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.verify_saved_validation import verify


def process(fetched, prefix):
    if not re.fullmatch('[A-Za-z0-9_-]+', prefix):
        raise ValueError('Safe new output prefix required')
    fetched=project_path(fetched)
    receipt=json.loads((fetched/'complete.json').read_text())
    completed_components=(receipt.get('training_and_validation_success') is True and
        receipt.get('cloud_analysis_import_failed') is True and (fetched/'root/failed.json').is_file())
    if not (receipt['workflow_success'] or completed_components) or not receipt['primary_checkpoint_hashes_verified']:
        raise ValueError('Completed matched workflow and primary checkpoint hashes required')
    verify_inputs(json.loads((fetched/'inputs.json').read_text()))
    base=ROOT/'reports/v8_generator_dev'
    output=base/(prefix+'_inputs')
    output.mkdir(exist_ok=False)
    runtime=output/'runtime'
    runtime.mkdir()
    trained=json.loads((fetched/'training/complete.json').read_text())
    prepared,=list((fetched/'protocol').glob('*/protocol.json'))
    config=json.loads((fetched/'training/protocol.json').read_text())['config']
    homes={name:project_path(path) for name,path in receipt['evaluation_folders'].items()}
    state_checks={}
    for model,relative in trained['checkpoints'].items():
        path=project_path(relative)
        if digest(path)!=trained['checkpoint_hashes'][model]:
            raise ValueError('Retrieved final checkpoint hash mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        checks={key:finite_state(payload[key]) for key in (
            'conditioning_state_dict','unet_state_dict','raw_conditioning_state_dict',
            'raw_unet_state_dict','optimizer_state_dict','scaler_state_dict')}
        checks['config_matches_frozen_continuation']=payload['config']==config
        checks['progress_matches_trained_pair']=payload['progress']==trained['progress'][model]
        checks['prepared_hash_matches']=payload['protocol_sha256']==trained['prepared_sha256']
        if not all(checks.values()):
            raise ValueError('Final checkpoint state audit failed')
        state_checks[model]=checks
        del payload
    json_write(output/'checkpoint_checks.json',state_checks)
    json_write(output/'complete.json',dict(complete=True,validation_retrieved=True,checkpoint_hashes_verified=True,
        evaluations=receipt['evaluation_folders'],checkpoints=trained['checkpoints']))
    archive=base/'full_bundle_20261116_01/inputs.zip'
    with zipfile.ZipFile(archive) as zipped:
        archived=set(zipped.namelist())
    candidates=[path for folder in (
        base/'failed_pair_runtime_inputs_20261117_01/files',
        base/'cloud_resume_fetch_20261116_01/protocol',prepared.parent)
        for path in folder.rglob('*') if path.is_file()]
    old_receipt=json.loads((base/'failed_pair_checkpoints_20261116_01/complete.json').read_text())
    candidates += [project_path(path) for path in old_receipt['checkpoints'].values()]
    authorisation=json.loads((fetched/'root/authorisation.json').read_text())
    run_id=authorisation['plan']['run_id']
    proposal=base/'continuation_preparation_20261117_01/plan.json'
    # The original local proposal is available unchanged; bind its bytes, not reconstructed JSON.
    if json.loads(proposal.read_text())['run_id']!=run_id:
        raise ValueError('Local continuation proposal identity mismatch')
    candidates.append(proposal)
    by_hash={digest(path):path for path in candidates}
    resolutions={}
    for home in homes.values():
        protocol=json.loads((home/'protocol.json').read_text())
        for name,expected in json.loads((home/'inputs.json').read_text()).items():
            local=project_path(name)
            if local.is_file() and digest(local)==expected:
                continue
            if name in archived:
                # The verifier checks the archive member against the expected hash itself.
                continue
            if name==protocol['checkpoint'].replace('\\','/'):
                continue
            if name.startswith('reports/v8_generator_dev/'+prepared.parent.name+'/'):
                continue
            if name in resolutions:
                continue
            if expected not in by_hash:
                raise ValueError('Unresolved original runtime input: '+name)
            source=by_hash[expected]
            destination=runtime/'files'/name.replace('\\','/')
            if not destination.resolve().is_relative_to(runtime):
                raise ValueError('Runtime alias escapes new report')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as src,destination.open('xb') as dst:
                shutil.copyfileobj(src,dst)
            resolutions[name]=dict(source=str(source.relative_to(ROOT)),sha256=digest(destination))
    json_write(runtime/'resolutions.json',resolutions)
    (runtime/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.postprocess_continuation.\n'
        'files/ retains exact immutable parent/protocol/plan bytes at original runtime-relative names. '
        'resolutions.json maps each original file/hash to its alias. No JSON or checkpoint is reconstructed.\n',encoding='utf-8')
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.postprocess_continuation`.\n'
        'complete.json adapts verified retrieval paths for the saved-array verifier; checkpoint_checks.json '
        'audits finite raw/EMA/optimizer/scaler state and frozen protocol/progress; runtime/ maps '
        'original immutable parent/protocol/plan bytes. inputs.json binds provenance. '
        'This path adapter makes no new scientific claim.\n',encoding='utf-8')
    json_write(output/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in (
        fetched/'complete.json',proposal,ROOT/'scripts/v8_generator_dev/postprocess_continuation.py')})
    verification_id=prefix+'_verification'
    verify(output,fetched,archive,runtime,verification_id)
    analyse_pair(fetched/'training',homes['additive'],homes['scale_shift'],prefix+'_comparison')
    previous=json.loads((base/'failed_pair_results_20261117_01/complete.json').read_text())
    comparison=base/(prefix+'_epoch70_comparison')
    comparison.mkdir(exist_ok=False)
    differences=[]
    evidence={}
    for model,home in homes.items():
        old=project_path(previous['evaluations'][model])
        before=primary_rows(read_grading_csv(old/'per_image.csv'),config)
        after=primary_rows(read_grading_csv(home/'per_image.csv'),config)
        delta=compare_rows(before,after,config)
        delta['model']=model
        delta['contrast']='continued_minus_epoch70'
        differences.append(delta)
        for path in (old/'per_image.csv',home/'per_image.csv',home/'protocol.json'):
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    csv_write(comparison/'paired_epoch70_differences.csv',pd.concat(differences,ignore_index=True))
    json_write(comparison/'inputs.json',evidence)
    json_write(comparison/'complete.json',dict(complete=True,matched_sources_settings_seeds=True,
        continued_epoch=trained['progress']['additive']['epoch'],previous_epoch=70,
        fresh_cohort=False,anatomy_validated=False,winner_selected=False))
    (comparison/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.postprocess_continuation`.\n'
        'paired_epoch70_differences.csv gives continued-minus-epoch70 matched contrasts for each '
        'model/axis/named judge/control/support/direction, using the frozen fixed points and all seeds. '
        'inputs.json hashes raw grading; complete.json retains development scope and unknown anatomy.\n',encoding='utf-8')
    # Identical training controls and implementation must reproduce the old cutoffs.
    # The final check refuses to interpret results if those cutoffs drift.
    benchmark(output,base/verification_id,prefix+'_preservation')
    old_thresholds=json.loads((base/'preservation_benchmark_v2_20261117_01/thresholds.json').read_text())
    new_thresholds=json.loads((base/(prefix+'_preservation')/'thresholds.json').read_text())
    if old_thresholds!=new_thresholds:
        raise AssertionError('Preservation cutoffs changed; retain unpromoted benchmark for inspection')
    explain(base/(prefix+'_preservation'),prefix+'_preservation_interpretation')
    print('CONTINUATION POSTPROCESS COMPLETE',prefix,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetched',required=True)
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    process(args.fetched,args.prefix)
