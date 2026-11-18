"""Independently regrade pilot arrays and compare parents, baselines and guide controls."""
import argparse
import gc
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.adapter_pilot import pilot_settings
from scripts.v8_generator_dev.analyse_full import compare_rows, primary_rows
from scripts.v8_generator_dev.audit_failed_pair import finite_state
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native, summarise
from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.preservation_screen_v2 import METRICS, measurements, screen


def audit_training(folder,train,settings):
    """Recheck actual logged exposure and matched draws without cross-version RNG claims."""
    batches=(len(train)+settings['batch_size']-1)//settings['batch_size']
    expected_keys={backbone+'__'+mode for backbone in ('additive','scale_shift')
        for mode in ('soft_source_guide','learned_zero_guide')}
    markers=sorted(folder.glob('block_*/complete.json'))
    if len(markers)!=4:
        raise ValueError('Four completed five-epoch pilot blocks required')
    previous=0
    audit=[]
    for number,marker in enumerate(markers,1):
        block=json.loads(marker.read_text())
        stop=number*settings['block_epochs']*batches
        if block['common_step']!=stop or block['engineering'] or set(block['copies'])!=expected_keys:
            raise ValueError('Pilot block exposure or copy identity mismatch')
        matched=None
        for key in sorted(expected_keys):
            records=[json.loads(line) for line in (marker.parent/(key+'.jsonl')).read_text().splitlines()]
            if [row['global_step'] for row in records]!=list(range(previous+1,stop+1)):
                raise ValueError('Missing/duplicated training update')
            draws=[]
            for row in records:
                step=row['global_step']-1
                size=min(settings['batch_size'],len(train)-(step%batches)*settings['batch_size'])
                indices=row['sample_indices']
                if len(indices)!=size or any(not 0<=index<len(train) for index in indices):
                    raise ValueError('Invalid logged source batch')
                if row['patient_ids']!=train.frame.iloc[indices].patient_id.tolist():
                    raise ValueError('Logged training source patient mismatch')
                if row['epoch_at_draw']!=step//batches or not row['echo_time_fixed'] or not row['applied']:
                    raise ValueError('Skipped update, changed EchoTime or mismatched epoch')
                if len(row['timesteps'])!=size or any(not 0<=t<1000 for t in row['timesteps']) or len(row['condition_dropped'])!=size:
                    raise ValueError('Invalid timestep/dropout draw')
                if not 0<=row['amp_retries']<=settings['max_same_batch_precision_retries']:
                    raise ValueError('Retry bound exceeded')
                if not np.isfinite([row[name] for name in ('total_loss','noise_mse','structure_proxy_loss','gradient_norm')]).all():
                    raise ValueError('Nonfinite logged training loss or gradient')
                draws.append({name:row[name] for name in ('global_step','sample_indices','patient_ids',
                    'timesteps','condition_dropped','noise_sha256')})
            if matched is None:
                matched=draws
            elif matched!=draws:
                raise ValueError('Guide/baseline/backbone training draws diverged')
            audit.append(dict(block=number,copy=key,updates=len(records),start_step=previous+1,
                final_step=stop,all_applied=True,finite_losses_and_gradients=True,matched_draws=True))
        previous=stop
    return audit


def analyse(fetched,parent_fetch,run_id,device='cuda'):
    deterministic_runtime()  # Match the frozen cloud and existing saved-array verifier.
    fetched,parent_fetch=map(project_path,(fetched,parent_fetch))
    receipt=json.loads((fetched/'complete.json').read_text())
    if not receipt['workflow_success'] or not receipt['final_adapters_verified']:
        raise ValueError('A completed four-copy pilot and frozen validation are required')
    verify_inputs(json.loads((fetched/'inputs.json').read_text()))
    verify_inputs(json.loads((parent_fetch/'inputs.json').read_text()))
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.analyse_adapter. protocol.json retains development '
        'scope. input_checks.csv verifies original runtime bytes against archive/current files. '
        'array_checks.csv and grading_checks.csv independently recompute sources, labels, support, '
        'EchoTime, SSIM/MAD and every named-judge probability/decision. comparisons.csv gives paired '
        'guide-minus-zero, pilot-minus-parent and guide-control-minus-correct patient CIs. '
        'estimates.csv retains all named judge/support/direction results. preservation_metrics.csv '
        'and preservation_intervals.csv apply the unchanged saved proxy envelope; no cutoffs are fit. '
        'checkpoint_checks.json audits finite adapter/optimiser/scaler and frozen-parent identity. '
        'training_audit.csv independently checks every logged applied update and matched draw. '
        'inputs.json binds evidence. complete.json means analysis completion, not clinical validity.\n',encoding='utf-8')
    prepared=project_path(receipt['prepared'])
    protocol=json.loads((prepared/'protocol.json').read_text())
    config=protocol['parent_config']
    if protocol['settings']!=pilot_settings():
        raise ValueError('Pilot protocol drift')
    train,val=[CanonicalDataset(protocol['view'],split) for split in ('train','val')]
    selected=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()
    expected={(int(index),axis,int(seed)) for index in selected for axis in ('manufacturer','field_strength')
        for seed in [config['evaluation']['selection_generation_seed'],*config['evaluation']['confirmation_generation_seeds']]}
    threshold_path=ROOT/'reports/v8_generator_dev/preservation_benchmark_v2_20261117_01/thresholds.json'
    thresholds=json.loads(threshold_path.read_text())['thresholds']
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in
        (fetched/'complete.json',fetched/'inputs.json',parent_fetch/'complete.json',parent_fetch/'inputs.json',
         prepared/'protocol.json',threshold_path,
         ROOT/'scripts/v8_generator_dev/analyse_adapter.py')}
    trained=json.loads((project_path(receipt['training'])/'complete.json').read_text())
    training_audit=audit_training(project_path(receipt['training']),train,protocol['settings'])
    csv_write(output/'training_audit.csv',pd.DataFrame(training_audit))
    for path in project_path(receipt['training']).rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl'):
            inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    checkpoint_checks={}
    for key,copy in trained['copies'].items():
        path=project_path(copy['checkpoint'])
        if digest(path)!=copy['sha256']:
            raise ValueError('Adapter bytes changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        checks={name:finite_state(ck[name]) for name in ('adapter_state_dict','optimizer_state_dict','scaler_state_dict')}
        checks.update(parent_hash_matches=digest(project_path(ck['parent']))==ck['parent_sha256'],
            settings_match=ck['settings']==protocol['settings'],epoch_twenty=ck['epoch']==20,
            nonengineering=not ck['engineering'],frozen_base=ck['checks']['base_weights_unchanged'],
            frozen_conditioner=ck['checks']['conditioner_unchanged'])
        if not all(checks.values()):
            raise ValueError('Adapter checkpoint audit failed')
        checkpoint_checks[key]=checks
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        del ck
    json_write(output/'checkpoint_checks.json',checkpoint_checks)
    json_write(output/'protocol.json',dict(prepared=receipt['prepared'],parent_fetch=str(parent_fetch.relative_to(ROOT)),
        original_fixed_points=config['evaluation']['fixed_settings'],clinical_preservation='unknown',
        local_torch_version=str(torch.__version__),cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,
        probability_absolute_tolerance=1e-5,
        fresh_test=False,guide_invariance_established=False,thresholds_retuned=False))
    homes={key:project_path(path) for key,path in receipt['evaluations'].items()}
    frames,checks,natives,preservation,input_checks={},{},{},[],[]
    array_rows=[]
    archive_path=ROOT/'reports/v8_generator_dev/full_bundle_20261116_01/inputs.zip'
    with zipfile.ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('_payload_manifest.json'))
        for key,home in homes.items():
            complete=json.loads((home/'complete.json').read_text())
            ep=json.loads((home/'protocol.json').read_text())
            if complete['engineering'] or not complete['complete'] or ep['config']!=config:
                raise ValueError('Incomplete or drifting pilot evaluation')
            for name,wanted in json.loads((home/'inputs.json').read_text()).items():
                path=project_path(name)
                if path.is_file() and digest(path)==wanted:
                    found=wanted
                    location='unchanged_local_input'
                elif name in manifest:
                    found=hashlib.sha256(archive.read(name)).hexdigest()
                    location='original_approved_archive'
                else:
                    raise ValueError('Unresolved pilot runtime input: '+name)
                if found!=wanted:
                    raise ValueError('Changed pilot runtime input: '+name)
                input_checks.append(dict(evaluation=key,path=name,sha256=found,location=location))
            requests=pd.read_csv(home/'requests.csv',keep_default_na=False)
            table=read_grading_csv(home/'per_image.csv')
            if len(table)!=len(requests)*9 or len(requests)!=complete['cases']:
                raise ValueError('Missing pilot request/control/judge rows')
            if requests.case.tolist()!=list(range(len(requests))) or requests.duplicated(['dataset_index','axis','generation_seed']).any():
                raise ValueError('Duplicate/nonsequential pilot requests')
            if set(requests[['dataset_index','axis','generation_seed']].itertuples(index=False,name=None))!=expected:
                raise ValueError('Unexpected source/axis/seed coverage')
            frames[key]=table
            checks[key]=table.set_index(['case','judge','variant'],verify_integrity=True)
            natives[key]={}
            for request in requests.itertuples():
                item=val[int(request.dataset_index)]
                m,f=item['manufacturer_label'],item['field_strength_label']
                original=m if request.axis=='manufacturer' else f
                tm,tf=(1-m,f) if request.axis=='manufacturer' else (m,1-f)
                if (request.patient_id,request.target_manufacturer,request.target_field_strength,request.direction,
                    request.actual_sampling_seed)!=(item['patient_id'],tm,tf,f'{original}_to_{1-original}',request.generation_seed+request.dataset_index):
                    raise ValueError('Source/target request identity mismatch')
                point=config['evaluation']['fixed_settings'][request.axis]
                if (request.timestep,request.guidance)!=(point['timestep'],point['guidance']):
                    raise ValueError('Fixed operating point changed')
                support=request_support(train.frame,tm,tf,float(item['echo_time_norm']),config['data']['minimum_cell_patients'])
                for name,value in support.items():
                    actual=getattr(request,name)
                    if isinstance(value,float):
                        if not np.isclose(actual,value,rtol=0,atol=1e-12):
                            raise ValueError('Support field mismatch')
                    elif actual!=value:
                        raise ValueError('Support field mismatch')
                guide_index=int(request.guide_dataset_index)
                expected_guide=selected[(selected.index(int(request.dataset_index))+1)%len(selected)] if ep['guide_control']=='wrong' else int(request.dataset_index)
                if guide_index!=expected_guide or request.guide_patient_id!=val.frame.iloc[guide_index].patient_id:
                    raise ValueError('Guide source identity mismatch')
                path=home/request.array_file
                with np.load(path,allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['source'],val.image_01(int(request.dataset_index)))
                    for echo in (arrays['source_echo_time_norm'].item(),arrays['target_echo_time_norm'].item(),
                        request.source_echo_time_norm,request.target_echo_time_norm):
                        if not np.isclose(echo,float(item['echo_time_norm']),rtol=0,atol=1e-15):
                            raise ValueError('EchoTime changed')
                    if (int(arrays['target_manufacturer']),int(arrays['target_field_strength']))!=(tm,tf):
                        raise ValueError('Saved target labels changed')
                    stack=np.stack([arrays[name] for name in ('source','null','target')])
                if stack.shape!=(3,128,128) or not np.isfinite(stack).all() or stack.min()<0 or stack.max()>1:
                    raise ValueError('Invalid image arrays')
                native=source_native(stack,val.plans[int(request.dataset_index)])
                natives[key][int(request.case)]=native
                for j,control in enumerate(('identity','null','target')):
                    rows=table[(table.case==request.case)&(table.variant==control)]
                    ssim=float(structural_similarity(stack[0],stack[j],data_range=1.,win_size=7))
                    mad=float(np.abs(stack[0]-stack[j]).mean())
                    if not np.allclose(rows.ssim,ssim,rtol=0,atol=1e-9) or not np.allclose(rows.mad,mad,rtol=0,atol=1e-9):
                        raise ValueError('Saved correspondence metric mismatch')
                    record=measurements(native[0],native[j])
                    preservation.append(dict(evaluation=key,patient_id=request.patient_id,axis=request.axis,
                        direction=request.direction,condition_support_status=request.condition_support_status,
                        generation_seed=request.generation_seed,control=control,**record,**screen(record,thresholds)))
                array_rows.append(dict(evaluation=key,case=request.case,source_labels_echo_support_exact=True,
                    guide_identity_exact=True,finite_images=True,ssim_mad_reproduced=True))
                inputs[path.relative_to(ROOT).as_posix()]=digest(path)
            for name in ('complete.json','protocol.json','per_image.csv','requests.csv'):
                path=home/name
                inputs[path.relative_to(ROOT).as_posix()]=digest(path)
            print('PILOT ARRAYS VERIFIED',key,len(requests),flush=True)
    grade_checks=[]
    for seed,path in enumerate(config['evaluation']['judge_paths']):
        judge=load_judge(project_path(path),device).eval().requires_grad_(False)
        name=f'heldout_orientation_robust_seed{seed}'
        for key,frame in frames.items():
            error_max=0.
            requests=pd.read_csv(homes[key]/'requests.csv')
            for request in requests.itertuples():
                probability=predict_judge(judge,natives[key][request.case],device)
                axis,other=('mfg','fs') if request.axis=='manufacturer' else ('fs','mfg')
                target=request.target_manufacturer if axis=='mfg' else request.target_field_strength
                null_hit=int(probability[axis][1].argmax()==target)
                for j,control in enumerate(('identity','null','target')):
                    saved=checks[key].loc[(request.case,name,control)]
                    hit=int(probability[axis][j].argmax()==target)
                    error=abs(float(probability[axis][j,target])-saved.target_probability)
                    error_max=max(error_max,error)
                    if error>1e-5 or (hit,hit-null_hit)!=(saved.target_hit,saved.paired_hit_delta):
                        raise ValueError('Pilot target grading mismatch')
                    if int(probability[axis][0].argmax()!=target)!=saved.source_correct:
                        raise ValueError('Pilot source grading mismatch')
                    if int(probability[other][j].argmax()==probability[other][0].argmax())!=saved.other_axis_stable:
                        raise ValueError('Pilot other-axis grading mismatch')
            grade_checks.append(dict(evaluation=key,judge=name,cases=len(requests),
                maximum_probability_absolute_error=error_max,discrete_decisions_exact=True))
            print('PILOT JUDGE VERIFIED',key,name,flush=True)
        del judge
        gc.collect()
    comparisons=[]
    parents=json.loads((parent_fetch/'complete.json').read_text())['evaluation_folders']
    for backbone in ('additive','scale_shift'):
        guided=frames[backbone+'__soft_source_guide__correct']
        zero=frames[backbone+'__learned_zero_guide__correct']
        parent=primary_rows(read_grading_csv(project_path(parents[backbone])/'per_image.csv'),config)
        contrasts=[('guide_minus_zero',zero,guided),('guided_minus_parent',parent,guided),('zero_minus_parent',parent,zero)]
        for control in ('absent','wrong'):
            contrasts.append((control+'_minus_correct',guided,frames[backbone+'__soft_source_guide__'+control]))
        for contrast,before,after in contrasts:
            part=compare_rows(before,after,config)
            part['backbone']=backbone
            part['contrast']=contrast
            comparisons.append(part)
        # A zero presence gate must reproduce the frozen parent, not merely a similar hit rate.
        parent_home=project_path(parents[backbone])
        parent_requests=pd.read_csv(parent_home/'requests.csv')
        absent_home=homes[backbone+'__soft_source_guide__absent']
        for request in pd.read_csv(absent_home/'requests.csv').itertuples():
            row=parent_requests[(parent_requests.dataset_index==request.dataset_index)&(parent_requests.axis==request.axis)&
                (parent_requests.generation_seed==request.generation_seed)&(parent_requests.timestep==request.timestep)&
                (parent_requests.guidance==request.guidance)]
            if len(row)!=1:
                raise ValueError('No unique matched parent control')
            with np.load(absent_home/request.array_file) as actual,np.load(parent_home/row.iloc[0].array_file) as expected_arrays:
                for name in ('source','null','target'):
                    if not np.array_equal(actual[name],expected_arrays[name]):
                        raise ValueError('Absent guide did not exactly reproduce the frozen parent output')
    csv_write(output/'comparisons.csv',pd.concat(comparisons,ignore_index=True))
    estimates=[]
    for key,frame in frames.items():
        part=summarise(frame,config)
        part['evaluation']=key
        estimates.append(part)
    csv_write(output/'estimates.csv',pd.concat(estimates,ignore_index=True))
    preservation=pd.DataFrame(preservation)
    csv_write(output/'preservation_metrics.csv',preservation)
    intervals=[]
    for values,group in preservation.groupby(['evaluation','axis','control']):
        for stratum,part in [('all',group),*[(f'direction:{key}',p) for key,p in group.groupby('direction')],
            *[(f'support:{key}',p) for key,p in group.groupby('condition_support_status')]]:
            part=part.copy()
            part['within_proxy_envelope']=1-part.screen_flag.astype(float)
            intervals.append(dict(zip(['evaluation','axis','control'],values),stratum=stratum,
                invalid_measurement_rows=int(part.screen_flag.isna().sum()),
                **bootstrap_patients(part,'within_proxy_envelope',config['evaluation']['bootstrap_seed'],10000)))
    csv_write(output/'preservation_intervals.csv',pd.DataFrame(intervals))
    csv_write(output/'input_checks.csv',pd.DataFrame(input_checks))
    csv_write(output/'array_checks.csv',pd.DataFrame(array_rows))
    csv_write(output/'grading_checks.csv',pd.DataFrame(grade_checks))
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,saved_validation_verified=True,cases=len(array_rows),
        discrete_judge_grades_exact=True,absent_guide_equals_parent_exact=True,all_four_copies_epoch20=True,
        thresholds_retuned=False,winner_selected=False,anatomy_validated=False,fresh_test=False))
    print('ADAPTER ANALYSIS COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--parent-fetch',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    a=p.parse_args()
    analyse(a.fetched,a.parent_fetch,a.run_id,a.device)
