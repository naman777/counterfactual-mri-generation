"""Independent float64 Torch calculation from every raw epsilon tensor and fixed input."""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import load_ema,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v9_detail.masks import noise_aware_mask

METRICS=('label_response_rms','response_ratio_to_parent','response_cosine_to_parent','unconditional_adapter_shift_rms')


def metrics(prediction,parent,mask):
    a=torch.as_tensor(prediction,dtype=torch.float64);b=torch.as_tensor(parent,dtype=torch.float64)
    m=torch.as_tensor(mask,dtype=torch.bool)
    if a.shape!=(3,1,128,128) or b.shape!=a.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid raw epsilon shape or values')
    if int(m.sum())<64:return {name:np.nan for name in METRICS}
    change=(a[1,0]-a[0,0])[m];reference=(b[1,0]-b[0,0])[m]
    rms=float(torch.sqrt(torch.mean(change.square())));baseline=float(torch.sqrt(torch.mean(reference.square())))
    denominator=float(torch.linalg.vector_norm(change)*torch.linalg.vector_norm(reference))
    return dict(label_response_rms=rms,response_ratio_to_parent=rms/baseline if baseline>1e-10 else np.nan,
        response_cosine_to_parent=float(torch.dot(change,reference))/denominator if denominator>1e-12 else np.nan,
        unconditional_adapter_shift_rms=float(torch.sqrt(torch.mean((a[2,0]-b[2,0])[m].square()))))


def run(prepared,probe,run_id):
    prepared,probe=map(project_path,(prepared,probe));p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((probe/'inputs.json').read_text()))
    complete=json.loads((probe/'complete.json').read_text())
    if not complete['complete'] or complete['observations']!=2160:raise ValueError('Complete raw probe required')
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.verify_probe. checks.json: '
        'every raw epsilon metric independently recalculated with float64 Torch, fixed source/mask/q '
        'and CPU-conditioner parity, exact replay of thirty local CUDA initial noise inputs; '
        'inputs.json: all raw and input identities; complete.json: verification scope. All model '
        'forwards are not independently replayed; no new reverse images, classifier rates or anatomy claim.\n',encoding='utf-8')
    deterministic_runtime();torch.set_num_threads(2)
    scientific=json.loads(project_path(p['scientific_protocol']).read_text())['reference_protocol']
    data=CanonicalDataset(scientific['view'],'val');selected=pd.read_csv(project_path(p['selection'])).dataset_index.astype(int).tolist()
    c,base,scheduler,payload=load_ema(project_path(scientific['parent']),'cpu');del base,payload
    runtime=json.loads((probe/'runtime.json').read_text())
    if runtime['torch']!=str(torch.__version__) or runtime['gpu']!=torch.cuda.get_device_name(0):
        raise ValueError('Exact local-noise replay runtime changed')
    frame=pd.read_csv(probe/'per_forward.csv',keep_default_na=False,na_values=[''],float_precision='round_trip')
    design=p['prospective_forward_probe']
    keys=['study','mode','training_seed','generation_seed','dataset_index','timestep','guide_control']
    expected=pd.MultiIndex.from_product([design['studies'],design['modes'],design['training_seeds'],
        design['generation_seeds'],selected,design['timesteps'],design['controls']],names=keys)
    actual=pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or set(actual)!=set(expected):raise ValueError('Raw probe design mismatch')
    parents={};evidence={};maximum=0.;q_error=0.;condition_error=0.;absent=0
    for row in frame.itertuples():
        item=data[row.dataset_index];m,f=int(item['manufacturer_label']),int(item['field_strength_label']);echo=float(item['echo_time_norm'])
        guide=selected[(selected.index(row.dataset_index)+1)%len(selected)] if row.guide_control=='wrong' else row.dataset_index
        if row.patient_id!=item['patient_id'] or row.guide_dataset_index!=guide or row.guide_patient_id!=data.frame.iloc[guide].patient_id:
            raise ValueError('Probe source/guide identity mismatch')
        if (row.source_manufacturer,row.target_manufacturer,row.source_field_strength,row.target_field_strength)!=(m,1-m,f,f):
            raise ValueError('Edited the wrong acquisition axis')
        if row.source_echo_time_norm!=echo or row.target_echo_time_norm!=echo or row.actual_sampling_seed!=row.generation_seed+row.dataset_index:
            raise ValueError('Probe EchoTime or seed drift')
        if row.parent_file not in parents:
            path=probe/row.parent_file
            if digest(path)!=row.parent_sha256:raise ValueError('Raw parent point changed')
            with np.load(path,allow_pickle=False) as z:point={k:z[k].copy() for k in z.files}
            source=data.image_01(row.dataset_index);np.testing.assert_array_equal(point['source'],source)
            mask=ndi.binary_erosion(noise_aware_mask(source)[0],iterations=8,border_value=0)
            np.testing.assert_array_equal(point['mask'],mask)
            generator=torch.Generator(device='cuda').manual_seed(row.actual_sampling_seed)
            noise=torch.randn((1,1,128,128),device='cuda',generator=generator).cpu()
            np.testing.assert_array_equal(point['noise'],noise.numpy())
            xt=scheduler.q_sample(item['image'][None],torch.tensor([row.timestep]),noise).numpy()
            q_error=max(q_error,float(np.max(np.abs(xt-point['xt']))))
            np.testing.assert_allclose(xt,point['xt'],atol=2e-6,rtol=2e-6)
            label=lambda x:torch.tensor([x]);ee=torch.tensor([echo])
            conditions=torch.cat([c(label(m),label(f),ee),c(label(1-m),label(f),ee),
                c(label(m),label(f),ee,force_unconditional=True)]).detach().numpy()
            condition_error=max(condition_error,float(np.max(np.abs(conditions-point['conditions']))))
            np.testing.assert_allclose(conditions,point['conditions'],atol=2e-6,rtol=2e-6)
            if float(point['source_echo_time_norm'])!=echo or float(point['target_echo_time_norm'])!=echo:
                raise ValueError('Raw fixed EchoTime changed')
            parents[row.parent_file]=point;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        point=parents[row.parent_file];path=probe/row.epsilon_file
        if digest(path)!=row.epsilon_sha256:raise ValueError('Raw predicted epsilon changed')
        with np.load(path,allow_pickle=False) as z:prediction=z['epsilon'].copy()
        calculated=metrics(prediction,point['epsilon'],point['mask'])
        for name,value in calculated.items():
            saved=getattr(row,name)
            if pd.isna(value) and pd.isna(saved):continue
            error=abs(value-saved);maximum=max(maximum,error)
            if not np.isfinite(value) or not np.isfinite(saved) or error>1e-12:
                raise ValueError('Independent raw epsilon metric differs')
        if row.guide_control=='absent':
            np.testing.assert_allclose(prediction,point['epsilon'],atol=2e-6,rtol=2e-6);absent+=1
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'checks.json',dict(observations=len(frame),parent_points=len(parents),absent_observations=absent,
        maximum_metric_error=maximum,maximum_cpu_q_error=q_error,maximum_cpu_condition_error=condition_error,
        all_initial_cuda_noise_inputs_exactly_replayed=True,all_source_masks_echo_guides_and_labels_verified=True))
    for path in (probe/'per_forward.csv',probe/'inputs.json',probe/'complete.json',probe/'runtime.json',
                 prepared/'protocol.json',project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,raw_epsilon_observations_independently_recomputed=len(frame),
        initial_noise_points_exactly_replayed=len(parents),all_model_forwards_replayed=False,
        actual_reverse_states_probed=False,paid_gpu_calls=0,clinical_preservation='unknown'))
    print('RAW EPSILON METRICS AND INPUTS VERIFIED',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for name in ('prepared','probe','run-id'):a.add_argument('--'+name,required=True)
    run(**vars(a.parse_args()))
