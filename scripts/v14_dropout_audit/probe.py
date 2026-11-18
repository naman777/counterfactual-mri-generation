"""Bounded laptop epsilon-response comparison, preserving every raw predicted vector."""
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet,multiscale_guide
from scripts.v8_generator_dev.full_train import load_ema,deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path,verify_inputs
from scripts.v9_detail.masks import noise_aware_mask
from scripts.v10_detail.metrics import warning_for_index
from scripts.v11_guide_audit.analyse import CrossedBootstrap
from scripts.v11_guide_audit.forward_probe import response_metrics
from scripts.v14_dropout_audit.common import paired,OLD,NEW

METRICS=['label_response_rms','response_ratio_to_parent','response_cosine_to_parent','unconditional_adapter_shift_rms']


def run(prepared,run_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    plan=p['prospective_forward_probe'];scientific=json.loads(project_path(p['scientific_protocol']).read_text())
    reference=scientific['reference_protocol']
    if not torch.cuda.is_available():raise ValueError('The bounded probe requires the laptop CUDA GPU')
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.probe. protocol.json/runtime.json: '
        'declared fixed-source/noise and local scope; parents/: saved source/noise/q/conditions/parent epsilon/mask; '
        'epsilon/: every predicted source/target/null epsilon tensor; journal.jsonl/per_forward.csv: '
        'observed metrics and raw hashes; paired_intervals.csv/per_training_seed.csv: all controls/timesteps '
        'and old/new/coarse/fine interactions, descriptive CIs; inputs.json: input/source/checkpoint hashes; '
        'complete.json: local forward-only scope. No reverse edits, training, new judge hit rates or causal proof.\n',encoding='utf-8')
    (out/'parents').mkdir();(out/'epsilon').mkdir();json_write(out/'protocol.json',plan)
    started=time.monotonic();deterministic_runtime();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    json_write(out/'runtime.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        tf32=False,autocast=False,source_generation_seeds=plan['generation_seeds']))
    data=CanonicalDataset(reference['view'],'val');train=CanonicalDataset(reference['view'],'train')
    selected=pd.read_csv(project_path(p['selection'])).dataset_index.astype(int).tolist()
    c,base,scheduler,payload=load_ema(project_path(reference['parent']),'cuda');del payload
    c.eval().requires_grad_(False);base.eval().requires_grad_(False)
    thresholds=json.loads((ROOT/'reports/v13_guide_dropout/preparation_20261118_01/mask_thresholds.json').read_text())['thresholds']['noise_aware_v1']
    inputs={};cache={};sources={};metadata={}
    for index in selected:
        item=data[index];source=data.image_01(index);sources[index]=source
        mask=ndi.binary_erosion(noise_aware_mask(source)[0],iterations=8,border_value=0)
        m,f=int(item['manufacturer_label']),int(item['field_strength_label']);echo=float(item['echo_time_norm'])
        label=lambda value:torch.tensor([value],device='cuda')
        ee=torch.tensor([echo],device='cuda')
        conditions=torch.cat([c(label(m),label(f),ee),c(label(1-m),label(f),ee),
                              c(label(m),label(f),ee,force_unconditional=True)])
        support=request_support(train.frame,1-m,f,echo,5)
        warning=warning_for_index(source,index,thresholds)
        metadata[index]=dict(patient_id=item['patient_id'],dataset_index=index,axis='manufacturer',direction=f'{m}_to_{1-m}',
            condition_support_status=support['condition_support_status'],qc_warning=bool(warning['qc_warning']),
            source_echo_time_norm=echo,target_echo_time_norm=echo,guidance=3.,
            source_manufacturer=m,target_manufacturer=1-m,source_field_strength=f,target_field_strength=f)
        source_path=project_path(data.frame.iloc[index].source_image_path)
        inputs[source_path.relative_to(ROOT).as_posix()]=digest(source_path)
        for generation in plan['generation_seeds']:
            for timestep in plan['timesteps']:
                generator=torch.Generator(device='cuda').manual_seed(generation+index)
                image=item['image'][None].cuda();noise=torch.randn(image.shape,device='cuda',generator=generator)
                times=torch.full((3,),timestep,device='cuda',dtype=torch.long)
                xt=scheduler.q_sample(image,label(timestep),noise).repeat(3,1,1,1)
                with torch.inference_mode():parent=base(xt,times,conditions).cpu().numpy()
                name=f'parents/index{index}_g{generation}_t{timestep}.npz'
                with (out/name).open('xb') as handle:
                    np.savez_compressed(handle,source=source,noise=noise.cpu().numpy(),xt=xt[:1].cpu().numpy(),
                        conditions=conditions.detach().cpu().numpy(),epsilon=parent,mask=mask,
                        source_echo_time_norm=echo,target_echo_time_norm=echo)
                cache[index,generation,timestep]=(xt,times,conditions,parent,mask,name,digest(out/name))
    trained={OLD:json.loads((ROOT/'reports/v10_detail/guide_replicates_20261117_01_training/complete.json').read_text()),
             NEW:json.loads((project_path(p['training'])/'complete.json').read_text())}
    rows=[];maximum_absent_error=0.;exact_absent=0;absent_count=0
    with (out/'journal.jsonl').open('x',encoding='utf-8') as journal:
        for study in plan['studies']:
            for seed in plan['training_seeds']:
                for mode in plan['modes']:
                    ck=trained[study]['copies'][f'seed{seed}__{mode}'];path=project_path(ck['checkpoint'])
                    if digest(path)!=ck['sha256']:raise ValueError('Frozen trained checkpoint changed')
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    if state['engineering'] or state['step']!=2780 or state['parent_sha256']!=reference['parent_sha256']:
                        raise ValueError('Incomplete or wrong-parent copy')
                    model=DetailConditionedUNet(base,16).cuda().eval();model.adapter.load_state_dict(state['adapter']);model.requires_grad_(False)
                    inputs[path.relative_to(ROOT).as_posix()]=digest(path);del state
                    guides={i:multiscale_guide(torch.from_numpy(sources[i].copy())[None,None].cuda(),mode) for i in selected}
                    for position,index in enumerate(selected):
                        for generation in plan['generation_seeds']:
                            for timestep in plan['timesteps']:
                                xt,times,conditions,parent,mask,parent_file,parent_sha=cache[index,generation,timestep]
                                for control in plan['controls']:
                                    if time.monotonic()-started>plan['max_seconds']:
                                        raise TimeoutError('Bounded laptop forward probe time limit; preserve partial evidence')
                                    guide=guides[selected[(position+1)%len(selected)]] if control=='wrong' else guides[index]
                                    guide=guide.clone()
                                    if control=='zero':guide.zero_()
                                    if control=='coarse_only':guide[:,3:]=0
                                    if control=='fine_only':guide[:,:3]=0
                                    present=torch.full((3,),0. if control=='absent' else 1.,device='cuda')
                                    with torch.inference_mode():
                                        prediction=model(xt,times,conditions,guide.repeat(3,1,1,1),present).cpu().numpy()
                                    if control=='absent':
                                        error=float(np.max(np.abs(prediction-parent)));maximum_absent_error=max(maximum_absent_error,error)
                                        absent_count+=1;exact_absent+=int(np.array_equal(prediction,parent))
                                        np.testing.assert_allclose(prediction,parent,atol=2e-6,rtol=2e-6)
                                    name=f'epsilon/observation_{len(rows):05d}.npz'
                                    with (out/name).open('xb') as handle:np.savez_compressed(handle,epsilon=prediction)
                                    row=dict(metadata[index],study=study,training_seed=seed,generation_seed=generation,
                                        timestep=timestep,guide_control=control,mode=mode,epsilon_file=name,
                                        epsilon_sha256=digest(out/name),parent_file=parent_file,parent_sha256=parent_sha,
                                        actual_sampling_seed=generation+index,
                                        guide_dataset_index=selected[(position+1)%len(selected)] if control=='wrong' else index,
                                        guide_patient_id=data.frame.iloc[selected[(position+1)%len(selected)] if control=='wrong' else index].patient_id,
                                        **response_metrics(prediction,parent,mask))
                                    rows.append(row);journal.write(json.dumps(row,allow_nan=True)+'\n');journal.flush()
                    del model,guides;gc.collect();torch.cuda.empty_cache()
                    print('LOCAL EPSILON COPY',study,seed,mode,len(rows),flush=True)
    if len(rows)!=plan['expected_observations']:raise ValueError('Forward design incomplete')
    frame=pd.DataFrame(rows);csv_write(out/'per_forward.csv',frame)
    bp=dict(p,generation_seeds=plan['generation_seeds']);bootstrap=CrossedBootstrap(bp);summary=[];replicas=[]
    terms=[('new_fine_minus_old_fine',[(NEW,'coarse_fine',1),(OLD,'coarse_fine',-1)]),
           ('new_coarse_minus_old_coarse',[(NEW,'coarse_control',1),(OLD,'coarse_control',-1)]),
           ('new_fine_minus_new_coarse',[(NEW,'coarse_fine',1),(NEW,'coarse_control',-1)]),
           ('old_fine_minus_old_coarse',[(OLD,'coarse_fine',1),(OLD,'coarse_control',-1)]),
           ('guide_effect_interaction',[(NEW,'coarse_fine',1),(NEW,'coarse_control',-1),(OLD,'coarse_fine',-1),(OLD,'coarse_control',1)])]
    # The epsilon response is not judge-labelled; pair keys therefore omit judge explicitly.
    keys=['patient_id','dataset_index','training_seed','generation_seed']
    for (timestep,control),part in frame.groupby(['timestep','guide_control']):
        for name,specification in terms:
            value=None;index=None
            for study,mode,coefficient in specification:
                subset=part[(part.study==study)&(part['mode']==mode)].set_index(keys).sort_index()
                if subset.index.has_duplicates or (index is not None and not subset.index.equals(index)):
                    raise ValueError('Unpaired epsilon-response evidence')
                index=subset.index;term=subset[METRICS]*coefficient;value=term if value is None else value+term
            result,per=bootstrap.intervals(value.reset_index(),METRICS)
            summary.extend(dict(timestep=timestep,guide_control=control,contrast=name,**r) for r in result)
            replicas.extend(dict(timestep=timestep,guide_control=control,contrast=name,**r) for r in per)
    csv_write(out/'paired_intervals.csv',pd.DataFrame(summary));csv_write(out/'per_training_seed.csv',pd.DataFrame(replicas))
    for path in (prepared/'protocol.json',prepared/'inputs.json',project_path(p['scientific_protocol']),
                 project_path(p['selection']),project_path(reference['parent']),
                 ROOT/reference['view']/'splits/val.csv',ROOT/reference['view']/'splits/train.csv',
                 ROOT/'reports/v13_guide_dropout/preparation_20261118_01/mask_thresholds.json',
                 project_path(__file__),ROOT/'scripts/v11_guide_audit/forward_probe.py'):
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    verify_inputs(inputs);json_write(out/'inputs.json',inputs)
    json_write(out/'complete.json',dict(complete=True,observations=len(rows),parents=len(cache),
        maximum_absent_error=maximum_absent_error,absent_observations=absent_count,exact_absent_observations=exact_absent,
        seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        raw_epsilon_saved_for_every_observation=True,local_gpu_only=True,paid_gpu_calls=0,
        reverse_sampling=False,new_training=False,clinical_preservation='unknown',mechanism_proven=False))
    print('RAW EPSILON PROBE COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--prepared',required=True);a.add_argument('--run-id',required=True)
    run(**vars(a.parse_args()))
