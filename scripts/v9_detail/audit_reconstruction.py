"""Matched source reconstruction probes with train-template and wrong-guide controls."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from scripts.v8_generator_dev.detail_candidates import multiscale_guide
from scripts.v8_generator_dev.io import csv_write,json_write
from scripts.v9_detail.common import start,sources,finish,interval,display
from scripts.v9_detail.reconstruction import fit_global,predict_global,fit_local,predict_local


def audit(run_id):
    torch.set_num_threads(2)
    output,inputs=start(run_id,'scripts.v9_detail.audit_reconstruction','Patient-separated source reconstruction probes')
    json_write(output/'protocol.json',dict(train_sources='Historical 72 training guide-audit sources',
        validation_sources='Frozen 15 editor-validation sources; same patients but different slices from historical guide probe',
        representations=['coarse_control','coarse_fine','learned_zero'],alpha=10.,target='Source 2x2 average-pooled to 64x64; not native small-detail reconstruction',
        global_probe='6x8x8 pooled guide -> 64x64 source via centred ridge; equal-patient training',
        local_probe='6 guide channels at each 64x64 location -> same-location 2x2 source mean via ridge',
        capacity_matched=True,standardisation='Training only',baselines=['train_patient_mean_template','wrong_patient_guide'],
        selection='No alpha/model/guide selection on validation',
        limits='Positive reconstruction indicates accessible correlated source appearance; weak negative probe cannot rule out nonlinear copying or clinical damage.'))
    selection,items=sources(inputs)
    train=selection.split.eq('train').to_numpy();val=~train
    targets=np.stack([F.avg_pool2d(torch.from_numpy(x.copy())[None,None],2)[0,0].numpy() for _,x in items])
    maps={}
    with torch.no_grad():
        for mode in ('coarse_control','coarse_fine','learned_zero'):
            maps[mode]=np.concatenate([multiscale_guide(torch.from_numpy(x.copy())[None,None],mode).numpy() for _,x in items])
    template=targets[train].mean(0)
    outputs={'train_patient_mean_template':np.repeat(template[None],val.sum(),axis=0)}
    features={k:F.adaptive_avg_pool2d(torch.from_numpy(v),(8,8)).flatten(1).numpy() for k,v in maps.items()}
    models={}
    for mode in maps:
        global_model=fit_global(features[mode][train],targets[train].reshape(train.sum(),-1))
        local_model=fit_local(maps[mode][train],targets[train])
        for name,model,predictor,data in [('global',global_model,predict_global,features[mode]),
                                         ('local',local_model,predict_local,maps[mode])]:
            label=name+'__'+mode
            models.update({label+'__'+k:v for k,v in model.items()})
            outputs[label]=predictor(model,data[val]).reshape(-1,64,64)
            outputs[label+'__wrong_guide']=predictor(model,np.roll(data[val],1,axis=0)).reshape(-1,64,64)
    records=[]
    for label,images in outputs.items():
        for row,target,image in zip(selection[val].itertuples(),targets[val],images):
            records.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,probe=label,
                mad=float(np.abs(target-image).mean()),mse=float(np.mean((target-image)**2)),
                ssim=float(structural_similarity(target,image,data_range=1.,win_size=7))))
    frame=pd.DataFrame(records);csv_write(output/'patient_reconstruction.csv',frame)
    intervals=[]
    for probe,g in frame.groupby('probe'):
        for metric in ('mad','mse','ssim'):
            intervals.append(dict(probe=probe,metric=metric,**interval(g,metric)))
    summary=pd.DataFrame(intervals);csv_write(output/'patient_intervals.csv',summary)
    contrasts=[]
    for architecture in ('global','local'):
        fine=frame[frame.probe.eq(architecture+'__coarse_fine')].set_index('patient_id')
        for reference in (architecture+'__coarse_control','train_patient_mean_template',architecture+'__coarse_fine__wrong_guide'):
            coarse=frame[frame.probe.eq(reference)].set_index('patient_id').loc[fine.index]
            delta=pd.DataFrame(dict(patient_id=fine.index,mad=(fine['mad']-coarse['mad']).to_numpy(),
                                    mse=(fine['mse']-coarse['mse']).to_numpy(),ssim=(fine['ssim']-coarse['ssim']).to_numpy()))
            for metric in ('mad','mse','ssim'):
                contrasts.append(dict(architecture=architecture,reference=reference,metric=metric,**interval(delta,metric)))
    paired=pd.DataFrame(contrasts);csv_write(output/'paired_differences.csv',paired)
    csv_write(output/'patient_selection.csv',selection)
    with (output/'probe_weights.npz').open('xb') as handle:
        np.savez_compressed(handle,**models)
    with (output/'validation_reconstructions.npz').open('xb') as handle:
        np.savez_compressed(handle,source=targets[val],**outputs)
    fig,axes=plt.subplots(int(val.sum()),5,figsize=(11,2*int(val.sum())),squeeze=False)
    labels=['Source 64px mean','Train mean template','Coarse global','Fine global','Fine wrong guide']
    images=[targets[val],outputs['train_patient_mean_template'],outputs['global__coarse_control'],
            outputs['global__coarse_fine'],outputs['global__coarse_fine__wrong_guide']]
    for i,row in enumerate(selection[val].itertuples()):
        for j,values in enumerate(images):
            axes[i,j].imshow(values[i],cmap='gray',vmin=0,vmax=1);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(labels[j],fontsize=9)
        axes[i,0].set_ylabel(row.patient_id,fontsize=8)
    fig.suptitle('Every validation patient; fixed train-only source reconstruction probes')
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/'all_validation_reconstructions.png',dpi=110);plt.close(fig)
    lines=['# Source reconstruction: accessible appearance and shortcut risk','',
        'Both guide arms have the same six channels and probe capacity. All coefficients and standardisation '
        'are fitted using the training sources only, without tuning. Validation uses the editor sources rather '
        'than the different slices in the earlier label-probe audit. Source reconstruction is diagnostic; '
        'no decoder is added to the generator.','',
        '| Probe | MAD [95% patient CI] | SSIM [95% patient CI] |','|---|---|---|']
    for label in outputs:
        g=summary[summary.probe.eq(label)].set_index('metric')
        lines.append(f'| {label} | {display(g.loc["mad"])} | {display(g.loc["ssim"])} |')
    lines += ['', '| Fine-minus-reference contrast | MAD change [95% patient CI] |','|---|---|']
    for row in paired[paired.metric.eq('mad')].itertuples():
        lines.append(f'| {row.architecture} vs {row.reference} | {display(row)} |')
    lines += ['', 'Negative MAD difference means easier reconstruction for the fine guide. '
        'This is evidence about accessible source appearance under a weak fixed decoder, not a claim that '
        'a trained adapter actually copies pixels. The targets are 2x2 source averages: the probe does '
        'not establish recovery of original native fine structures or pathology. Wrong guides are another '
        'validation patient in a deterministic cyclic order. A good spatial prior or source confounding can '
        'also improve these scores. Low reconstruction accuracy cannot rule out stronger nonlinear shortcuts. '
        'No scanner-invariance, clinical preservation or efficacy decision follows. Intervals are exploratory '
        'equal-patient bootstraps without multiplicity adjustment.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'OUTPUTS.md').write_text('Producer: scripts.v9_detail.audit_reconstruction. '
        'patient_reconstruction.csv: every prediction metric; patient_intervals.csv: all patient CIs; '
        'paired_differences.csv: fine-minus-coarse/template/wrong-guide CIs; probe_weights.npz: train-only decoder coefficients; '
        'validation_reconstructions.npz: all predictions; all_validation_reconstructions.png: all-patient panel; '
        'patient_selection.csv: source membership; RESULTS.md: interpretation.\n',encoding='utf-8')
    finish(output,inputs,train_patients=int(train.sum()),validation_patients=int(val.sum()),
        probe_training_only=True,source_copy_resistance_established=False,guide_selected=False)
    print('SOURCE RECONSTRUCTION COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    audit(p.parse_args().run_id)
