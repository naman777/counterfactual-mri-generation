"""Visualise only completed verified study outputs; preserve all patients and judges."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,json_write,digest,project_path

MODES=('coarse_control','coarse_fine','learned_zero')
LABELS=('Coarse control','Coarse + fine','Learned zero')
COLORS=('#496a8a','#b45434','#777777')


def plot(prepared,results,analysis,verified,run_id):
    if not run_id.replace('_','').isalnum():raise ValueError('Safe fresh output ID required')
    prepared,results,analysis,verified=map(project_path,(prepared,results,analysis,verified))
    if not json.loads((analysis/'complete.json').read_text())['complete'] or not json.loads((verified/'complete.json').read_text())['saved_outputs_verified']:
        raise ValueError('Completed strict verification/analysis required')
    p=json.loads((prepared/'protocol.json').read_text())
    receipt=json.loads((results/'complete.json').read_text())
    frame=pd.read_csv(analysis/'crossed_intervals.csv',keep_default_na=False,na_values=[''])
    output=ROOT/'reports/v10_detail'/run_id;output.mkdir(parents=True,exist_ok=False)
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in
        (prepared/'protocol.json',results/'complete.json',analysis/'complete.json',analysis/'crossed_intervals.csv',verified/'complete.json',project_path(__file__))}
    for metric,filename,title in [('target_hit','named_judge_hits.png','Target-hit rate'),
                                  ('paired_hit_delta','named_judge_target_minus_null.png','Target-minus-null hit difference')]:
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for row,axis in enumerate(('manufacturer','field_strength')):
            for j in range(3):
                ax=axes[row,j];judge=f'heldout_orientation_robust_seed{j}'
                data=frame[(frame.axis==axis)&(frame.guide_control=='correct')&(frame.judge==judge)&
                    (frame.variant=='target')&(frame.stratum=='all')&(frame.metric==metric)].set_index('mode')
                for x,mode in enumerate(MODES):
                    r=data.loc[mode]
                    ax.errorbar(x,100*r.estimate,yerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],
                        fmt='o',capsize=4,color=COLORS[x])
                ax.set_xticks(range(3),LABELS,rotation=15,fontsize=8)
                ax.set_title(('Manufacturer' if axis=='manufacturer' else 'Field strength (exploratory)')+f' / robust seed {j}')
                ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.2)
                if metric=='target_hit':ax.set_ylim(-3,103)
                else:ax.set_ylim(-5,105)
                if j==0:ax.set_ylabel(title+' (%)' if metric=='target_hit' else 'Target minus null\n(percentage points)')
        fig.suptitle(title+' under each heldout_orientation_robust judge\n95% descriptive crossed patient/adapter-replica CIs; reused development patients; clinical preservation unknown')
        fig.savefig(output/filename,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,axis in enumerate(('manufacturer','field_strength')):
        for col,metric in enumerate(('interior_mad','interior_hf_residual_mad')):
            ax=axes[row,col]
            for k,(mode,label,color) in enumerate(zip(MODES,LABELS,COLORS)):
                values=frame[(frame.axis==axis)&(frame['mode']==mode)&(frame.guide_control=='correct')&
                    (frame.judge=='heldout_orientation_robust_seed0')&(frame.stratum=='all')&(frame.metric==metric)].set_index('variant')
                for x,variant in enumerate(('null','target')):
                    r=values.loc[variant]
                    ax.errorbar(x+(k-1)*.13,r.estimate,yerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],
                        fmt='o',capsize=3,color=color,label=label if x==0 else None)
            ax.set_xticks([0,1],['Original-label null','Target']);ax.grid(axis='y',alpha=.2)
            ax.set_title(axis.replace('_',' ').title()+' / '+metric.replace('interior_','').replace('_',' '))
            ax.legend(fontsize=8)
    fig.suptitle('Strict source-proxy-interior correspondence\n95% descriptive crossed CIs; source-mask margin 8px; these are not clinical preservation scores')
    fig.savefig(output/'interior_correspondence.png',dpi=150);plt.close(fig)
    generation=p['evaluation']['selection_generation_seed']
    for seed in p['training_seeds']:
        requests={};homes={}
        for mode in MODES:
            key=f'seed{seed}__{mode}__manufacturer__correct'
            homes[mode]=project_path(receipt['evaluations'][key])
            requests[mode]=pd.read_csv(homes[mode]/'requests.csv')
            requests[mode]=requests[mode][requests[mode].generation_seed.eq(generation)].set_index('patient_id').sort_index()
            path=homes[mode]/'requests.csv';inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        patients=requests[MODES[0]].index.tolist()
        if len(patients)!=15 or any(list(t.index)!=patients for t in requests.values()):raise ValueError('Matched complete all-patient panel required')
        fig,axes=plt.subplots(15,7,figsize=(14,29),squeeze=False)
        for i,patient in enumerate(patients):
            source=None
            for k,mode in enumerate(MODES):
                request=requests[mode].loc[patient];path=homes[mode]/request.array_file
                with np.load(path) as z:
                    if source is None:source=z['source'].copy()
                    else:np.testing.assert_array_equal(source,z['source'])
                    for offset,control in enumerate(('null','target')):
                        axes[i,1+2*k+offset].imshow(z[control],cmap='gray',vmin=0,vmax=1)
                inputs[path.relative_to(ROOT).as_posix()]=digest(path)
            axes[i,0].imshow(source,cmap='gray',vmin=0,vmax=1)
            q=requests[MODES[0]].loc[patient]
            marks=(' Q' if q.qc_warning else '')+(' O' if q.condition_support_status=='outside_target_echo_range' else '')
            axes[i,0].set_ylabel(patient+marks,fontsize=7)
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        for ax,title in zip(axes[0],['Source',*[label+' / '+control for label in LABELS for control in ('null','target')]]):
            ax.set_title(title,fontsize=8)
        fig.suptitle(f'Every manufacturer validation source / adapter training seed {seed}\nFixed generation seed {generation}, t=300 w=3; Q=source-mask warning, O=outside condition range; clinical preservation unknown')
        fig.tight_layout(rect=(0,0,1,.975));fig.savefig(output/f'all_patients_seed{seed}.png',dpi=115);plt.close(fig)
    json_write(output/'inputs.json',inputs)
    (output/'README.md').write_text('Producer: scripts.v10_detail.plot_results_v2. named_judge_hits.png and named_judge_target_minus_null.png: '
        'every named robust judge, both axes, all correct-guide modes with descriptive crossed CIs; interior_correspondence.png: '
        'null/target strict source-mask metrics; all_patients_seed*.png: every validation patient for each adapter training seed '
        'at the prospectively fixed selection generation seed, same [0,1] display range; inputs.json: exact consumed hashes; '
        'complete.json: verified-data-only scope. Simultaneous advancement-gate intervals are in the analysis JSON, not these '
        'descriptive plots. No visual clinical truth or default promotion is claimed.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,all_named_judges=True,all_validation_patients=True,
        clinical_preservation='unknown',default_promoted=False,illustration_selection='Prospectively fixed generation seed; all patients and adapter replicas'))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','analysis','verified','run-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();plot(a.prepared,a.results,a.analysis,a.verified,a.run_id)
