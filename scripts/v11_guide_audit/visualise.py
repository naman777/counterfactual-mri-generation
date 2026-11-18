"""All-judge control charts and all-patient panels, without favourable selection."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path


def row(frame,**conditions):
    selected=frame
    for key,value in conditions.items():selected=selected[selected[key]==value]
    if len(selected)!=1:raise ValueError('Missing/duplicate plotted interval')
    return selected.iloc[0]


def plot(analysis,probe,run_id):
    analysis,probe=map(project_path,(analysis,probe))
    assert json.loads((analysis/'complete.json').read_text())['complete']
    assert json.loads((probe/'complete.json').read_text())['complete']
    output=ROOT/'reports/v11_guide_audit'/run_id;output.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(analysis/'contrast_intervals.csv');f=f[(f.axis=='manufacturer')&(f.stratum=='all')]
    response=pd.read_csv(probe/'paired_intervals.csv')
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in
        (analysis/'contrast_intervals.csv',analysis/'complete.json',probe/'paired_intervals.csv',probe/'complete.json',project_path(__file__))}
    colors=('#496a8a','#b45434','#777777')
    pairs=[(mode,control) for mode in ('coarse_control','coarse_fine') for control in ('zero','absent','wrong','coarse_only','fine_only')]
    fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True,layout='constrained')
    for j,ax in enumerate(axes):
        for y,(mode,control) in enumerate(pairs):
            r=row(f,contrast=f'{mode}__correct_minus_{control}',judge=f'heldout_orientation_robust_seed{j}',metric='paired_hit_delta')
            color=colors[0 if mode=='coarse_control' else 1]
            ax.hlines(y,100*r.ci_low,100*r.ci_high,color=color);ax.scatter(100*r.estimate,y,color=color,s=22)
        ax.axvline(0,color='black',linewidth=.7);ax.grid(axis='x',alpha=.2);ax.set_xlim(-65,65)
        ax.set_title(f'Robust judge seed {j}');ax.set_xlabel('Editing difference (percentage points)')
    axes[0].set_yticks(range(len(pairs)),[('Coarse' if m=='coarse_control' else 'Fine')+' / full minus '+c for m,c in pairs],fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('All declared within-model guide controls / heldout_orientation_robust judges\nDescriptive 95% crossed CIs; correct minus control; reused development; wrong/split guides are stress tests')
    fig.savefig(output/'all_control_contrasts.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True,layout='constrained')
    for j,ax in enumerate(axes):
        for x,(metric,label,sign) in enumerate([('target_hit','Target hit',1),('null_hit','Minus null hit',-1),('paired_hit_delta','Net editing',1)]):
            r=row(f,contrast='fine_minus_coarse__correct',judge=f'heldout_orientation_robust_seed{j}',metric=metric)
            lo,hi=sorted([100*sign*r.ci_low,100*sign*r.ci_high])
            ax.vlines(x,lo,hi,color=colors[x]);ax.scatter(x,100*sign*r.estimate,color=colors[x])
        ax.set_xticks(range(3),['Target hit','Minus null hit','Net editing'],rotation=15,fontsize=8)
        ax.set_title(f'Robust judge seed {j}');ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Fine minus coarse (percentage points)')
    fig.suptitle('Decomposing primary manufacturer editing change\nDescriptive 95% crossed CIs; net editing = target hit minus original-label null hit; original gates stay unchanged')
    fig.savefig(output/'primary_decomposition.png',dpi=140);plt.close(fig)
    controls=['correct','zero','absent','wrong','coarse_only','fine_only']
    metrics=['response_ratio_to_parent','response_cosine_to_parent','unconditional_adapter_shift_rms']
    labels=['Response RMS / parent RMS','Response cosine to parent','Unconditional adapter shift RMS']
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharey='row',layout='constrained')
    for col,t in enumerate((100,300)):
        for k,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axes[k,col]
            for x,control in enumerate(controls):
                r=row(response,timestep=t,guide_control=control,metric=metric)
                ax.vlines(x,r.ci_low,r.ci_high,color=colors[1]);ax.scatter(x,r.estimate,color=colors[1],s=20)
            ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.2)
            ax.set_xticks(range(6),controls,rotation=25,fontsize=8);ax.set_title(f't={t} / {label}',fontsize=10)
    fig.suptitle('Local one-step manufacturer-response differences: fine minus coarse\nDescriptive 95% crossed CIs; same q(source,t,noise), one fixed noise seed; not a reverse trajectory or a hit-rate result')
    fig.savefig(output/'forward_responses.png',dpi=140);plt.close(fig)
    protocol=json.loads((analysis/'protocol.json').read_text())
    receipt=json.loads(project_path(protocol['previous_results']).read_text())
    old=json.loads(project_path(protocol['previous_protocol']).read_text())
    generation=old['evaluation']['selection_generation_seed']
    for seed in old['training_seeds']:
        routes={};homes={}
        for mode,control in [('coarse_control','correct'),*[('coarse_fine',c) for c in controls]]:
            key=f'seed{seed}__{mode}__manufacturer__{control}';home=project_path(receipt['evaluations'][key])
            r=pd.read_csv(home/'requests.csv');r=r[r.generation_seed==generation].set_index('patient_id').sort_index()
            routes[mode,control]=r;homes[mode,control]=home
            inputs[(home/'requests.csv').relative_to(ROOT).as_posix()]=digest(home/'requests.csv')
        patients=routes['coarse_fine','correct'].index.tolist()
        if len(patients)!=15:raise ValueError('All patients required')
        loaded={};source_lookup={}
        for key,requests in routes.items():
            if requests.index.tolist()!=patients:raise ValueError('Unpaired panel patients')
            for patient,r in requests.iterrows():
                path=homes[key]/r.array_file
                if digest(path)!=r.array_sha256:raise ValueError('Panel array changed')
                with np.load(path) as z:arrays={name:z[name].copy() for name in ('source','null','target')}
                loaded[key,patient]=arrays;source_lookup[int(r.dataset_index)]=arrays['source']
                inputs[path.relative_to(ROOT).as_posix()]=r.array_sha256
        for kind in ('primary','controls'):
            columns=7 if kind=='primary' else 8
            fig,axes=plt.subplots(15,columns,figsize=(2*columns,29),squeeze=False)
            last=None
            for i,patient in enumerate(patients):
                fine=loaded[('coarse_fine','correct'),patient];coarse=loaded[('coarse_control','correct'),patient]
                np.testing.assert_array_equal(fine['source'],coarse['source'])
                r=routes['coarse_fine','correct'].loc[patient]
                if kind=='primary':
                    images=[fine['source'],coarse['null'],coarse['target'],fine['null'],fine['target'],
                        np.abs(coarse['target']-fine['source']),np.abs(fine['target']-fine['source'])]
                else:
                    wrong=routes['coarse_fine','wrong'].loc[patient]
                    images=[fine['source'],source_lookup[int(wrong.guide_dataset_index)],*[loaded[('coarse_fine',c),patient]['target'] for c in controls]]
                for j,im in enumerate(images):
                    difference=kind=='primary' and j>=5
                    last=axes[i,j].imshow(im,cmap='magma' if difference else 'gray',vmin=0,vmax=.3 if difference else 1)
                    axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                mark=(' Q' if bool(r.qc_warning) else '')+(' O' if r.condition_support_status=='outside_target_echo_range' else '')
                axes[i,0].set_ylabel(patient+mark,fontsize=7)
            headers=(['Source','Coarse / null','Coarse / target','Fine / null','Fine / target','|Coarse target-source|','|Fine target-source|']
                if kind=='primary' else ['Source','Wrong guide source',*['Fine / '+c for c in controls]])
            for ax,label in zip(axes[0],headers):ax.set_title(label,fontsize=7)
            fig.suptitle(f'All manufacturer patients / adapter seed {seed} / generation seed {generation}\n'
                f'{kind.title()} comparison; Q=source-mask warning, O=outside target range; clinical preservation unknown',fontsize=10,y=.985)
            fig.subplots_adjust(left=.065,right=.995,bottom=.03,top=.95,wspace=.035,hspace=.08)
            if kind=='primary':
                bar=fig.add_axes([.745,.012,.24,.006]);fig.colorbar(last,cax=bar,orientation='horizontal').set_label('Absolute difference (fixed 0–0.3; larger values saturate)',fontsize=7)
                bar.tick_params(labelsize=6)
            fig.savefig(output/f'all_patients_{kind}_seed{seed}.png',dpi=110);plt.close(fig)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,all_judges_controls=True,all_patient_panels=6,
        fixed_generation_seed=generation,clinical_preservation='unknown',paid_gpu_calls=0))
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.visualise. all_control_contrasts.png: all within-model '
        'guide-control editing contrasts and named judges; primary_decomposition.png: target/null contributions; '
        'forward_responses.png: all one-step response controls/timesteps; all_patients_primary_seed*.png: every patient '
        'and training seed with null/target/source differences (fixed difference range 0–0.3); '
        'all_patients_controls_seed*.png: every correct/zero/absent/wrong/split target and actual wrong-guide source; '
        'inputs.json: exact plotted data/arrays; complete.json: scope. CIs are exploratory/descriptive, not old advancement intervals.\n',encoding='utf-8')
    print('ALL DIAGNOSTIC FIGURES COMPLETE',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for n in ('analysis','probe','run-id'):a.add_argument('--'+n,required=True)
    q=a.parse_args();plot(q.analysis,q.probe,q.run_id)
