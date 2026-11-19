"""Readable all-judge effects, every-patient contributions and actual guide exposure."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path
from scripts.v14_dropout_audit.common import warnings


def run(prepared,cases,exposure,probe,run_id):
    prepared,cases,exposure,probe=map(project_path,(prepared,cases,exposure,probe))
    p=json.loads((prepared/'protocol.json').read_text());out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.visualise. effects.png: '
        'all three named-judge direct/interaction effects with descriptive 95% CIs; patients.png: '
        'every source patient and training seed (generation means), fixed ordering and warnings; '
        'exposure.png: observed process frequencies for unique matched training streams; '
        'epsilon_response.png: fixed-source surrogate effects with descriptive CIs; inputs.json: '
        'plotted data hashes; complete.json: production only, visual inspection still required.\n',encoding='utf-8')
    f=pd.read_csv(cases/'contrast_intervals.csv');s=f[(f.axis=='manufacturer')&(f.stratum=='all')&(f.metric=='paired_hit_delta')]
    contrasts=[('coarse_fine__dropout_minus_old__correct','Fine guide: dropout − old'),
               ('coarse_control__dropout_minus_old__correct','Coarse guide: dropout − old'),
               ('guide_effect_interaction__correct','Change in fine − coarse difference')]
    judges=[f'heldout_orientation_robust_seed{j}' for j in range(3)]
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for ax,(name,title) in zip(axes,contrasts):
        for j,judge in enumerate(judges):
            row=s[(s.contrast==name)&(s.judge==judge)].iloc[0]
            ax.errorbar(100*row.estimate,j,xerr=[[100*(row.estimate-row.ci_low)],[100*(row.ci_high-row.estimate)]],
                        fmt='o',color='#285a70',capsize=4)
        ax.axvline(0,color='#777777',linewidth=.8);ax.set_yticks(range(3),judges if ax is axes[0] else ['seed0','seed1','seed2'],fontsize=8)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Editing difference (percentage points)');ax.grid(axis='x',alpha=.15)
        ax.invert_yaxis()
    fig.suptitle('Exploratory paired effects; 95% descriptive crossed intervals',fontsize=12)
    fig.savefig(out/'effects.png',dpi=160);plt.close(fig)
    patients=pd.read_csv(cases/'primary_patients.csv');patients['qc_warning']=warnings(patients.qc_warning)
    selected=pd.read_csv(project_path(p['selection']));order=selected.patient_id.tolist();seeds=p['training_seeds']
    warning_ids=set(patients.loc[patients.qc_warning,'patient_id'])
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    for j,judge in enumerate(judges):
        for col,(name,title) in enumerate((('coarse_fine__dropout_minus_old__correct','New fine − cached old fine'),
                                         ('guide_dropout__fine_minus_coarse__correct','New fine − new coarse'))):
            part=patients[(patients.axis=='manufacturer')&(patients.judge==judge)&(patients.contrast==name)]
            values=part.pivot(index='patient_id',columns='training_seed',values='paired_hit_delta').reindex(index=order,columns=seeds)
            if values.isna().any().any():raise ValueError('A patient/seed would be omitted from the figure')
            ax=axes[j,col];im=ax.imshow(values,aspect='auto',vmin=-1,vmax=1,cmap='BrBG',interpolation='nearest')
            ax.set_yticks(range(15),[pid+(' *' if pid in warning_ids else '') for pid in order],fontsize=7)
            ax.set_xticks(range(3),seeds,fontsize=8);ax.set_title(title+'\n'+judge,fontsize=9)
            for row in range(15):
                for column in range(3):
                    ax.text(column,row,f'{values.iloc[row,column]:+.2f}',ha='center',va='center',fontsize=7,
                            color='white' if abs(values.iloc[row,column])>.6 else '#202020')
    fig.colorbar(im,ax=axes,shrink=.6,label='Observed target-minus-null editing difference; mean over generation seeds')
    fig.suptitle('All source patients; * denotes a retained QC warning',fontsize=12)
    fig.savefig(out/'patients.png',dpi=160);plt.close(fig)
    states=pd.read_csv(exposure/'state_exposure.csv');states=states[states['mode']=='coarse_control']
    names=['full','coarse_only','second_block_only','zero_input','absent_adapter'];positions=np.arange(5)
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    for j,seed in enumerate(seeds):
        part=states[states.training_seed==seed].set_index('state').reindex(names)
        ax.bar(positions+(j-1)*.23,part.observed_fraction,width=.22,label=str(seed))
    ax.plot(positions,[.4,.2,.2,.1,.1],'k_',markersize=24,label='Intended probability')
    ax.set_xticks(positions,names,fontsize=9);ax.set_ylabel('Fraction of all actual training samples')
    ax.set_title('Actual guide states; each fine arm has identical matched counts')
    ax.legend(ncol=4,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.savefig(out/'exposure.png',dpi=160);plt.close(fig)
    e=pd.read_csv(probe/'paired_intervals.csv');e=e[(e.timestep==300)&(e.guide_control=='correct')]
    terms=[('old_fine_minus_old_coarse','Old fine − old coarse'),('new_fine_minus_new_coarse','New fine − new coarse'),
           ('new_fine_minus_old_fine','New fine − old fine'),('new_coarse_minus_old_coarse','New coarse − old coarse'),
           ('guide_effect_interaction','Change in fine − coarse')]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for ax,metric,title in zip(axes,['response_ratio_to_parent','response_cosine_to_parent'],
                               ['Label-response / parent-RMS ratio','Label-response alignment to parent']):
        for j,(name,label) in enumerate(terms):
            row=e[(e.contrast==name)&(e.metric==metric)].iloc[0]
            ax.errorbar(row.estimate,j,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',color='#285a70',capsize=4)
        ax.axvline(0,color='#777777',linewidth=.8);ax.set_yticks(range(5),[label for _,label in terms],fontsize=8)
        ax.set_title(title,fontsize=10);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_xlabel('Difference, descriptive 95% crossed CI')
    fig.suptitle('Fixed noisy source at t=300; surrogate epsilon response, not editing success',fontsize=11)
    fig.savefig(out/'epsilon_response.png',dpi=160);plt.close(fig)
    inputs=[cases/'contrast_intervals.csv',cases/'primary_patients.csv',exposure/'state_exposure.csv',
            probe/'paired_intervals.csv',project_path(p['selection']),project_path(__file__)]
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in inputs})
    json_write(out/'complete.json',dict(complete=True,figures=4,all_source_patients_shown=True,visual_inspection_pending=True))
    print('DIAGNOSTIC FIGURES PRODUCED',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__)
    for name in ('prepared','cases','exposure','probe','run-id'):a.add_argument('--'+name,required=True)
    run(**vars(a.parse_args()))
