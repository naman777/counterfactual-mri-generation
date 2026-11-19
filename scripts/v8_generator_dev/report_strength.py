"""Report every frozen guide-amplitude profile without selecting a default."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.report_adapter import single
from scripts.v8_generator_dev.report_continuation import interval


def report(fetched,analysis_prefix,run_id):
    fetched=project_path(fetched)
    base=ROOT/'reports/v8_generator_dev'
    verified=base/(analysis_prefix+'_verification')
    compared=base/(analysis_prefix+'_comparison')
    if not json.loads((verified/'complete.json').read_text())['saved_validation_verified']:
        raise ValueError('Strict saved-array verification required')
    if not json.loads((compared/'complete.json').read_text())['complete']:
        raise ValueError('Completed prospective paired profiles required')
    verify_inputs(json.loads((verified/'inputs.json').read_text()))
    verify_inputs(json.loads((compared/'inputs.json').read_text()))
    receipt=json.loads((fetched/'complete.json').read_text())
    prepared=project_path(receipt['prepared'])
    config=json.loads((prepared/'protocol.json').read_text())['parent_config']
    estimates=pd.read_csv(verified/'estimates.csv',keep_default_na=False)
    parent_path=base/'continuation_final_20261117_01_comparison/primary_estimates.csv'
    parents=pd.read_csv(parent_path,keep_default_na=False)
    rows=[]
    keep=('target_hit','paired_hit_delta','ssim','mad','other_axis_stable')
    for backbone in ('additive','scale_shift'):
        parent=parents[(parents.model==backbone)&(parents.stratum=='all_flagged_and_in_range')&parents.metric.isin(keep)].copy()
        parent=parent.rename(columns={'control':'variant'})
        parent['backbone']=backbone
        parent['guide_strength']=0.
        parent['configuration']='parent_alpha000'
        parent['stratum']='all'
        rows.append(parent)
        sources=[(backbone+'_alpha'+str(int(a*100)).zfill(3),a,'source_guide') for a in (.25,.5,.75)]
        sources += [(backbone+'__soft_source_guide__correct',1.,'source_guide'),
                    (backbone+'__learned_zero_guide__correct','', 'learned_zero_guide')]
        for key,alpha,label in sources:
            part=estimates[(estimates.evaluation==key)&(estimates.stratum=='all')&estimates.metric.isin(keep)].copy()
            if part.empty:
                raise ValueError('Missing complete profile: '+key)
            part['backbone']=backbone
            part['guide_strength']=alpha
            part['configuration']=label
            rows.append(part)
    profiles=pd.concat(rows,ignore_index=True)
    columns=['backbone','configuration','guide_strength','axis','judge','variant','metric',
             'patients','mean','ci95_lo','ci95_hi']
    profiles=profiles[columns]
    output=base/run_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.report_strength. profiles.csv binds all cached endpoints, '
        'three prospective strengths and learned-zero controls to their actual patient CIs. '
        'conditioning_profile.png/pdf shows every named judge; correspondence_profile.png/pdf '
        'shows SSIM once because it is judge-independent. RESULTS.md explains scope and all '
        'profiles without picking a default. support.json retains eligibility counts. budget.json '
        'separates the new allowance estimate from unknown invoice. inputs.json binds evidence/code. '
        'complete.json is report completion, not clinical validation or model promotion.\n',encoding='utf-8')
    csv_write(output/'profiles.csv',profiles)
    launch=base/'strength_launch_20261117_01/process_result.json'
    process=json.loads(launch.read_text())
    budget=dict(approved_additional_usd=3,previous_allowances_reset=False,cli_wall_seconds=process['seconds'],
        rate_with_margin=.000441264,reserve_usd=.5,conservative_estimated_usd=process['seconds']*.000441264+.5,
        exact_invoice_known=False,training_ran=False)
    json_write(output/'budget.json',budget)
    alpha_values=[0.,.25,.5,.75,1.]
    judge_names=[f'heldout_orientation_robust_seed{seed}' for seed in range(3)]
    def get(backbone,axis,judge,alpha,metric):
        return single(profiles,backbone=backbone,axis=axis,judge=judge,variant='target',
            guide_strength=alpha,metric=metric)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    target_deltas=profiles[(profiles.variant=='target')&(profiles.metric=='paired_hit_delta')]
    lower_limit=min(-5.,float(target_deltas.ci95_lo.min())*100-3.)
    for column,backbone in enumerate(('additive','scale_shift')):
        for row,axis in enumerate(('manufacturer','field_strength')):
            ax=axes[row,column]
            for judge,color in zip(judge_names,('#245e87','#ad642f','#547a47')):
                data=[get(backbone,axis,judge,alpha,'paired_hit_delta') for alpha in alpha_values]
                mean=np.array([part['mean'] for part in data],float)*100
                low=np.array([part['ci95_lo'] for part in data],float)*100
                high=np.array([part['ci95_hi'] for part in data],float)*100
                ax.errorbar(alpha_values,mean,yerr=[mean-low,high-mean],marker='o',capsize=3,
                    color=color,label=judge.replace('heldout_orientation_robust_',''),linewidth=1.2,markersize=4)
                zero=single(profiles,backbone=backbone,axis=axis,judge=judge,variant='target',
                    configuration='learned_zero_guide',metric='paired_hit_delta')
                ax.axhline(zero['mean']*100,color=color,linestyle=':',linewidth=1,alpha=.75)
            ax.set_title(backbone.replace('_',' ')+' / '+axis.replace('_',' '))
            ax.set_xticks(alpha_values)
            ax.set_ylim(lower_limit,105)
            ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
            if column==0:
                ax.set_ylabel('Target minus null hit rate (percentage points)')
            if row==1:
                ax.set_xlabel('Source-guide residual strength')
    axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
    fig.suptitle('Frozen-weight source-guide strength sensitivity',fontsize=13)
    fig.text(.5,.025,'Solid: source-guide profiles (0 = cached parent, 1 = cached full guide). Dotted: learned-zero-guide mean.\n'
        'All heldout_orientation_robust_seed{0,1,2} judges; 15 reused patients, three generation seeds; 95% patient CIs, unadjusted.',
        ha='center',fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,top=.90,bottom=.14,hspace=.28,wspace=.15)
    fig.savefig(output/'conditioning_profile.png',dpi=180)
    fig.savefig(output/'conditioning_profile.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7.2),sharex=True)
    for column,backbone in enumerate(('additive','scale_shift')):
        for row,axis in enumerate(('manufacturer','field_strength')):
            ax=axes[row,column]
            data=[get(backbone,axis,judge_names[0],alpha,'ssim') for alpha in alpha_values]
            mean=np.array([part['mean'] for part in data],float)
            low=np.array([part['ci95_lo'] for part in data],float)
            high=np.array([part['ci95_hi'] for part in data],float)
            ax.errorbar(alpha_values,mean,yerr=[mean-low,high-mean],fmt='o-',color='#245e87',capsize=4)
            zero=single(profiles,backbone=backbone,axis=axis,judge=judge_names[0],variant='target',
                configuration='learned_zero_guide',metric='ssim')
            ax.axhline(zero['mean'],linestyle=':',color='#777777',label='Learned zero guide')
            ax.set_title(backbone.replace('_',' ')+' / '+axis.replace('_',' '))
            ax.set_xticks(alpha_values)
            ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
            if column==0:
                ax.set_ylabel('Target SSIM with source')
            if row==1:
                ax.set_xlabel('Source-guide residual strength')
    fig.suptitle('Image correspondence profile; clinical preservation unknown',fontsize=13)
    fig.text(.5,.025,'SSIM is judge-independent; one copy is plotted. 95% equal-patient bootstrap CIs.\n'
        '15 reused development patients; cached endpoints and unchanged fixed editing settings; no independent clinical truth.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.15,hspace=.30,wspace=.22)
    fig.savefig(output/'correspondence_profile.png',dpi=180)
    fig.savefig(output/'correspondence_profile.pdf')
    plt.close(fig)
    first_home=project_path(next(iter(receipt['evaluations'].values())))
    requests=pd.read_csv(first_home/'requests.csv')
    eligible=requests[requests.condition_support_status.isin(config['evaluation']['selection_support_statuses'])]
    support={axis:{direction:int(eligible[(eligible.axis==axis)&(eligible.direction==direction)].patient_id.nunique())
        for direction in ('0_to_1','1_to_0')} for axis in ('manufacturer','field_strength')}
    json_write(output/'support.json',dict(eligible_patients_by_axis_direction=support,
        required_per_direction=config['evaluation']['minimum_eligible_patients_per_direction'],
        eligibility_unchanged=True,selection_performed=False))
    lines=['# Prospective lower-guide-strength sensitivity','',
        'Both learned backbones, acquisition conditioners and adapters were frozen. This adds '
        'six prospective inference-only passes (540 cases) and reuses the original parent/full-guide '
        'endpoints and learned-zero-guide controls. No training or new checkpoint occurred.','',
        'Alpha scales the spatial adapter residuals inside the denoiser. It is distinct from '
        'classifier-free acquisition guidance. Fixed editing points: '+
        '; '.join(f'{axis}: t={point["timestep"]}, w={point["guidance"]}' for axis,point in
            config['evaluation']['fixed_settings'].items())+'. Source EchoTime stays fixed. '
        'The learned zero-guide baseline is a separately trained control, not the alpha-zero parent. '
        'Every interior amplitude uses actual sampled outputs; plotted lines only join measured settings.','',
        'A null output is denoised under the original acquisition labels; a target output changes '
        'one acquisition label. Both use the same per-case noise seed. Identity is the unchanged '
        'source image. The plotted target-minus-null quantity is a difference in named-judge '
        'hit rates, not a clinical success rate.','',
        'The strict local verifier rechecked all 540 new cases together with 720 original controls '
        '(1,260 cases total). Source/guide identity, fixed EchoTime, support, SSIM/MAD and every '
        'named-judge decision must agree. The original training audit is reused and explicitly '
        'labelled; it is not new training.','',
        'The same fifteen development patients and three generation seeds are reused. Values '
        'are equal-patient estimates [95% patient-bootstrap CI], not multiplicity adjusted. '
        'The underlying adapters have one training seed per copy; generation and judge seeds '
        'do not estimate training-seed variability. Target-minus-null differences below are '
        'percentages / percentage points.','']
    for backbone in ('additive','scale_shift'):
        for axis in ('manufacturer','field_strength'):
            lines += ['## '+backbone.replace('_',' ')+' / '+axis.replace('_',' '),'',
                'Judge columns are `heldout_orientation_robust_seed0`, `seed1`, `seed2`. '
                'SSIM is image correspondence, not verified anatomy.','',
                '| Configuration | Target SSIM | Judge 0 target minus null | Judge 1 target minus null | Judge 2 target minus null |',
                '|---|---|---|---|---|']
            for alpha in alpha_values:
                values=[interval(get(backbone,axis,judge,alpha,'paired_hit_delta')) for judge in judge_names]
                label='Parent (alpha 0)' if alpha==0 else 'Full guide (alpha 1)' if alpha==1 else f'Guide alpha {alpha:.2f}'
                lines.append('| '+label+' | '+interval(get(backbone,axis,judge_names[0],alpha,'ssim'),False)+' | '+' | '.join(values)+' |')
            zero=profiles[(profiles.backbone==backbone)&(profiles.axis==axis)&(profiles.variant=='target')&
                (profiles.configuration=='learned_zero_guide')]
            lines.append('| Learned zero-guide adapter | '+interval(single(zero,judge=judge_names[0],metric='ssim'),False)+' | '+
                ' | '.join(interval(single(zero,judge=judge,metric='paired_hit_delta')) for judge in judge_names)+' |')
            lines.append('')
    lines += ['## Interpretation and limits','',
        'Read the conditioning and correspondence profiles jointly. A higher SSIM alone can '
        'reflect retention of source appearance; a classifier hit alone does not verify anatomy '
        'or a physical scanner effect. Raw target-hit rates, nonedited-axis stability, all '
        'direction/support strata and paired changes versus all three references remain in '
        'the companion CSVs. Every prospective strength is retained, including worse results.','',
        'The preserved structural screen uses the identical earlier thresholds and metric code. '
        'Its results are in the verification folder; it has known false flags and limited '
        'tiny-erasure sensitivity. Clinical anatomy and lesion preservation remain unknown.','',
        'No strength, model or default is automatically promoted. Existing eligible-patient '
        'coverage rules remain unchanged (support.json); the comparison folder also applies '
        'the original provisional T/w feasibility criteria separately at each alpha, using '
        'only the original selection seed, without selecting an alpha. A strength parameter cannot supply '
        'missing supported patients. Repeated validation reuse is development, not a fresh '
        'cohort or independent clinical confirmation. Historical publication outputs are unchanged.','',
        f'Separate allowance: US$3. Conservative whole-CLI estimate US${budget["conservative_estimated_usd"]:.2f} '
        'including the frozen rate margin and US$0.50 reserve, not an invoice. Previous allowances '
        'remain separate; no paid restart or training was used.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    evidence=[verified/'complete.json',verified/'inputs.json',verified/'estimates.csv',
        verified/'preservation_intervals.csv',compared/'complete.json',compared/'paired_strength_differences.csv',
        parent_path,launch,first_home/'requests.csv',prepared/'protocol.json',
        compared/'provisional_eligibility.json',ROOT/'scripts/v8_generator_dev/report_strength.py']
    json_write(output/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in evidence})
    json_write(output/'complete.json',dict(complete=True,prospective_strengths=[.25,.5,.75],new_cases=540,
        reused_reference_cases=720,training_ran=False,new_checkpoint_written=False,
        model_promoted=False,clinical_anatomy_validated=False))
    print('STRENGTH REPORT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--analysis-prefix',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    report(a.fetched,a.analysis_prefix,a.run_id)
