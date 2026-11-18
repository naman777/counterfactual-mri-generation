"""Traceable pilot report and predetermined matched MRI panels, with all controls."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.report_continuation import interval


def single(frame,**conditions):
    selected=frame
    for name,value in conditions.items():
        selected=selected[selected[name]==value]
    if len(selected)!=1:
        raise ValueError('Missing/duplicate report source: '+str(conditions))
    return selected.iloc[0]


def report(fetched,analysis,parent_fetch,run_id):
    fetched,analysis,parent_fetch=map(project_path,(fetched,analysis,parent_fetch))
    complete=json.loads((analysis/'complete.json').read_text())
    if not complete['saved_validation_verified'] or not complete['absent_guide_equals_parent_exact']:
        raise ValueError('Completed independent verification and parent equality required')
    verify_inputs(json.loads((analysis/'inputs.json').read_text()))
    receipt=json.loads((fetched/'complete.json').read_text())
    prepared=project_path(receipt['prepared'])
    protocol=json.loads((prepared/'protocol.json').read_text())
    config=protocol['parent_config']
    estimates=pd.read_csv(analysis/'estimates.csv',keep_default_na=False)
    comparisons=pd.read_csv(analysis/'comparisons.csv',keep_default_na=False)
    preservation=pd.read_csv(analysis/'preservation_intervals.csv',keep_default_na=False)
    parent_comparison=ROOT/'reports/v8_generator_dev/continuation_final_20261117_01_comparison/primary_estimates.csv'
    parent_estimates=pd.read_csv(parent_comparison,keep_default_na=False)
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.report_adapter. RESULTS.md gives every named-judge '
        'parent/zero/source-guide rate, paired guide effects, correspondence and fixed proxy limits. '
        'headline_estimates.csv/paired_changes.csv retain exact source rows; budget.json separates '
        'this allowance estimate from the unknown invoice. matched_*.png/pdf show the first two '
        'original preselected patients at the first frozen generation seed, both axes, with no '
        'outcome-based example selection. example_selection.csv binds each displayed raw array. '
        'inputs.json hashes evidence/code. complete.json is reporting completion, not model promotion.\n',encoding='utf-8')
    primary=estimates[estimates.stratum=='all']
    paired=comparisons[comparisons.stratum=='all_flagged_and_in_range']
    csv_write(output/'headline_estimates.csv',primary)
    csv_write(output/'paired_changes.csv',paired)
    launch=ROOT/'reports/v8_generator_dev/adapter_launch_20261117_01/process_result.json'
    process=json.loads(launch.read_text())
    budget=dict(approved_additional_usd=8,previous_allowances_reset=False,
        cli_wall_seconds=process['seconds'],rate_with_margin=.000441264,reserve_usd=.5,
        conservative_estimated_usd=process['seconds']*.000441264+.5,exact_invoice_known=False,
        method='Whole CLI wall at the frozen 20%-margin rate plus reserve; no paid restart or extra app')
    json_write(output/'budget.json',budget)
    grade_checks=pd.read_csv(analysis/'grading_checks.csv')
    training=pd.read_csv(analysis/'training_audit.csv')
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in (
        analysis/'complete.json',analysis/'inputs.json',analysis/'estimates.csv',analysis/'comparisons.csv',
        analysis/'preservation_intervals.csv',analysis/'grading_checks.csv',analysis/'training_audit.csv',
        parent_comparison,launch,prepared/'validation_selection.csv',prepared/'protocol.json',
        ROOT/'scripts/v8_generator_dev/report_adapter.py')}
    lines=['# Source-guide adapter pilot: verified development results','',
        'All four adapter copies completed twenty matched epochs on the frozen common-epoch-130 '
        'backbones. The pilot used only the original training patients; both acquisition '
        'conditioners and all parent weights remained unchanged. All eight validation passes completed.','',
        f'Local verification checked {complete["cases"]} cases, all three frozen judges and '
        f'{int(training.updates.sum())} actual applied training updates across four copies. '
        f'Maximum probability difference on local regrading: {grade_checks.maximum_probability_absolute_error.max():.3g}. '
        'Every discrete decision agrees. Disabling the trained source guide exactly reproduces '
        'the matched parent source/null/target images at the full fixed operating points.','',
        'These are development comparisons on the same fifteen validation patients and three '
        'generation seeds, including flagged condition requests. Every interval is a 95% '
        'equal-patient bootstrap CI; intervals are exploratory and not multiplicity adjusted. '
        'Neither repeated seeds nor multiple judges supply additional independent patients.','',
        '## All named-judge target-hit rates','',
        'Values are percentages [95% CI]. G0 is the learned zero-guide adapter; G is the '
        'source-guide adapter. G target-minus-null tests conditioning above matched source '
        'reconstruction. It is not a clinical success rate.','',
        '| Backbone | Axis | Frozen judge | Parent target | G0 target | G target | G target minus null |',
        '|---|---|---|---|---|---|---|']
    for backbone in ('additive','scale_shift'):
        for axis in ('manufacturer','field_strength'):
            for seed in range(3):
                judge=f'heldout_orientation_robust_seed{seed}'
                common=dict(axis=axis,judge=judge,variant='target')
                parent=single(parent_estimates,model=backbone,axis=axis,judge=judge,control='target',
                    metric='target_hit',stratum='all_flagged_and_in_range')
                zero=single(primary,evaluation=backbone+'__learned_zero_guide__correct',metric='target_hit',**common)
                guided=single(primary,evaluation=backbone+'__soft_source_guide__correct',metric='target_hit',**common)
                delta=single(primary,evaluation=backbone+'__soft_source_guide__correct',metric='paired_hit_delta',**common)
                lines.append(f'| {backbone} | {axis} | {judge} | {interval(parent)} | {interval(zero)} | '
                    f'{interval(guided)} | {interval(delta)} |')
    lines += ['','## Paired source-guide minus learned-zero-guide changes','',
        'All rows use identical patients, source slices, generation seeds and settings. Positive '
        'SSIM change means greater image correspondence, which can also result from copying the '
        'source appearance. It does not establish anatomical validity.','',
        '| Backbone | Axis | Judge | Change in target-minus-null (pp) | Change in target SSIM |',
        '|---|---|---|---|---|']
    for backbone in ('additive','scale_shift'):
        for axis in ('manufacturer','field_strength'):
            for seed in range(3):
                judge=f'heldout_orientation_robust_seed{seed}'
                common=dict(backbone=backbone,axis=axis,judge=judge,control='target',contrast='guide_minus_zero')
                hit=single(paired,metric='paired_hit_delta',**common)
                ssim=single(paired,metric='ssim',**common)
                lines.append(f'| {backbone} | {axis} | {judge} | {interval(hit)} | {interval(ssim,False)} |')
    lines += ['','## Fixed automated proxy envelope','',
        'Values are the percentage of outputs within the earlier training-calibrated mild-transform '
        'envelope [95% empirical patient CI]. No threshold is fitted or retuned here. These are '
        'rank/edge/texture checks with known false flags and limited tiny-erasure sensitivity; '
        'they are not clinical preservation or lesion-sensitivity rates.','',
        '| Backbone | Axis | Guide condition | Target within envelope | Null within envelope |',
        '|---|---|---|---|---|']
    for backbone in ('additive','scale_shift'):
        for axis in ('manufacturer','field_strength'):
            for suffix in ('learned_zero_guide__correct','soft_source_guide__correct','soft_source_guide__wrong','soft_source_guide__absent'):
                name=backbone+'__'+suffix
                target=single(preservation,evaluation=name,axis=axis,control='target',stratum='all')
                null=single(preservation,evaluation=name,axis=axis,control='null',stratum='all')
                lines.append(f'| {backbone} | {axis} | {suffix} | {interval(target)} | {interval(null)} |')
    lines += ['','## Interpretation and remaining limits','',
        'Report joint editing/correspondence behaviour, including any trade-off: a higher SSIM '
        'with lower target-minus-null scores is not a stronger acquisition editor. Wrong-guide '
        'and absent-guide paired contrasts, every named judge, direction and support stratum '
        'remain in the analysis CSVs. The absent guide is an exact software control; wrong '
        'guides are artificial mismatches rather than anatomy labels.','',
        'Clinical anatomy and lesion preservation remain unknown. Guide acquisition leakage '
        'was measured before this pilot and remains unresolved. The original field-strength '
        'operating-point support limitation remains; all new prospective patients stay unassigned '
        'and outside these experiments. There is no fresh cohort, paired real scanner target, '
        'expert review or physical scanner-validation claim.','',
        'No checkpoint, architecture or default editor is promoted automatically. This report '
        'completes the bounded four-copy pilot and verification, not the full strong-generator '
        'development programme or clinical validation. Historical paper/demo outputs remain unchanged.','',
        '## Matched examples and budget','',
        'The separate additive and scale-shift panels show the first two patients from the original '
        'frozen validation selection and the first frozen generation seed, for both axes. Every '
        'panel uses the same intensity display range. Sources were not chosen by a favourable result.','',
        f'Approved separate pilot allowance: US$8. Conservative whole-CLI estimate: '
        f'US${budget["conservative_estimated_usd"]:.2f}, including the frozen rate margin and '
        'US$0.50 reserve. The actual invoice is unknown. No paid restart or additional GPU app '
        'was used, and the app is stopped. Earlier US$6 allowances remain separate.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    homes={key:project_path(path) for key,path in receipt['evaluations'].items()}
    parent_homes={key:project_path(path) for key,path in json.loads((parent_fetch/'complete.json').read_text())['evaluation_folders'].items()}
    selection=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()[:2]
    generation_seed=config['evaluation']['selection_generation_seed']
    examples=[]
    for backbone in ('additive','scale_shift'):
        locations=[('Parent target',parent_homes[backbone],'target'),
            ('Zero-guide target',homes[backbone+'__learned_zero_guide__correct'],'target'),
            ('Guide null',homes[backbone+'__soft_source_guide__correct'],'null'),
            ('Guide target',homes[backbone+'__soft_source_guide__correct'],'target'),
            ('Wrong-guide target',homes[backbone+'__soft_source_guide__wrong'],'target')]
        fig,axes=plt.subplots(4,6,figsize=(13,9.4))
        for row,(axis,index) in enumerate((axis,index) for axis in ('manufacturer','field_strength') for index in selection):
            arrays=[]
            point=config['evaluation']['fixed_settings'][axis]
            for label,home,control in locations:
                requests=pd.read_csv(home/'requests.csv')
                request=single(requests,dataset_index=index,axis=axis,generation_seed=generation_seed,
                    timestep=point['timestep'],guidance=point['guidance'])
                path=home/request.array_file
                with np.load(path,allow_pickle=False) as saved:
                    source=saved['source'].copy()
                    arrays.append(saved[control].copy())
                inputs[path.relative_to(ROOT).as_posix()]=digest(path)
                examples.append(dict(backbone=backbone,axis=axis,dataset_index=index,
                    generation_seed=generation_seed,panel=label,array_file=str(path.relative_to(ROOT)),control=control))
            for column,image in enumerate([source,*arrays]):
                ax=axes[row,column]
                ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                if row==0:
                    ax.set_title((['Source']+[label for label,_,_ in locations])[column],fontsize=10)
                if column==0:
                    ax.set_ylabel(f'{axis.replace("_"," ")}\n{request.patient_id}',fontsize=9)
        fig.suptitle(backbone.replace('_',' ')+' / predetermined matched development examples',fontsize=13)
        fig.text(.5,.025,'128 x 128 saved images; identical [0,1] display range. First two frozen selected patients; first frozen seed.\n'
            'Spatial appearance and classifier outcomes do not establish clinical anatomy or lesion preservation.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.10,right=.99,top=.91,bottom=.09,wspace=.08,hspace=.12)
        fig.savefig(output/('matched_'+backbone+'.png'),dpi=180)
        fig.savefig(output/('matched_'+backbone+'.pdf'))
        plt.close(fig)
    csv_write(output/'example_selection.csv',pd.DataFrame(examples))
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,verified_development_pilot=True,
        outcome_selected_examples=False,model_promoted=False,anatomy_validated=False,fresh_test=False,
        approved_additional_usd=8,conservative_estimated_usd=budget['conservative_estimated_usd']))
    print('ADAPTER REPORT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--analysis',required=True)
    p.add_argument('--parent-fetch',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    report(a.fetched,a.analysis,a.parent_fetch,a.run_id)
