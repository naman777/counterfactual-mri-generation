"""Compose the proven strict saved-array verifier with prospective alpha contrasts."""
import argparse
import json

import pandas as pd

from scripts.v8_generator_dev.analyse_adapter import analyse as verify_all
from scripts.v8_generator_dev.analyse_full import compare_rows, primary_rows
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.full_protocol import select_setting
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs


def analyse(fetched,pilot_fetch,parent_fetch,prefix):
    if not prefix.replace('_','').isalnum():
        raise ValueError('Safe new comparison prefix required')
    fetched,pilot_fetch,parent_fetch=map(project_path,(fetched,pilot_fetch,parent_fetch))
    fresh=json.loads((fetched/'complete.json').read_text())
    original=json.loads((pilot_fetch/'complete.json').read_text())
    if not fresh['workflow_success'] or fresh['training_ran'] or not original['workflow_success']:
        raise ValueError('Completed inference-only sweep and verified original pilot required')
    verify_inputs(json.loads((fetched/'inputs.json').read_text()))
    verify_inputs(json.loads((pilot_fetch/'inputs.json').read_text()))
    base=ROOT/'reports/v8_generator_dev'
    adapter=base/(prefix+'_verifier_inputs')
    adapter.mkdir(exist_ok=False)
    merged=dict(json.loads((pilot_fetch/'inputs.json').read_text()))
    for name,expected in json.loads((fetched/'inputs.json').read_text()).items():
        if name in merged and merged[name]!=expected:
            raise ValueError('Conflicting original/sweep evidence')
        merged[name]=expected
    homes={**original['evaluations'],**fresh['evaluations']}
    expected_keys={backbone+'_alpha'+str(int(alpha*100)).zfill(3)
        for backbone in ('additive','scale_shift') for alpha in (.25,.5,.75)}
    if set(fresh['evaluations'])!=expected_keys:
        raise ValueError('The complete prospectively declared six-profile grid is required')
    for key,relative in fresh['evaluations'].items():
        backbone,encoded=key.rsplit('_alpha',1)
        alpha=int(encoded)/100
        if backbone not in ('additive','scale_shift') or alpha not in (.25,.5,.75):
            raise ValueError('Unexpected prospectively frozen strength identity')
        home=project_path(relative)
        ep=json.loads((home/'protocol.json').read_text())
        requests=pd.read_csv(home/'requests.csv')
        if ep['backbone']!=backbone or ep['adapter_mode']!='soft_source_guide' or ep['guide_strength']!=alpha:
            raise ValueError('Strength/backbone/checkpoint identity mismatch')
        if not requests.guide_strength.eq(alpha).all() or not requests.guide_control.eq('strength').all():
            raise ValueError('Per-case source amplitude drift')
    json_write(adapter/'inputs.json',merged)
    json_write(adapter/'complete.json',dict(complete=True,workflow_success=True,
        final_adapters_verified=original['final_adapters_verified'],training=original['training'],
        prepared=fresh['prepared'],evaluations=homes,training_reused_from_original_pilot=True,
        additional_training_ran=False,clinical_preservation='unknown'))
    (adapter/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.analyse_strength. This path adapter binds six new '
        'inference-only evaluations to eight already verified control runs and the unchanged '
        'original epoch-20 training evidence. No new training is claimed. inputs.json binds '
        'both immutable retrieval ledgers. The strict existing verifier rechecks all fourteen '
        'runs under frozen runtime flags, including exact absent-guide parent equality.\n',encoding='utf-8')
    verification_id=prefix+'_verification'
    verify_all(adapter,parent_fetch,verification_id)
    verified=base/verification_id
    prepared=project_path(fresh['prepared'])
    config=json.loads((prepared/'protocol.json').read_text())['parent_config']
    parents=json.loads((parent_fetch/'complete.json').read_text())['evaluation_folders']
    output=base/(prefix+'_comparison')
    output.mkdir(exist_ok=False)
    rows=[]
    eligibility=[]
    evidence={path.relative_to(ROOT).as_posix():digest(path) for path in (
        fetched/'complete.json',pilot_fetch/'complete.json',verified/'complete.json',
        ROOT/'scripts/v8_generator_dev/analyse_strength.py')}
    for key,relative in fresh['evaluations'].items():
        backbone,encoded=key.rsplit('_alpha',1)
        alpha=int(encoded)/100
        if backbone not in ('additive','scale_shift') or alpha not in (.25,.5,.75):
            raise ValueError('Unexpected prospective amplitude or backbone')
        home=project_path(relative)
        ep=json.loads((home/'protocol.json').read_text())
        if ep['guide_strength']!=alpha:
            raise ValueError('Saved strength identity mismatch')
        after=read_grading_csv(home/'per_image.csv')
        for axis in ('manufacturer','field_strength'):
            eligibility.append(dict(backbone=backbone,guide_strength=alpha,axis=axis,
                original_selection_seed=config['evaluation']['selection_generation_seed'],
                check=select_setting(after,axis,config)))
        references={'parent':primary_rows(read_grading_csv(project_path(parents[backbone])/'per_image.csv'),config),
            'full_guide':read_grading_csv(project_path(original['evaluations'][backbone+'__soft_source_guide__correct'])/'per_image.csv'),
            'learned_zero_guide':read_grading_csv(project_path(original['evaluations'][backbone+'__learned_zero_guide__correct'])/'per_image.csv')}
        for reference,before in references.items():
            part=compare_rows(before,after,config)
            part['backbone']=backbone
            part['guide_strength']=alpha
            part['contrast']='strength_minus_'+reference
            rows.append(part)
        for name in ('protocol.json','requests.csv','per_image.csv','complete.json'):
            path=home/name
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    csv_write(output/'paired_strength_differences.csv',pd.concat(rows,ignore_index=True))
    json_write(output/'provisional_eligibility.json',dict(checks=eligibility,
        method='Original selection seed, support/direction, SSIM, other-axis stability and positive target-minus-null gates; evaluated separately per strength',
        alpha_selected=False,clinical_anatomy_validated=False,criteria_changed=False))
    json_write(output/'selection.json',dict(selected=None,
        reason='Prospective development sensitivity profile only; no automatic clinical/default promotion or relaxation of existing support gates',
        unchanged_support_eligibility=True,reused_validation_patients=True,fresh_test=False))
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,new_strength_evaluations=6,
        new_cases=540,reused_reference_cases=720,additional_training_ran=False,
        fixed_weights=True,anatomy_validated=False,default_promoted=False))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.analyse_strength. paired_strength_differences.csv '
        'contains every prospective amplitude/backbone/axis/named-judge/control/support/direction '
        'contrast to parent/full-guide/learned-zero references, with patient CIs. The companion '
        'verification report retains strict regrading and unchanged preservation screening. '
        'provisional_eligibility.json applies the original provisional T/w criteria separately '
        'at each alpha using only the original selection seed; it never selects an alpha. '
        'selection.json keeps no automatic winner and unchanged support limits. inputs.json '
        'binds evidence; complete.json records inference-only development scope.\n',encoding='utf-8')
    print('STRENGTH ANALYSIS COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--pilot-fetch',required=True)
    p.add_argument('--parent-fetch',required=True)
    p.add_argument('--prefix',required=True)
    a=p.parse_args()
    analyse(a.fetched,a.pilot_fetch,a.parent_fetch,a.prefix)
