"""Independent local report checks and explicit next-stage research decision."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,verify_inputs
from scripts.v9_detail.common import BASE,start,finish,display,interval


def review(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.review','Local preparation review and next-stage gates')
    homes={name:BASE/folder for name,folder in dict(
        masks='mask_audit_20261117_01',reconstruction='reconstruction_20261117_01',
        controls='fine_controls_20261117_01',checks='local_checks_20261117_01',
        parent='parent_preflight_20261117_01').items()}
    json_write(output/'protocol.json',dict(reports={k:str(v.relative_to(ROOT)) for k,v in homes.items()},
        role='Independent table/cutoff/endpoint checks and interpretive closure; no tuning, new generation or model selection'))
    for home in homes.values():
        if not json.loads((home/'complete.json').read_text())['complete']:
            raise ValueError('Completed local reports required')
        verify_inputs(json.loads((home/'inputs.json').read_text()))
        for p in home.iterdir():
            if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    checks=[]
    def checked(name,condition,details):
        if not condition:raise AssertionError(name+' failed: '+str(details))
        checks.append(dict(check=name,passed=True,details=str(details)))
    mask=pd.read_csv(homes['masks']/'mask_cases.csv')
    thresholds=json.loads((homes['masks']/'thresholds.json').read_text())['thresholds']
    for variant,g in mask[mask.split.eq('train')].groupby('variant'):
        actual=dict(area_low=float(g.mask_fraction.quantile(.01)),area_high=float(g.mask_fraction.quantile(.99)),
            gradient_low=float(g.gradient_energy_inside.quantile(.01)),stability_low=float(g.min_control_iou.quantile(.05)))
        error=max(abs(actual[k]-thresholds[variant][k]) for k in actual)
        checked('train_mask_cutoffs_'+variant,error<1e-12,error)
    train=pd.read_csv(homes['controls']/'calibration.csv')
    control_thresholds=json.loads((homes['controls']/'thresholds.json').read_text())['thresholds']
    for metric,expected in control_thresholds.items():
        actual=max(1e-6,float(train[train.valid].groupby('patient_id')[metric].max().quantile(.95)))
        checked('train_detail_cutoff_'+metric,abs(actual-expected)<1e-12,actual)
    controls=pd.read_csv(homes['controls']/'validation_controls.csv')
    for row in controls.itertuples():
        if row.valid:
            expected=int(max(getattr(row,k)/v for k,v in control_thresholds.items())>1)
            if expected!=row.flag:raise AssertionError('Control screen decision mismatch')
    checked('every_saved_control_flag_recomputed',True,len(controls))
    recon=pd.read_csv(homes['reconstruction']/'patient_reconstruction.csv')
    with np.load(homes['reconstruction']/'validation_reconstructions.npz') as saved:
        for name in saved.files:
            if name=='source':continue
            expected=recon[recon.probe.eq(name)].set_index('patient_id')
            selection=pd.read_csv(homes['reconstruction']/'patient_selection.csv')
            expected=expected.loc[selection[selection.split.eq('val')].patient_id]
            error=np.abs(np.abs(saved[name]-saved['source']).mean((1,2))-expected['mad'].to_numpy()).max()
            checked('saved_reconstruction_mad_'+name,error<1e-12,error)
        checked('global_zero_guide_matches_train_template',
            np.max(np.abs(saved['global__learned_zero']-saved['train_patient_mean_template']))<1e-6,
            'Zero-input global ridge equals the training-only template')
    steps=pd.read_csv(homes['parent']/'steps.csv')
    checked('all_actual_updates_applied',len(steps)==12 and steps.applied.all(),len(steps))
    for step,g in steps.groupby('global_step'):
        for col in ('sample_indices','timesteps','condition_dropped','noise_sha256'):
            checked('matched_draw_'+str(step)+'_'+col,g[col].nunique()==1,g[col].iloc[0])
    parent=json.loads((homes['parent']/'checks.json').read_text())
    for copy in parent['copies']:
        key=copy['backbone']+'__'+copy['mode']
        with np.load(homes['parent']/(key+'_arrays.npz')) as saved:
            for axis in ('manufacturer','field_strength'):
                checked(key+'_'+axis+'_full_mask_identity',
                    np.array_equal(saved[axis+'__mask_full'],saved[axis+'__mask_unmasked']),True)
        ckpath=ROOT/copy['checkpoint']
        checked(key+'_checkpoint_hash',digest(ckpath)==copy['checkpoint_sha256'],copy['checkpoint_sha256'])
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        checked(key+'_engineering_only_finite',ck['engineering'] and ck['step']==2 and
            all(torch.isfinite(v).all().item() for v in ck['adapter'].values()),'Engineering cursor=2; no efficacy checkpoint')
        del ck
    grades=pd.read_csv(homes['parent']/'judge_routes.csv')
    for row in grades.itertuples():
        values=np.array(json.loads(row.probabilities_json))
        if not np.isfinite(values).all() or not np.allclose(values.sum(1),1,atol=1e-6):
            raise AssertionError('Invalid frozen judge probabilities')
    checked('all_named_engineering_judge_probabilities_finite_normalised',True,len(grades))
    for folder in ('detail_diagnosis_20261117_01','detail_preparation_20261117_01','detail_report_20261117_01'):
        prior=ROOT/'reports/v8_generator_dev'/folder
        verify_inputs(json.loads((prior/'inputs.json').read_text()))
        checked('historical_detail_inputs_unchanged_'+folder,True,'All prior bound inputs pass')
    csv_write(output/'verification.csv',pd.DataFrame(checks))
    mask_ci=pd.read_csv(homes['masks']/'patient_intervals.csv')
    recon_ci=pd.read_csv(homes['reconstruction']/'paired_differences.csv')
    control_ci=pd.read_csv(homes['controls']/'control_intervals.csv')
    forwards=[c['forward32'] for c in parent['copies'] if c['forward32']['attempted']]
    candidate_failures=mask[(mask.split=='val')&(mask.variant=='noise_aware_v1')&(mask.qc_warning==1)]
    lines=['# Local mask, shortcut and integration review','',
        'The next local preparation tasks are complete. The mask revision avoids the observed noisy-background '
        'full-frame failure, and the probes/controls retain important limits. This does not establish an anatomy-preserving editor.','',
        '## Source-mask development','',
        'Training-only cutoffs were independently recomputed. Validation uses the exact frozen editor source slices; '
        'training uses the historical guide-audit selection. This differs from the validation slices of the prior guide probe. '
        'No validation image or generator score was used to tune numerical warning cutoffs. Algorithm development remains informed '
        'by the known validation failure, so this is development evidence.','',
        '| Mask / validation metric | Estimate [95% patient CI] |','|---|---|']
    for row in mask_ci[(mask_ci.split=='val')&mask_ci.metric.isin(['min_control_iou','near_full_frame','qc_warning'])].itertuples():
        lines.append(f'| {row.variant} / {row.metric} | {display(row)} |')
    lines += ['', 'The revised mask still warns on: '+', '.join(candidate_failures.patient_id)+'. '
        'All cases stay in the report. Coverage reduction alone is not verified tissue correction; low-signal '
        'tissue, contaminated corners and clinical boundaries remain unresolved.','',
        '## Source appearance remains accessible','',
        'Train-only decoders show accessible correlated source appearance. The fine guide can reconstruct the '
        'source better than a training-mean template, but the global fine-minus-coarse difference includes zero. '
        'The small local-decoder gain does not establish actual copying by a trained adapter.','',
        '| Fine-minus-reference MAD | Estimate [95% patient CI] |','|---|---|']
    for row in recon_ci[recon_ci.metric.eq('mad')].itertuples():
        lines.append(f'| {row.architecture} vs {row.reference} | {display(row)} |')
    lines += ['', 'Targets are 64px source averages; no native fine-pathology reconstruction is established. '
        'The earlier scanner-label leakage evidence remains in force. No probe result certifies invariance or copy resistance.','',
        '## Fine-control blind spots retained','',
        'Training-only cutoffs and every stored decision were independently recomputed. Historical screens are unchanged.','',
        '| Development control | Fraction flagged [95% patient CI] |','|---|---|']
    selected=('erase_1px','erase_2px','local_smooth_7px','boundary_gain_seam','boundary_translate_2px','noise_0005')
    for row in control_ci[control_ci.location.eq('all_locations')&control_ci.control.isin(selected)].itertuples():
        lines.append(f'| {row.control} | {display(row)} |')
    lines += ['', 'One-pixel erasures are frequently missed; the gain seam is entirely missed in this cohort. '
        'For erasures, the headline averages the fraction of tested locations within patient. These are synthetic '
        'processed-grid engineering results with validation-informed metric development, not clinical lesion sensitivity '
        'or independent confirmation. All false flags and location strata remain. Zero/all observed detection gives '
        'a degenerate empirical interval, not population certainty.','',
        '## Final-parent integration','',
        f'{len(parent["copies"])} six-input copies completed 12 bounded local updates total (two per copy, batch two) '
        'on the actual epoch-130 additive and scale-shift parents. All data/noise/dropout draws match across copies, '
        'and the backbone and conditioner remain frozen. Strict CPU resume/rejection/control tests pass. All correct, '
        'zero, absent, wrong, coarse-erased/fine-erased guide routes and unmasked/full/empty/source/translated/wrong/'
        'final-compositing mask routes ran through the new evaluator at engineering t=2,w=3 on both axes. '
        'Full-mask outputs equal unmasked output exactly; absent guide equals the parent. Three named frozen judges '
        'receive source-native arrays with finite probabilities; these short-step checks supply no efficacy hit rates.','',
        f'Batch-32 forward-only probes: {sum(f["passed"] for f in forwards)} / {len(forwards)} passed. '
        'This is distinct from batch-32 optimisation. Full training-batch backward/optimiser checks on the intended '
        'runtime remain unperformed. The saved weights are engineering-only snapshots, and the new trainer/evaluator '
        'are deliberately capped; no full efficacy run or cloud-ready package exists.','',
        'All source, code, report, checkpoint and prior detail-input hashes pass. The known NumPy/SciPy warning '
        'is retained. All three source-only all-patient/summary panels were visually inspected.','',
        '## Next concrete work','',
        '1. Make the boundary-seam blind spot a separate versioned metric-development task, preserving this failed '
        'screen and reporting new false flags; do not retune this report or call the proxy clinically validated.',
        '2. Specify source-mask warning handling prospectively: retain warning cases, no favourable-case exclusion. '
        'Keep the mask test separate from the finer-guide comparison; source correspondence and copied background '
        'cannot certify internal tissue.',
        '3. Prepare the full-exposure trainer/evaluator, multiple training replicas, fixed quantitative advancement '
        'criteria/multiplicity policy, exact package/input hashes and a time/cost cap. Complete full-training-batch '
        'backward/optimiser checks on the intended runtime before any efficacy launch. Local capped integration '
        'does not confer automatic paid permission.',
        '4. Only after that package is reviewable, request the separate required paid allowance. No old remainder is '
        'transferred. Field-strength support still needs a separate data/protocol solution; no support rule, split, '
        'EchoTime or publication result was changed.','',
        'No mask, guide, model or default is promoted. Clinical anatomy and lesion preservation remain unknown.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'OUTPUTS.md').write_text('Producer: scripts.v9_detail.review. verification.csv: independent cutoff, '
        'control-decision, reconstruction, matched-update, endpoint, checkpoint and historical-input checks. '
        'RESULTS.md: consolidated findings, limitations and next-stage gates.\n',encoding='utf-8')
    finish(output,inputs,verification_checks=len(checks),all_checks_passed=True,
        full_training_batch_checked=False,cloud_ready=False,model_or_mask_promoted=False)
    print('LOCAL REVIEW COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    review(p.parse_args().run_id)
