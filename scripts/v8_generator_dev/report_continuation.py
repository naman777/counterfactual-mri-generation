"""Write a traceable completed-continuation development report, preserving negative results."""
import argparse
import json

import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def interval(row,rate=True):
    multiplier,places=(100,1) if rate else (1,3)
    return ('{0:.'+str(places)+'f} [{1:.'+str(places)+'f}, {2:.'+str(places)+'f}]').format(
        row['mean']*multiplier,row['ci95_lo']*multiplier,row['ci95_hi']*multiplier)


def report(fetched,prefix,run_id):
    fetched=project_path(fetched)
    base=ROOT/'reports/v8_generator_dev'
    comparison=base/(prefix+'_comparison')
    differences=base/(prefix+'_epoch70_comparison')
    preservation=base/(prefix+'_preservation')
    verification=base/(prefix+'_verification')
    paths=[fetched/'complete.json',fetched/'training/complete.json',comparison/'primary_estimates.csv',
        comparison/'paired_primary_differences.csv',differences/'paired_epoch70_differences.csv',
        preservation/'edit_intervals.csv',preservation/'complete.json',verification/'complete.json',
        verification/'grading_checks.csv',base/'continuation_launch_20261117_01/process_result.json']
    for path in paths:
        if not path.is_file():
            raise ValueError('Completed continuation evidence missing: '+str(path))
    trained=json.loads(paths[1].read_text())
    verified=json.loads((verification/'complete.json').read_text())
    estimates=pd.read_csv(comparison/'primary_estimates.csv',keep_default_na=False)
    delta=pd.read_csv(differences/'paired_epoch70_differences.csv',keep_default_na=False)
    pair=pd.read_csv(comparison/'paired_primary_differences.csv',keep_default_na=False)
    output=base/run_id
    output.mkdir(exist_ok=False)
    primary=estimates[estimates.stratum=='all_flagged_and_in_range']
    csv_write(output/'headline_estimates.csv',primary)
    launch=json.loads((base/'continuation_launch_20261117_01/process_result.json').read_text())
    ledger=dict(approved_additional_usd=6,allowance_reset=False,cli_wall_seconds=launch['seconds'],
        rate_with_margin=.000441264,reserve_usd=.5,
        conservative_estimated_usd=launch['seconds']*.000441264+.5,exact_invoice_known=False,
        method='Whole CLI wall time, including build/queue, at 20%-margin worker rate plus reserve; original US$6 separate')
    json_write(output/'budget.json',ledger)
    lines=['# Completed matched generator continuation','',
        f'Both models completed at common epoch {trained["progress"]["additive"]["epoch"]}. '
        f'Stopping reason: `{trained["stop_reason"]}`. The bounded precision repair retained matched '
        'training exposure and applied the recovered formerly failing draw. Completion of training '
        'is not proof of strong editing, physical scanner simulation or clinical anatomy preservation.','',
        'The cloud controller exited with a retained packaging failure in its final analysis step '
        '(missing `scripts.v8_generator_dev.analyse_full`); training and both validation phases '
        'had already completed successfully. The analysis below was recomputed locally from '
        'those immutable outputs. The cloud failure is not relabelled as a successful workflow, '
        'and no GPU restart was used for recovery.','',
        f'Independent local saved-array verification checked {verified["cases"]} cases and '
        f'{verified["grading_rows"]} named-judge/control grades. All discrete decisions and paired '
        'deltas must agree; probabilities use the declared cross-runtime tolerance.','',
        'All tables below use the original fifteen development-validation patients, fixed operating '
        'points and all three generation seeds. Values are equal-patient estimates with 95% '
        'patient-bootstrap intervals; direction and condition-support strata remain in the source CSVs. '
        'Confirmation seeds are repeated generation on the same patients. There is no fresh-test claim.','',
        '## Fixed-point classifier results','',
        'Rates and differences are percentages / percentage points. Each judge is reported separately.','',
        '| Backbone | Edited axis | Frozen judge | Target hit | Null hit | Target minus null |','|---|---|---|---|---|---|']
    for (model,axis,judge),group in primary.groupby(['model','axis','judge']):
        def row(control,metric):
            found=group[(group.control==control)&(group.metric==metric)]
            if len(found)!=1:
                raise ValueError('Missing unique headline metric')
            return found.iloc[0]
        lines.append(f'| {model} | {axis} | {judge} | {interval(row("target","target_hit"))} | '
            f'{interval(row("null","target_hit"))} | {interval(row("target","paired_hit_delta"))} |')
    lines += ['','## Matched change from epoch 70','',
        'The sign is continued minus epoch 70. A target-hit increase accompanied by a similar null-hit '
        'increase is not evidence that target conditioning became stronger.','',
        '| Backbone | Axis | Judge | Change in target hit (pp) | Change in target-minus-null (pp) |',
        '|---|---|---|---|---|']
    for (model,axis,judge),group in delta[(delta.stratum=='all_flagged_and_in_range')&(delta.control=='target')].groupby(['model','axis','judge']):
        values={name:part.iloc[0] for name,part in group.groupby('metric')}
        lines.append(f'| {model} | {axis} | {judge} | {interval(values["target_hit"])} | {interval(values["paired_hit_delta"])} |')
    lines += ['','## Correspondence and preservation limits','',
        '| Backbone | Axis | Target SSIM | Target MAD |','|---|---|---|---|']
    for (model,axis),group in primary[(primary.control=='target')&(primary.judge=='heldout_orientation_robust_seed0')].groupby(['model','axis']):
        values={name:part.iloc[0] for name,part in group.groupby('metric')}
        lines.append(f'| {model} | {axis} | {interval(values["ssim"],False)} | {interval(values["mad"],False)} |')
    lines += ['','The separately versioned automated screen uses exactly the earlier training-calibrated '
        'cutoffs. Its complete rates and synthetic-control false flags are retained in '
        f'`{prefix}_preservation/`. Rank, edge and local-texture disagreement do not label clinical '
        'lesion damage; good SSIM and a passed proxy screen would not establish clinical preservation '
        'either. The screen has limited sensitivity to tiny erased patches, and its metric development '
        'used validation diagnostics. Clinical anatomy and lesion preservation remain unknown.','',
        '## Support, architecture and next experiment','',
        'Retain the saved infeasible field-strength selection and all outside-range requests. The '
        'current support limitation is not repaired by a better classifier score or by the two '
        'unassigned prospective candidates. No architecture is automatically selected as a winner. '
        'All matched scale-shift-minus-additive intervals remain in `paired_primary_differences.csv`.','',
        'The separately approved next pilot freezes both verified parents, trains source-guide and '
        'learned zero-guide adapter copies with matched exposure, and tests wrong/absent guides. '
        'It directly investigates source-detail and copy-shortcut trade-offs. The guide still leaks '
        'acquisition information. No judge optimisation or current test-cohort tuning is authorised.','',
        '## Budget and provenance','',
        f'This allowance: US$6. Conservative whole-CLI estimate: US${ledger["conservative_estimated_usd"]:.2f}, '
        'including the 20% rate margin and US$0.50 reserve. This is not an invoice. The original '
        'US$6 ledger and separately approved US$8 pilot allowance remain separate.','',
        'Producer: `python -m scripts.v8_generator_dev.report_continuation`. `inputs.json` hashes '
        'the training, retrieval, verification and analysis sources; `headline_estimates.csv` '
        'contains the complete overall source rows. The full analysis CSVs retain support/direction '
        'strata, all controls and uncertainty. No historical report or paper is edited.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    json_write(output/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in
        (*paths,ROOT/'scripts/v8_generator_dev/report_continuation.py')})
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.report_continuation. RESULTS.md is the completed '
        'development report; headline_estimates.csv retains overall source estimates; budget.json '
        'separates the conservative new-allowance estimate from the unknown invoice. inputs.json '
        'hashes all raw analysis/training evidence. No clinical preservation or winner claim.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,training_completed=True,saved_validation_verified=True,
        clinical_preservation='unknown',fresh_test=False,winner_selected=False))
    print('CONTINUATION REPORT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fetched',required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    report(a.fetched,a.prefix,a.run_id)
