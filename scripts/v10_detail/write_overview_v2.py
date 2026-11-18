"""Assemble the verified study entrypoint without choosing favourable judges or controls."""
import argparse
import json

import pandas as pd

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def interval(row, scale=1, suffix=''):
    return f'{scale*row.estimate:.4f} [{scale*row.ci_low:.4f}, {scale*row.ci_high:.4f}]{suffix}'


def warning_ids(frame):
    flags = frame.qc_warning
    if not flags.isin([0, 1, False, True]).all():
        raise ValueError('Missing or invalid source-warning flags')
    return sorted(frame.loc[flags.astype(bool), 'patient_id'].unique())


def write(results, prefix, metadata, budget, training, visuals, run_id):
    results, metadata, budget, training, visuals = map(project_path, (results, metadata, budget, training, visuals))
    base = ROOT / 'reports/v10_detail'
    verified, analysis, report = [base / (prefix + suffix) for suffix in ('_verified', '_analysis', '_report')]
    checks = json.loads((verified / 'complete.json').read_text())
    extra = json.loads((metadata / 'complete.json').read_text())
    trained = json.loads((training / 'complete.json').read_text())
    pictured = json.loads((visuals / 'complete.json').read_text())
    if trained['actual_logged_updates_checked'] != 25020 or not trained['finite_applied_updates']:
        raise ValueError('Every actual applied training update must be checked')
    if not pictured['all_named_judges'] or not pictured['all_validation_patients']:
        raise ValueError('All-patient/all-judge figures required')
    if not checks['saved_outputs_verified'] or checks['cases'] != 2295 or checks['grades'] != 20655:
        raise ValueError('All saved cases and named grades must pass before interpretation')
    if extra['cases'] != 2295 or extra['grades'] != 20655 or not extra['request_grade_metadata_equal']:
        raise ValueError('Full supplementary metadata audit required')
    gate = json.loads((analysis / 'advancement.json').read_text())
    cost = json.loads((budget / 'budget.json').read_text())
    if not cost['app_stopped'] or not cost['worker_time_within_cap'] or not cost['conservative_estimate_within_cap']:
        raise ValueError('Closed allocation within both caps required')
    rates = pd.read_csv(analysis / 'crossed_intervals.csv', keep_default_na=False, na_values=[''])
    context = pd.read_csv(report / 'contextual_reference.csv')
    receipt = json.loads((results / 'complete.json').read_text())
    protocol_path = project_path(receipt['prepared']) / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    fs_path = project_path(next(path for key, path in receipt['evaluations'].items()
                                if '__field_strength__' in key)) / 'requests.csv'
    fs = pd.read_csv(fs_path)
    eligible = fs[fs.condition_support_status.isin(protocol['evaluation']['selection_support_statuses'])]
    support = {direction: int(eligible.loc[eligible.direction == direction, 'patient_id'].nunique())
               for direction in ('0_to_1', '1_to_0')}
    warning_patients = warning_ids(fs)
    outcome = ('met' if gate['research_advancement'] else 'did not meet')
    lines = ['# Completed replicated source-guide study', '',
        f'The finer guide **{outcome} the frozen development advancement rule** against its matched coarse control. '
        'This is a reused-development comparison, not a clinical preservation or independent-test result. '
        'No editor default is promoted.', '',
        'All nine adapters completed the same 20-epoch exposure: 2,780 applied updates each (25,020 total). '
        'The full 51-profile matrix contains 2,295 source/generation cases and 20,655 named-judge/control grades '
        'from 15 fixed validation source slices/patients, three adapter-training seeds and three generation seeds. '
        'These repeated cases are not 2,295 independent patients. EchoTime stays at each actual source value.', '',
        'Independent local regrading checked every saved source/array, image diagnostic, named-judge probability/decision '
        f'and absent-guide equality; maximum probability error was {checks["max_probability_error"]:.9g}. '
        'The supplementary audit checks all request/grade metadata, guide identities, array target labels and actual-source EchoTime. '
        'Full sampling trajectories and raw training noise arrays were not independently regenerated.', '',
        '## Frozen primary gates', '',
        'Each interval below uses the predeclared simultaneous family correction. Advancement also requires '
        'the predeclared per-training-seed directions and original support/correspondence safeguards. '
        'The tolerances are research choices, not clinical safety/equivalence margins.', '',
        '| Control / metric / named judge | Fine minus coarse [simultaneous interval] | Interval and replica-direction pass |',
        '|---|---:|---|']
    for row in gate['gates']:
        lines.append(f'| {row["control"]} / {row["metric"]} / {row["judge"]} | '
            f'{row["estimate"]:.6f} [{row["ci_low"]:.6f}, {row["ci_high"]:.6f}] | '
            f'{row["interval_passed"] and row["replica_direction_passed"]} |')
    lines += ['', 'Original safeguards by adapter-training seed:', '']
    for row in gate['original_safeguards']:
        lines.append(f'- {row["training_seed"]}: {row["check"]["reason"]}; selected operating point: {row["check"]["selected"]}.')
    for axis in ('manufacturer', 'field_strength'):
        lines += ['', '## ' + axis.replace('_', ' ').title() + (' (exploratory)' if axis == 'field_strength' else ' (primary)'), '',
            'Raw target-hit and target-minus-null editing are different quantities. These descriptive 95% '
            'intervals resample patients and adapter-training seeds independently, averaging the generation seeds within each cell.', '',
            '| Mode / named judge | Target-hit % [95% CI] | Target-minus-null, percentage points [95% CI] |', '|---|---:|---:|']
        for mode in ('coarse_control', 'coarse_fine', 'learned_zero'):
            for j in range(3):
                judge = f'heldout_orientation_robust_seed{j}'
                part = rates[(rates.axis == axis) & (rates['mode'] == mode) & (rates.judge == judge)
                    & (rates.guide_control == 'correct') & (rates.variant == 'target') & (rates.stratum == 'all')]
                hit, delta = [part[part.metric == metric].iloc[0] for metric in ('target_hit', 'paired_hit_delta')]
                lines.append(f'| {mode} / {judge} | {interval(hit, 100)} | {interval(delta, 100)} |')
    lines += ['', f'Field-strength support is {support["0_to_1"]} forward / {support["1_to_0"]} reverse eligible patients, '
        f'versus the unchanged minimum of {protocol["evaluation"]["minimum_eligible_patients_per_direction"]} per direction. '
        'Its results cannot pass that support requirement. Source-mask warning patients '
        f'({", ".join(warning_patients)}) and all unsupported/out-of-range cases remain in the outputs and analysis strata.', '',
        '## Historical reference (context only)', '',
        'The historical alpha-.50 editor has one trained adapter held fixed. These matched source/generation comparisons '
        'do not constitute matched training-seed replication for that old adapter.', '',
        '| Named judge | New fine minus historical alpha-.50 target-minus-null, percentage points [95% crossed CI] |', '|---|---:|']
    for row in context[(context.control == 'target') & (context.metric == 'paired_hit_delta')].itertuples():
        lines.append(f'| {row.judge} | {interval(row, 100)} |')
    lines += ['', '## Limits and retained controls', '',
        'The source-mask interior, high-frequency residual and SSIM diagnostics measure correspondence to the source. '
        'They do not establish lesion/anatomy preservation or physical scanner realism. The same validation cohort has '
        'been used in earlier development; the three training replicas give limited replication, not independent confirmation.', '',
        'Every correct/zero/absent/wrong/split-guide condition, original-label null, identity, direction, support and '
        'mask-warning stratum remains in the analysis tables. For coarse_control, both input channel blocks contain coarse '
        'guides: its fine_only condition retains the second coarse block, not fine information. For coarse_fine, fine_only '
        'retains the actual finer block. Learned_zero supplies no source guide. No adverse control is discarded.', '',
        '## Allocation and evidence', '',
        f'The provider reports the app stopped. Worker duration: {cost["worker_seconds"]:.2f} seconds; '
        f'conservative app-wall estimate including the frozen rate margin and reserve: US${cost["conservative_estimate_usd"]:.4f} '
        'against the separate US$12 allowance. The provider invoice is unknown. Previous allocations stay closed; '
        'unused balances do not authorise another paid run.', '']
    evidence_homes = [('Raw retrieval', results), ('Saved-output verification', verified), ('Metadata verification', metadata),
        ('Actual training-journal verification', training), ('All crossed/per-replica/control tables', analysis),
        ('Frozen postprocessing/contextual reference', report), ('All-patient/all-judge figures', visuals), ('Budget/app-stop evidence', budget)]
    lines.extend(f'- [{label}]({(path / "README.md").as_posix()})' for label, path in evidence_homes)
    output = base / run_id; output.mkdir(parents=True, exist_ok=False)
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    paths = [project_path(__file__), verified / 'complete.json', metadata / 'complete.json', analysis / 'advancement.json',
        analysis / 'crossed_intervals.csv', report / 'contextual_reference.csv', budget / 'budget.json', results / 'complete.json',
        protocol_path, fs_path, training / 'complete.json', visuals / 'complete.json', visuals / 'inputs.json']
    json_write(output / 'inputs.json', {path.relative_to(ROOT).as_posix(): digest(path) for path in paths})
    json_write(output / 'complete.json', dict(complete=True, research_advancement=gate['research_advancement'],
        default_promoted=False, clinical_preservation='unknown', fresh_test=False, app_stopped=True))
    (output / 'README.md').write_text('Producer: scripts.v10_detail.write_overview_v2. RESULTS.md: verified study outcome, '
        'every correct-guide named-judge rate/editing interval, simultaneous gates, support/clinical limits, contextual '
        'historical comparison and budget/evidence entrypoint; inputs.json: exact consumed outputs and producer; '
        'complete.json: scope. No sampling, selection, gate change, publication edit or default promotion.\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ('results', 'prefix', 'metadata', 'budget', 'training', 'visuals', 'run-id'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); write(a.results, a.prefix, a.metadata, a.budget, a.training, a.visuals, a.run_id)
