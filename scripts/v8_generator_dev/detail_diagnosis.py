"""Read-only matched source/null/target diagnostics on saved development arrays."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from scripts.v5_anatomy.metrics import foreground_proxy, image_array
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.preservation_screen import local_rank

BASE = ROOT / 'reports/v8_generator_dev'
METRICS = ('mad', 'ssim', 'rank_mad', 'hf_residual_mad', 'hf_energy_ratio')
KEYS = ['patient_id', 'dataset_index', 'axis', 'generation_seed']


def regions(source, threshold=.05):
    """Fixed source-only partition; boundary spans both sides of proxy outline."""
    fg = foreground_proxy(source, threshold)
    interior = ndi.binary_erosion(fg, iterations=3, border_value=0)
    expanded = ndi.binary_dilation(fg, iterations=3)
    return dict(whole=np.ones_like(fg), interior=interior,
                boundary=expanded & ~interior, background=~expanded)


def maps(a, b):
    a, b = image_array(a), image_array(b)
    if a.shape != b.shape:
        raise ValueError('No implicit registration or resizing')
    _, ssim = structural_similarity(a, b, data_range=1., win_size=7, full=True)
    ha = a - ndi.gaussian_filter(a, 1, mode='reflect')
    hb = b - ndi.gaussian_filter(b, 1, mode='reflect')
    return dict(mad=np.abs(b-a), ssim=ssim,
                rank_mad=np.abs(local_rank(b)-local_rank(a)),
                hf_residual_mad=np.abs(hb-ha), hf_a=ha**2, hf_b=hb**2)


def measure(values, mask):
    n = int(mask.sum())
    result = dict(pixels=n, valid=n >= 64)
    for metric in METRICS:
        result[metric] = np.nan
    if result['valid']:
        for metric in METRICS[:-1]:
            result[metric] = float(values[metric][mask].mean())
        energy = float(values['hf_a'][mask].mean())
        if energy > 1e-10:
            result['hf_energy_ratio'] = float(values['hf_b'][mask].mean()/energy)
    return result


def summarise(frame, seed, resamples):
    """Equal patient weight, seeds averaged first; missing metrics explicit."""
    rows = []
    grouping = ['profile', 'axis', 'threshold', 'region', 'comparison']
    for key, group in frame.groupby(grouping, sort=True):
        strata = [('all', group)]
        if key[2] == .05:
            for col in ('direction', 'condition_support_status'):
                strata.extend((col+':'+str(k), v) for k, v in group.groupby(col))
        for stratum, part in strata:
            for metric in METRICS:
                means = part.groupby('patient_id')[metric].mean().dropna().to_numpy()
                estimate, lo, hi = np.nan, np.nan, np.nan
                if len(means):
                    rng = np.random.default_rng(seed)
                    draws = rng.integers(len(means), size=(resamples, len(means)))
                    estimates = means[draws].mean(axis=1)
                    estimate = float(means.mean())
                    lo, hi = np.quantile(estimates, [.025, .975])
                rows.append(dict(zip(grouping, key), stratum=stratum, metric=metric,
                    estimate=estimate, ci_low=lo, ci_high=hi,
                    patients_valid=len(means), patients_total=part.patient_id.nunique(),
                    cases_valid=int(part[metric].notna().sum()), cases_total=len(part)))
    return pd.DataFrame(rows)


def inventory():
    receipts = {name: BASE / folder / 'complete.json' for name, folder in dict(
        parent='continuation_results_20261117_01', pilot='adapter_results_20261117_01',
        strength='strength_results_20261117_01').items()}
    saved = {k: json.loads(p.read_text()) for k, p in receipts.items()}
    if not saved['parent']['training_and_validation_success'] or not all(
        saved[k]['workflow_success'] for k in ('pilot', 'strength')):
        raise ValueError('Require completed component evidence; preserve parent packaging failure')
    homes = {k+'_parent': v for k, v in saved['parent']['evaluation_folders'].items()}
    homes.update(saved['pilot']['evaluations'])
    homes.update(saved['strength']['evaluations'])
    evidence = {p.relative_to(ROOT).as_posix(): digest(p) for p in receipts.values()}
    # Bind the existing strict regrading receipt without claiming new regrading.
    prior = BASE / 'strength_analysis_20261117_01_verification/complete.json'
    if not json.loads(prior.read_text())['saved_validation_verified']:
        raise ValueError('Prior strict verification is required')
    evidence[prior.relative_to(ROOT).as_posix()] = digest(prior)
    expected = {}
    for folder in ('strength_results_20261117_01', 'adapter_results_20261117_01',
                   'continuation_results_20261117_01'):
        expected.update(json.loads((BASE/folder/'inputs.json').read_text()))
        evidence[(BASE/folder/'inputs.json').relative_to(ROOT).as_posix()] = digest(BASE/folder/'inputs.json')
    return homes, evidence, expected


def run(run_id):
    if not run_id.replace('_', '').isalnum():
        raise ValueError('Use a safe fresh ID')
    output = BASE / run_id
    output.mkdir(exist_ok=False)
    homes, evidence, expected = inventory()
    first = project_path(next(iter(homes.values())))
    config = json.loads((first/'protocol.json').read_text())['config']
    evaluation = config['evaluation']
    protocol = dict(profiles=homes, reference='additive_alpha050', axes=['manufacturer', 'field_strength'],
        fixed_settings=evaluation['fixed_settings'], thresholds=[.03, .05, .1],
        partition='source foreground proxy; erode/dilate 3 pixels; whole/interior/boundary/background',
        minimum_region_pixels=64, primary_threshold=.05, metrics=list(METRICS),
        hf_definition='image minus Gaussian sigma=1 pixel; squared-energy ratio b/a; not anatomy',
        ssim_definition='Mean 7x7 SSIM map on selected centres; windows can cross ROI boundary',
        comparisons=['source_null', 'source_target', 'null_target', 'target_minus_null_source_error'],
        attribution='Paired diagnostic; nonlinear changes are not additive causal decomposition',
        inference='Equal-patient percentile bootstrap; generation seeds averaged within patient',
        bootstrap_seed=evaluation['bootstrap_seed'], bootstrap_resamples=evaluation['bootstrap_resamples'],
        stage='Retrospective exploratory development, all cases retained; no fresh test',
        selection='No guide, mask, model, threshold or operating-point selection',
        clinical_preservation='unknown', paid_compute=False)
    json_write(output/'protocol.json', protocol)
    rows, index_rows, panels, reference = [], [], [], {}
    matched = None
    for profile, relative in homes.items():
        home = project_path(relative)
        for filename in ('requests.csv', 'protocol.json', 'complete.json'):
            p = home/filename
            name = p.relative_to(ROOT).as_posix()
            actual = digest(p)
            if expected.get(name) != actual:
                raise ValueError('Missing or changed retrieved input: '+name)
            evidence[name] = actual
        requests = pd.read_csv(home/'requests.csv')
        keep = np.zeros(len(requests), bool)
        for axis, point in evaluation['fixed_settings'].items():
            keep |= ((requests.axis == axis) & (requests.timestep == point['timestep']) &
                     np.isclose(requests.guidance, point['guidance']))
        requests = requests[keep].copy()
        keys = set(map(tuple, requests[KEYS].to_numpy()))
        if requests.duplicated(KEYS).any() or (matched is not None and keys != matched):
            raise ValueError('Unmatched/duplicate patient-axis-seed cases')
        matched = keys
        if set(requests.generation_seed) != {evaluation['selection_generation_seed'],
                *evaluation['confirmation_generation_seeds']} or len(keys) != 90:
            raise ValueError('Missing fixed-point cases/seeds')
        for request in requests.to_dict('records'):
            key = tuple(request[k] for k in KEYS)
            p = project_path(home/request['array_file'])
            if not p.is_relative_to(home):
                raise ValueError('Array escaped evaluation folder')
            name = p.relative_to(ROOT).as_posix()
            actual = digest(p)
            if expected.get(name) != actual:
                raise ValueError('Array not identical to retrieved bytes: '+name)
            evidence[name] = actual
            with np.load(p, allow_pickle=False) as data:
                source, null, target = [image_array(data[k]).copy() for k in ('source', 'null', 'target')]
                if float(data['source_echo_time_norm']) != float(data['target_echo_time_norm']):
                    raise ValueError('Edited EchoTime')
                if not np.isclose(float(data['source_echo_time_norm']), request['source_echo_time_norm'], rtol=0, atol=1e-12):
                    raise ValueError('EchoTime metadata mismatch')
                for label in ('target_manufacturer', 'target_field_strength'):
                    if int(data[label]) != request[label]:
                        raise ValueError('Target label mismatch')
            identity = (source.tobytes(), request['actual_sampling_seed'], request['direction'],
                        request['condition_support_status'], request['source_echo_time_norm'],
                        request['timestep'], request['guidance'])
            if key in reference and identity != reference[key]:
                raise ValueError('Sources/noise/settings/support differ between profiles')
            reference[key] = identity
            index_rows.append(dict(profile=profile, **request, array_sha256=actual))
            meta = {k: request[k] for k in (*KEYS, 'direction', 'condition_support_status')}
            values = dict(source_null=maps(source, null), source_target=maps(source, target),
                          null_target=maps(null, target))
            for threshold in protocol['thresholds']:
                for region, mask in regions(source, threshold).items():
                    results = {k: measure(v, mask) for k, v in values.items()}
                    for comparison, measured in results.items():
                        rows.append(dict(meta, profile=profile, threshold=threshold, region=region,
                                         comparison=comparison, **measured))
                    difference = {m: results['source_target'][m]-results['source_null'][m] for m in METRICS}
                    rows.append(dict(meta, profile=profile, threshold=threshold, region=region,
                        comparison='target_minus_null_source_error', pixels=int(mask.sum()), valid=mask.sum()>=64,
                        **difference))
            if profile == protocol['reference'] and request['axis'] == 'manufacturer' and request['generation_seed'] == evaluation['selection_generation_seed']:
                panels.append((request, source, null, target))
        print('DIAGNOSED', profile, len(requests), flush=True)
    frame = pd.DataFrame(rows)
    csv_write(output/'case_metrics.csv', frame)
    csv_write(output/'case_index.csv', pd.DataFrame(index_rows))
    summary = summarise(frame, evaluation['bootstrap_seed'], evaluation['bootstrap_resamples'])
    csv_write(output/'patient_intervals.csv', summary)
    # Every selected reference patient is shown; no favourable-example selection.
    fig, axes = plt.subplots(len(panels), 6, figsize=(13, 2*len(panels)), squeeze=False)
    for row, (request, source, null, target) in enumerate(sorted(panels, key=lambda p: p[0]['patient_id'])):
        partition = regions(source)
        mask = partition['interior'].astype(float)+2*partition['boundary']
        images = [source, null, target, np.abs(null-source), np.abs(target-source), mask]
        for col, image in enumerate(images):
            axes[row,col].imshow(image, cmap='viridis' if col==5 else 'gray',
                vmin=0, vmax=2 if col==5 else (.3 if col in (3,4) else 1))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0:
                axes[row,col].set_title(['Source', 'Original-label null', 'Target', '|null-source| (0-.3)', '|target-source| (0-.3)', 'Proxy regions'][col], fontsize=9)
        axes[row,0].set_ylabel(request['patient_id']+'\n'+request['direction'], fontsize=8)
    fig.suptitle('Additive alpha .50 manufacturer; every development patient, selection generation seed\nCanonical source grid; source-only proxy masks; clinical preservation unknown', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(output/'all_reference_patients.png', dpi=110)
    plt.close(fig)
    for module in (__file__, ROOT/'scripts/v5_anatomy/metrics.py', ROOT/'scripts/v8_generator_dev/preservation_screen.py', ROOT/'scripts/v8_generator_dev/io.py'):
        p = Path(module)
        evidence[p.relative_to(ROOT).as_posix()] = digest(p)
    verify_inputs(evidence)
    json_write(output/'inputs.json', evidence)
    json_write(output/'complete.json', dict(complete=True, profiles=len(homes), cases=len(index_rows),
        matched_cases_per_profile=len(matched), metric_rows=len(frame), clinical_preservation='unknown',
        regraded_judges=False, prior_strict_grading_receipt_reused=True, test_pixels_used=False,
        new_training=False, paid_compute=False, historical_files_modified=False))
    (output/'README.md').write_text(
        '# Saved-array source-detail diagnosis\n\nProducer: `python -m scripts.v8_generator_dev.detail_diagnosis --run-id '+run_id+'`.\n\n'
        '`protocol.json`: analysis specification written before metric computation. '
        '`case_index.csv`: every fixed-point request and exact array hash. '
        '`case_metrics.csv`: every comparison, proxy region and threshold, including invalid/empty regions. '
        '`patient_intervals.csv`: equal-patient estimates and 95% bootstrap intervals, direction/support strata at the primary threshold. '
        '`all_reference_patients.png`: every manufacturer reference patient at the selection generation seed. '
        '`inputs.json`: consumed byte hashes; `complete.json`: checks/scope. '
        'SSIM uses region centres but its windows may cross boundaries. Energy ratios describe appearance, not anatomy; '
        'near-zero energy denominators/regions under 64 pixels are missing explicitly. '
        'No actual model inference or new judge grading; no test pixels, annotation truth or fresh confirmation.\n', encoding='utf-8')
    print('DETAIL DIAGNOSIS COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args().run_id)
