"""Additional local metadata-integrity audit; no generation or gate changes."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def compare_rows(requests, grades, judges):
    """Every grade must carry the exact request metadata, including missingness."""
    if requests.case.duplicated().any() or set(requests.case) != set(range(len(requests))):
        raise ValueError('Non-unique/non-contiguous case identities')
    keys = ['case', 'judge', 'variant']
    expected = pd.MultiIndex.from_product([requests.case, judges, ['identity', 'null', 'target']], names=keys)
    observed = pd.MultiIndex.from_frame(grades[keys])
    if observed.has_duplicates or len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError('Missing/duplicate/unexpected grade identities')
    shared = [c for c in requests if c != 'case']
    if not set(shared).issubset(grades.columns):
        raise ValueError('Grade request metadata missing')
    aligned = requests.set_index('case').loc[grades.case, shared].reset_index(drop=True)
    actual = grades[shared].reset_index(drop=True)
    agrees = actual.eq(aligned) | (actual.isna() & aligned.isna())
    if not agrees.all().all():
        raise ValueError('Grade/request metadata drift: ' + ', '.join(agrees.columns[~agrees.all()].tolist()))


def verify(prepared, results, run_id):
    prepared, results = map(project_path, (prepared, results))
    p = json.loads((prepared / 'protocol.json').read_text())
    receipt = json.loads((results / 'complete.json').read_text())
    if not receipt['workflow_success'] or len(receipt['evaluations']) != 51:
        raise ValueError('Complete frozen evaluation matrix required')
    selected = pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int).tolist()
    position = {index: j for j, index in enumerate(selected)}
    val = CanonicalDataset(p['view'], 'val')
    judges = [f'heldout_orientation_robust_seed{j}' for j in range(3)]
    inputs = {}; cases = 0; grades = 0
    for profile, relative in receipt['evaluations'].items():
        home = project_path(relative)
        requests = pd.read_csv(home / 'requests.csv', keep_default_na=False, na_values=[''])
        table = read_grading_csv(home / 'per_image.csv')
        compare_rows(requests, table, judges)
        meta = json.loads((home / 'protocol.json').read_text())
        for r in requests.itertuples():
            item = val[r.dataset_index]
            original = item['manufacturer_label'] if r.axis == 'manufacturer' else item['field_strength_label']
            expected_guide = selected[(position[r.dataset_index] + 1) % len(selected)] if r.guide_control == 'wrong' else r.dataset_index
            if (r.patient_id != item['patient_id'] or r.direction != f'{original}_to_{1-original}'
                    or r.guide_dataset_index != expected_guide
                    or r.guide_patient_id != val.frame.iloc[expected_guide].patient_id):
                raise ValueError('Source/direction/guide identity drift')
            for field in ('mode', 'axis', 'guide_control', 'training_seed'):
                if getattr(r, field) != meta[field]:
                    raise ValueError('Request/profile metadata drift: ' + field)
            if not all(np.isclose(getattr(r, field), float(item['echo_time_norm']), rtol=0, atol=1e-8)
                       for field in ('source_echo_time_norm', 'target_echo_time_norm')):
                raise ValueError('EchoTime differs from the actual source value')
            path = home / r.array_file
            if digest(path) != r.array_sha256:
                raise ValueError('Array hash differs from the immutable request')
            inputs[path.relative_to(ROOT).as_posix()] = r.array_sha256
            with np.load(path, allow_pickle=False) as z:
                if (int(z['target_manufacturer']) != r.target_manufacturer
                        or int(z['target_field_strength']) != r.target_field_strength):
                    raise ValueError('Array/request target-label drift')
                for field in ('source_echo_time_norm', 'target_echo_time_norm'):
                    if not np.isclose(float(z[field]), float(item['echo_time_norm']), rtol=0, atol=1e-8):
                        raise ValueError('Array EchoTime differs from the actual source value')
        for filename in ('requests.csv', 'per_image.csv', 'protocol.json'):
            path = home / filename; inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        cases += len(requests); grades += len(table)
    output = ROOT / 'reports/v10_detail' / run_id
    output.mkdir(parents=True, exist_ok=False)
    for path in (project_path(__file__), prepared / 'protocol.json', prepared / 'validation_selection.csv', results / 'complete.json'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, profiles=51, cases=cases, grades=grades,
        request_grade_metadata_equal=True, source_echo_value_checked=True, source_guide_target_identities_checked=True,
        generation_not_replayed=True, scientific_gates_unchanged=True, paid_gpu_calls=0))
    (output / 'README.md').write_text(
        'Producer: scripts.v10_detail.verify_metadata. inputs.json: immutable metadata files and checker source; '
        'complete.json: every source/guide/direction/array-target identity, actual-source fixed EchoTime and '
        'complete request-to-grade metadata/identity correspondence. This supplements saved-image/judge regrading; '
        'it does not replay sampling, change gates or establish clinical preservation.\n', encoding='utf-8')
    print('METADATA VERIFIED', cases, grades, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ('prepared', 'results', 'run-id'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); verify(a.prepared, a.results, a.run_id)
