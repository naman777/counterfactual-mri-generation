"""Check whether existing train/val metadata can repair direction support, without resampling."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path


def audit(prepared, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new audit ID required')
    prepared = project_path(prepared)
    config = json.loads((prepared / 'protocol.json').read_text())['config']
    train, val = [CanonicalDataset(config['view'], split) for split in ('train', 'val')]
    selected = set(pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int))
    frames, rows = {'train': train.frame, 'val': val.frame}, []
    eligible_status = set(config['evaluation']['selection_support_statuses'])
    for split, frame in frames.items():
        for index, source in frame.iterrows():
            # Match the actual engine's float32 conditioning, not a changed EchoTime.
            echo = float(np.float32(source.echo_time_norm))
            for axis in ('manufacturer', 'field_strength'):
                m, f = int(source.manufacturer_label), int(source.field_strength_label)
                original = m if axis == 'manufacturer' else f
                if axis == 'manufacturer':
                    m = 1-m
                else:
                    f = 1-f
                support = request_support(train.frame, m, f, echo, config['data']['minimum_cell_patients'])
                rows.append(dict(split=split, patient_id=source.patient_id, dataset_index=index,
                    axis=axis, direction=f'{original}_to_{1-original}', source_echo_time_norm=echo,
                    target_manufacturer=m, target_field_strength=f,
                    frozen_selected=split == 'val' and index in selected,
                    eligible=support['condition_support_status'] in eligible_status, **support))
    table = pd.DataFrame(rows)
    coverage = []
    for (split, axis, direction), group in table.groupby(['split', 'axis', 'direction']):
        anywhere = group[group.eligible].patient_id.nunique()
        frozen = group[group.eligible & group.frozen_selected].patient_id.nunique() if split == 'val' else None
        coverage.append(dict(split=split, axis=axis, direction=direction,
            source_patients=group.patient_id.nunique(), any_existing_slice_eligible_patients=anywhere,
            frozen_selected_eligible_patients=frozen,
            minimum_required_patients=config['evaluation']['minimum_eligible_patients_per_direction'],
            deficit_even_if_any_slice_allowed=max(0, config['evaluation']['minimum_eligible_patients_per_direction']-anywhere),
            selection_unchanged=True))
    coverage = pd.DataFrame(coverage)
    cells = []
    for (m, f), group in train.frame.groupby(['manufacturer_label', 'field_strength_label']):
        cells.append(dict(manufacturer_label=int(m), field_strength_label=int(f), patients=group.patient_id.nunique(),
            slices=len(group), echo_norm_min=group.echo_time_norm.min(), echo_norm_max=group.echo_time_norm.max(),
            distinct_echo_times=group.echo_time_norm.nunique()))
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    csv_write(output / 'all_source_requests.csv', table)
    csv_write(output / 'direction_coverage.csv', coverage)
    csv_write(output / 'training_cells.csv', pd.DataFrame(cells))
    unique_keys = ['split', 'patient_id', 'axis', 'direction', 'source_echo_time_norm', 'target_manufacturer',
                   'target_field_strength', 'condition_support_status', 'eligible']
    csv_write(output / 'patient_conditions.csv', table[unique_keys].drop_duplicates())
    inputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in (prepared / 'protocol.json',
        prepared / 'validation_selection.csv', ROOT / 'scripts/v8_generator_dev/audit_support_coverage.py',
        ROOT / 'scripts/v7_geometry/support.py')}
    for split in ('train', 'val'):
        for p in (ROOT / config['view'] / 'splits' / f'{split}.csv', ROOT / 'data/processed/ax_t1_128/splits' / f'{split}.csv'):
            inputs[p.relative_to(ROOT).as_posix()] = digest(p)
    json_write(output / 'inputs.json', inputs)
    bottleneck = coverage[(coverage.split == 'val') & (coverage.axis == 'field_strength') & (coverage.direction == '0_to_1')].iloc[0]
    notes = ['# Existing-data condition-support audit', '',
        'This is a metadata-only feasibility audit, not a new evaluation-source selection or editing experiment. '
        'It checks every existing train/validation slice against the unchanged training-only support rule '
        'with EchoTime fixed at its source value. Test metadata/pixels and image pixels were not read.', '',
        f'Field-strength 0-to-1 validation coverage: the frozen selection has {int(bottleneck.frozen_selected_eligible_patients)} '
        f'eligible patient(s); allowing any existing validation slice still gives '
        f'{int(bottleneck.any_existing_slice_eligible_patients)}. The frozen minimum is '
        f'{int(bottleneck.minimum_required_patients)}.', '',
        'Repeating or choosing more slices cannot create additional independent eligible patients here. '
        'Do not lower the coverage rule, edit EchoTime, move patients across splits or use test patients '
        'to repair development support. Audit compatible additional data or explicitly narrow the task '
        'under a separately frozen future protocol. Interpolation eligibility is an engineering criterion; '
        'it does not establish causal scanner identifiability or anatomy preservation.', '',
        '`direction_coverage.csv` gives both axes/directions for train/val; `training_cells.csv` gives '
        'training patient counts and fixed-conditioning ranges. `all_source_requests.csv` retains all slice '
        'requests, and `patient_conditions.csv` deduplicates acquisition-condition rows. '
        '`inputs.json` binds code and metadata. Producer: `python -m scripts.v8_generator_dev.audit_support_coverage`.', '']
    (output / 'README.md').write_text('\n'.join(notes), encoding='utf-8')
    json_write(output / 'complete.json', dict(complete=True, metadata_only=True, source_selection_changed=False,
        frozen_fs_forward_eligible_patients=int(bottleneck.frozen_selected_eligible_patients),
        any_existing_fs_forward_eligible_patients=int(bottleneck.any_existing_slice_eligible_patients),
        test_metadata_used=False, image_pixels_used=False, new_data_downloaded=False, anatomy_validated=False))
    print('SUPPORT AUDIT', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    audit(args.prepared, args.run_id)
