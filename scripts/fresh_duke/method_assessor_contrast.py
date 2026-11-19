"""Compare the epoch-10 diffusion advantage over frozen histogram matching."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_stats import PRIMARY_JUDGES, patient_estimate


def analyse(source):
    records = pd.read_csv(source / 'per_slice.csv')
    keys = ['slice_id', 'patient_id', 'mfg', 'fs']
    records = records[records.judge.isin(PRIMARY_JUDGES)
                      & records.arm.isin(['target', 'train_histogram'])]
    wide = records.pivot(index=keys, columns=['judge', 'arm'], values='hit')
    if wide.shape != (300, 4) or wide.isna().any().any():
        raise ValueError('Expected four paired decisions on each of 300 sources')
    rows = wide.index.to_frame(index=False)
    if not rows.slice_id.is_unique or not rows.groupby('patient_id').size().eq(5).all():
        raise ValueError('Expected five distinct sources per patient')
    reference = pd.read_csv(source / 'control_differences.csv')
    results = []
    for judge in PRIMARY_JUDGES:
        rows[judge] = (wide[judge, 'target'].astype(int)
                       - wide[judge, 'train_histogram'].astype(int)).to_numpy()
        estimate = patient_estimate(rows, judge, True)
        archived = reference[(reference.judge == judge) & (reference.subset == 'all')
                             & (reference.control == 'train_histogram')].iloc[0]
        for key in ['estimate', 'lo95', 'hi95']:
            np.testing.assert_allclose(estimate[key], archived[key])
        results.append({'contrast': judge, **estimate})
    rows['interaction'] = rows[PRIMARY_JUDGES[0]] - rows[PRIMARY_JUDGES[1]]
    results.append({'contrast': 'original_minus_heldout_method_advantage',
                    **patient_estimate(rows, 'interaction', True)})
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyse(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, mode='x')
