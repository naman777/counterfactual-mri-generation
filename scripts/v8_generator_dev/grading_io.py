"""Read saved grading controls without turning literal 'null' into missing data."""
import numpy as np
import pandas as pd


def read_grading_csv(path):
    frame = pd.read_csv(path, keep_default_na=False, na_values=[''],
                        dtype={'variant': str, 'judge': str})
    required = {'case', 'judge', 'variant', 'target_hit', 'target_probability', 'paired_hit_delta'}
    if required - set(frame) or frame.empty:
        raise ValueError('Missing grading schema or empty grading table')
    if not frame.variant.isin(['identity', 'null', 'target']).all():
        raise ValueError('Unknown/missing grading control')
    for _, group in frame.groupby(['case', 'judge']):
        if len(group) != 3 or set(group.variant) != {'identity', 'null', 'target'}:
            raise ValueError('Each case/judge must retain all three distinct controls')
    if not frame.target_hit.isin([0, 1]).all() or not frame.paired_hit_delta.isin([-1, 0, 1]).all():
        raise ValueError('Invalid grading outcomes')
    probabilities = pd.to_numeric(frame.target_probability, errors='raise')
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError('Invalid target probabilities')
    return frame
