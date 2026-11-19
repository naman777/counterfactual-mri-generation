"""Outcome-independent pairing and patient-weighted fixed-point comparisons."""
import unittest

import pandas as pd

from scripts.v8_generator_dev.analyse_full import compare_rows, primary_rows
from scripts.v8_generator_dev.full_protocol import read_config


class ComparisonTests(unittest.TestCase):
    def data(self):
        config = read_config('configs/generator_full_v2.json')
        config['evaluation']['bootstrap_resamples'] = 100
        rows = []
        seeds = [config['evaluation']['selection_generation_seed'], *config['evaluation']['confirmation_generation_seeds']]
        for axis, point in config['evaluation']['fixed_settings'].items():
            for i in range(2):
                for seed in seeds:
                    for control in ('identity', 'null', 'target'):
                        rows.append(dict(patient_id=f'p{i}', dataset_index=i, axis=axis,
                                         direction=f'{i}_to_{1-i}', generation_seed=seed, judge='frozen0',
                                         variant=control, timestep=point['timestep'], guidance=point['guidance'],
                                         condition_support_status='interpolation_only', target_hit=0,
                                         paired_hit_delta=0, ssim=0.8, mad=0.1, other_axis_stable=1))
        return pd.DataFrame(rows), config

    def test_patient_paired_contrast_retains_direction_and_controls(self):
        r0, config = self.data()
        r1 = r0.copy()
        r1.loc[r1.variant == 'target', ['target_hit', 'paired_hit_delta']] = 1
        result = compare_rows(primary_rows(r0, config), primary_rows(r1, config), config)
        target = result[(result.control == 'target') & (result.metric == 'target_hit') &
                        (result.stratum == 'all_flagged_and_in_range')]
        self.assertEqual(len(target), 2)
        self.assertTrue(target['mean'].eq(1).all())
        self.assertTrue(target.patients.eq(2).all())
        null = result[(result.control == 'null') & (result.metric == 'target_hit')]
        self.assertTrue(null['mean'].eq(0).all())

    def test_missing_seed_or_unmatched_source_cannot_silently_compare(self):
        r0, config = self.data()
        with self.assertRaisesRegex(ValueError, 'missing a declared generation seed'):
            primary_rows(r0[r0.generation_seed != config['evaluation']['selection_generation_seed']], config)
        r1 = r0.copy()
        r1.loc[0, 'patient_id'] = 'different_patient'
        with self.assertRaisesRegex(ValueError, 'do not match'):
            compare_rows(r0, r1, config)


if __name__ == '__main__':
    unittest.main()
