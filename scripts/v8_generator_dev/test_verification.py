"""Regression tests for literal null controls and complete saved grading groups."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.v8_generator_dev.grading_io import read_grading_csv


class GradingReaderTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([
            dict(case=0, judge='frozen0', variant=variant, target_hit=0,
                 target_probability=0.2, paired_hit_delta=0, proxy_dice=None)
            for variant in ('identity', 'null', 'target')
        ])

    def test_literal_null_survives_and_unknown_proxy_stays_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'grading.csv'
            self.table().to_csv(path, index=False)
            result = read_grading_csv(path)
            self.assertEqual(result.variant.tolist(), ['identity', 'null', 'target'])
            self.assertTrue(result.proxy_dice.isna().all())

    def test_missing_or_duplicate_control_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'grading.csv'
            for frame in (self.table().iloc[:2], pd.concat([self.table(), self.table().iloc[[1]]])):
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'three distinct controls'):
                    read_grading_csv(path)


if __name__ == '__main__':
    unittest.main()
