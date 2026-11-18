"""Reject independently introduced identity and metadata corruption."""
import unittest
import numpy as np
import pandas as pd
from scripts.v10_detail.verify_metadata import compare_rows


class MetadataChecks(unittest.TestCase):
    def setUp(self):
        self.requests = pd.DataFrame(dict(case=[0, 1], patient_id=['p1', 'p2'], echo=[.1, .2], missing=[np.nan, 3.]))
        self.grades = pd.DataFrame([dict(row, judge=j, variant=v)
            for row in self.requests.to_dict('records') for j in ['j0', 'j1', 'j2'] for v in ['identity', 'null', 'target']])

    def test_metadata_and_missingness_corruption(self):
        compare_rows(self.requests, self.grades, ['j0', 'j1', 'j2'])
        for column, value in [('patient_id', 'wrong'), ('echo', .9), ('missing', 0.)]:
            with self.subTest(column=column):
                corrupted = self.grades.copy(); corrupted.loc[0, column] = value
                with self.assertRaises(ValueError):
                    compare_rows(self.requests, corrupted, ['j0', 'j1', 'j2'])

    def test_grade_identity_corruption(self):
        for corrupted in (self.grades.iloc[:-1], pd.concat([self.grades.iloc[:-1], self.grades.iloc[[0]]]),
                          self.grades.assign(case=2)):
            with self.assertRaises(ValueError):
                compare_rows(self.requests, corrupted, ['j0', 'j1', 'j2'])


if __name__ == '__main__':
    unittest.main()
