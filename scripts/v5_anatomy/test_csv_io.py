"""Regression checks for null controls and genuinely absent human labels."""
import io
import unittest
import pandas as pd
from scripts.v5_anatomy.csv_io import read_table


class CsvTests(unittest.TestCase):
    def test_null_control_is_not_missing_but_empty_human_label_is(self):
        frame=read_table(io.StringIO('variant,human_preservation_pass\nnull,\nsdedit,\n'))
        self.assertEqual(frame.variant.tolist(),['null','sdedit'])
        self.assertTrue(frame.human_preservation_pass.isna().all())

    def test_null_mapping_preserved_with_string_dtype(self):
        frame=read_table(io.StringIO('review_id,variant\na,null\n'),dtype=str)
        self.assertEqual(frame.iloc[0].variant,'null')


if __name__=='__main__':
    unittest.main(verbosity=2)
