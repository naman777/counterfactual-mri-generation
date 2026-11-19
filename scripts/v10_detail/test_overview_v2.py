"""Integer warning flags must select flagged rows, never positional row labels."""
import unittest
import pandas as pd
from scripts.v10_detail.write_overview_v2 import warning_ids


class WarningIdentityTest(unittest.TestCase):
    def test_flags_independent_of_index_labels(self):
        frame = pd.DataFrame(dict(patient_id=['clear0','warn1','clear1','warn2'], qc_warning=[0,1,0,1]),
                             index=[0,45,1,7])
        self.assertEqual(warning_ids(frame), ['warn1','warn2'])
        self.assertEqual(warning_ids(frame.assign(qc_warning=frame.qc_warning.astype(bool))), ['warn1','warn2'])
        with self.assertRaises(ValueError):
            warning_ids(frame.assign(qc_warning=[0,1,None,1]))


if __name__ == '__main__':
    unittest.main()
