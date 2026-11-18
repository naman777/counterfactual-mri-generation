import unittest
import numpy as np
from scripts.v10_detail.verify_saved_v3 import metric_agrees
from scripts.v10_detail.postprocess_v2 import contextual_changes
import pandas as pd


class MissingDiagnosticTests(unittest.TestCase):
    def test_missing_is_retained_and_finite_mismatch_is_rejected(self):
        self.assertTrue(metric_agrees(None,np.nan))
        self.assertTrue(metric_agrees(np.nan,None))
        self.assertTrue(metric_agrees(None,None))
        self.assertFalse(metric_agrees(None,0.))
        self.assertFalse(metric_agrees(0.,np.nan))
        self.assertFalse(metric_agrees(np.inf,np.inf))
        self.assertFalse(metric_agrees(.3,.4))
        self.assertTrue(metric_agrees(.3,.30000001))

    def test_contextual_legacy_schema_without_training_seed_or_region_metrics(self):
        joined=pd.DataFrame(dict(patient_id=['a'],dataset_index=[1],generation_seed=[2],training_seed=[3],
            target_hit_new=[1],target_hit_legacy=[0],interior_mad=[.2]))
        lookup={(1,2,'target'):{'interior_mad':.3}}
        self.assertEqual(contextual_changes(joined,'target_hit',lookup,'target').difference.iloc[0],1)
        self.assertAlmostEqual(contextual_changes(joined,'interior_mad',lookup,'target').difference.iloc[0],-.1)


if __name__=='__main__':unittest.main()
