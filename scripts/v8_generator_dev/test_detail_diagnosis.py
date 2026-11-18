"""Meaningful failure/synthetic controls for the new regional diagnosis."""
import unittest
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scripts.v8_generator_dev.detail_diagnosis import maps, measure, regions, summarise


class DetailDiagnosisTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.source = np.zeros((128,128), np.float32)
        self.source[20:108,20:108] = rng.uniform(.15,.85,(88,88))

    def test_partition_is_disjoint_complete(self):
        masks = regions(self.source)
        count = sum(masks[k].astype(int) for k in ('interior','boundary','background'))
        np.testing.assert_array_equal(count, np.ones_like(count))

    def test_identity_and_blur(self):
        mask = regions(self.source)['interior']
        identity = measure(maps(self.source, self.source), mask)
        self.assertEqual(identity['mad'], 0)
        self.assertEqual(identity['ssim'], 1)
        self.assertEqual(identity['hf_energy_ratio'], 1)
        blurred = measure(maps(self.source, ndi.gaussian_filter(self.source,1)), mask)
        self.assertLess(blurred['hf_energy_ratio'], .2)
        self.assertGreater(blurred['hf_residual_mad'], 0)

    def test_copied_background_cannot_repair_interior_score(self):
        masks = regions(self.source)
        edit = ndi.gaussian_filter(self.source, 2)
        copied = edit.copy()
        copied[masks['background']] = self.source[masks['background']]
        # Direct pixel error is strictly local; SSIM/HF windows are disclosed separately.
        a = measure(maps(self.source, edit), masks['interior'])
        b = measure(maps(self.source, copied), masks['interior'])
        self.assertEqual(a['mad'], b['mad'])

    def test_empty_regions_and_energy_are_missing(self):
        values = maps(np.zeros((128,128),np.float32), np.zeros((128,128),np.float32))
        self.assertFalse(measure(values, np.zeros((128,128),bool))['valid'])
        self.assertTrue(np.isnan(measure(values, np.ones((128,128),bool))['hf_energy_ratio']))

    def test_patient_bootstrap_does_not_weight_extra_seed_rows(self):
        base = dict(profile='test', axis='manufacturer', threshold=.05, region='whole',
                    comparison='source_null', direction='0_to_1', condition_support_status='flagged')
        records=[]
        for patient, values in [('a',[0,0,0]),('b',[1])]:
            records.extend(dict(base, patient_id=patient, **{k:v for k in
                ('mad','ssim','rank_mad','hf_residual_mad','hf_energy_ratio')}) for v in values)
        result = summarise(pd.DataFrame(records), 19, 100)
        self.assertTrue(result.estimate.eq(.5).all())


if __name__ == '__main__':
    unittest.main()
