"""Guide invariance claims, spatial sensitivity and degenerate-proxy checks."""
import unittest

import numpy as np

from scripts.v8_generator_dev.guide_audit import guides


class GuideTests(unittest.TestCase):
    def test_rank_guide_preserves_monotone_intensity_order(self):
        image = np.random.default_rng(42).uniform(0.01, 0.99, (128, 128))
        before, _ = guides(image)
        after, _ = guides(image ** 0.7)
        np.testing.assert_array_equal(before['canonical_rank_gradients'], after['canonical_rank_gradients'])

    def test_rank_guide_responds_to_spatial_rearrangement(self):
        image = np.random.default_rng(7).uniform(0, 1, (128, 128))
        before, _ = guides(image)
        after, _ = guides(np.roll(image, (8, 8), (0, 1)))
        self.assertGreater(np.abs(before['canonical_rank_gradients'] - after['canonical_rank_gradients']).mean(), 0)

    def test_degenerate_proxy_is_flagged_not_certified(self):
        maps, valid = guides(np.zeros((128, 128)))
        self.assertFalse(any(valid))
        self.assertTrue(np.isfinite(maps['canonical_proxy_distance']).all())
        self.assertTrue((maps['canonical_proxy_distance'] == 0).all())


if __name__ == '__main__':
    unittest.main()
