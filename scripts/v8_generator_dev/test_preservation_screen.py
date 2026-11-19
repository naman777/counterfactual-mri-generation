import unittest

import numpy as np

from scripts.v8_generator_dev.preservation_screen import local_rank, measurements, screen, controls, METRICS


class PreservationScreenTests(unittest.TestCase):
    def test_rank_invariant_to_strictly_monotonic_contrast_and_identity_zero(self):
        image = np.random.default_rng(3).uniform(.05, .95, (32, 32)).astype(np.float32)
        np.testing.assert_array_equal(local_rank(image), local_rank(image**.7))
        result = measurements(image, image)
        self.assertTrue(result['screen_measurements_valid'])
        for name in METRICS:
            self.assertEqual(result[name], 0)
        self.assertEqual(screen(result, dict.fromkeys(METRICS, 1))['screen_flag'], 0)

    def test_empty_image_does_not_pass_and_translation_moves_without_wrap(self):
        blank = np.zeros((32, 32), np.float32)
        result = measurements(blank, blank)
        self.assertIsNone(screen(result, dict.fromkeys(METRICS, 1))['within_calibrated_envelope'])
        image = blank.copy()
        image[8:20, 8:20] = .8
        moved = controls(image, 4)['translate_4px']
        self.assertEqual(float(moved[:4].max()), 0)
        self.assertGreater(measurements(image, moved)['edge_chamfer_px'], 0)


if __name__ == '__main__':
    unittest.main()
