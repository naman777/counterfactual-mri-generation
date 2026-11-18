import unittest

import numpy as np

from scripts.v8_generator_dev.preservation_screen_v2 import measurements, controls


class LocalTextureTests(unittest.TestCase):
    def test_identity_and_gain_preserve_normalised_texture_and_erasure_loses_it(self):
        y, x = np.indices((64, 64))
        image = np.zeros((64, 64), np.float32)
        image[8:56, 8:56] = (.4+.1*np.sin(x[8:56, 8:56])+ .1*np.cos(y[8:56, 8:56])).astype(np.float32)
        self.assertLess(measurements(image, image)['local_texture_deficit'], 1e-8)
        self.assertLess(measurements(image, image*.8)['local_texture_deficit'], 1e-5)
        erased = controls(image, 4)['erase_5px']
        self.assertGreater(measurements(image, erased)['local_texture_deficit'], .99)


if __name__ == '__main__':
    unittest.main()
