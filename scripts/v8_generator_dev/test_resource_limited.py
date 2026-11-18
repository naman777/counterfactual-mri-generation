"""Recovery must preserve common training exposure and the original allowance."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.v8_generator_dev.resource_limited import choose_common, remaining_allowance


class ResourceRecoveryTests(unittest.TestCase):
    def test_remainder_never_resets_approved_total(self):
        first = remaining_allowance({'seconds': 3600})
        later = remaining_allowance({'seconds': 7200})
        self.assertLess(first['remaining_usd'], 6)
        self.assertLess(later['remaining_usd'], first['remaining_usd'])
        self.assertEqual(remaining_allowance({'seconds': 20000})['remaining_usd'], 0)

    def test_selects_last_common_block_not_unmatched_latest_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'resource_stop.json').write_text('{}')
            for epoch in (5, 10):
                block = root / f'block_{epoch:04d}'
                block.mkdir()
                progress = dict(epoch=epoch, sample_offset=0, global_step=epoch * 3, applied_steps=epoch * 3)
                (block / 'complete.json').write_text(json.dumps(dict(
                    engineering=False, matched_sources_timesteps_noise=True,
                    progress={'additive': progress, 'scale_shift': progress})))
            self.assertEqual(choose_common(root)['common_epoch'], 10)
            (root / 'complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'incomplete resource-stop'):
                choose_common(root)


if __name__ == '__main__':
    unittest.main()
