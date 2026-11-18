"""Failure recovery retains training status, pairing and a single allowance."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.v8_generator_dev.failed_pair import allowance, select_failed_pair


class FailedPairTests(unittest.TestCase):
    def test_budget_debits_every_launch_and_rejects_invalid_values(self):
        first = allowance([100, 5000], 4900)
        self.assertLess(allowance([100, 5000, 200], 4900)['remaining_usd'], first['remaining_usd'])
        self.assertEqual(allowance([20000], 100)['remaining_usd'], 0)
        for values in ([], [-1], [float('nan')]):
            with self.assertRaises(ValueError):
                allowance(values, 100)

    def test_requires_failure_and_last_complete_matched_pair(self):
        failure = dict(complete=False, phases=[dict(phase='train', exit_code=1)])
        log = 'FloatingPointError: Nonfinite full-training gradient'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for epoch in (5, 10):
                block = root / f'block_{epoch:04d}'
                block.mkdir()
                progress = dict(epoch=epoch, sample_offset=0, global_step=epoch * 3, applied_steps=epoch * 3)
                record = dict(engineering=False, matched_sources_timesteps_noise=True,
                              progress={variant: dict(progress) for variant in ('additive', 'scale_shift')},
                              checkpoints={'additive': 'a.pt', 'scale_shift': 'b.pt'},
                              checkpoint_hashes={'additive': 'a', 'scale_shift': 'b'})
                (block / 'complete.json').write_text(json.dumps(record))
            (root / 'block_0015').mkdir()
            result = select_failed_pair(root, failure, log)
            self.assertEqual(result['common_epoch'], 10)
            self.assertFalse(result['full_training_completed'])
            self.assertFalse(result['converged'])
            with self.assertRaises(ValueError):
                select_failed_pair(root, failure, 'unknown error')
            record['progress']['additive']['epoch'] = 11
            (root / 'block_0010/complete.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Unmatched'):
                select_failed_pair(root, failure, log)


if __name__ == '__main__':
    unittest.main()
