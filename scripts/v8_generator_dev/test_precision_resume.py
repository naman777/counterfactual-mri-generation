"""Strict migration rejection and actual-model continuation with retained state."""
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.v8_generator_dev.full_train import TrainingEngine
from scripts.v8_generator_dev.io import ROOT, digest
from scripts.v8_generator_dev.numerical_replay import equal_state, update_state
from scripts.v8_generator_dev.prepare_precision_resume import migrate_payload, resumed_config, training_state_equal
from scripts.v8_generator_dev.test_full import SyntheticTraining, fixture_config


class PrecisionResumeTests(unittest.TestCase):
    def test_strict_loader_rejects_silent_change_then_explicit_migration_preserves_continuation(self):
        torch.set_num_threads(2)
        config, dataset = fixture_config(), SyntheticTraining()
        for variant in ('additive', 'scale_shift'):
            original = TrainingEngine(config, variant, dataset, 'old-protocol')
            original.train_batch()
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                parent, migrated_path = Path(folder)/'parent.pt', Path(folder)/'migrated.pt'
                original.save(parent)
                parent_hash = digest(parent)
                candidate = resumed_config(config)
                with self.assertRaisesRegex(ValueError, 'Resume incompatibility: config'):
                    TrainingEngine(candidate, variant, dataset, 'new-protocol', resume=parent)
                old = torch.load(parent, map_location='cpu', weights_only=False)
                payload = migrate_payload(old, candidate, 'new-protocol', parent_hash)
                self.assertTrue(training_state_equal(old, payload))
                with migrated_path.open('xb') as handle:
                    torch.save(payload, handle)
                direct = TrainingEngine(config, variant, dataset, 'old-protocol', resume=parent)
                expected = direct.train_batch()
                state = update_state(direct)
                resumed = TrainingEngine(candidate, variant, dataset, 'new-protocol', resume=migrated_path)
                actual = resumed.train_batch()
                self.assertTrue(equal_state(state, update_state(resumed)))
                for key in ('sample_indices', 'timesteps', 'noise_sha256', 'noise_mse'):
                    self.assertEqual(expected[key], actual[key])
                self.assertEqual(digest(parent), parent_hash)

    def test_reject_unrelated_scientific_change(self):
        original = dict(config=fixture_config())
        candidate = resumed_config(original['config'])
        candidate['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'Only declared precision policy'):
            migrate_payload(original, candidate, 'new-protocol', 'parent-hash')


if __name__ == '__main__':
    unittest.main()
