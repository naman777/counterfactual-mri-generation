import copy
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.v8_generator_dev.continuation import proposed_config, authorised_config, continuation_payload
from scripts.v8_generator_dev.full_train import TrainingEngine
from scripts.v8_generator_dev.io import ROOT, digest
from scripts.v8_generator_dev.numerical_replay import equal_state, update_state
from scripts.v8_generator_dev.test_full import SyntheticTraining, fixture_config


class ContinuationTests(unittest.TestCase):
    def test_requires_new_budget_and_rejects_scientific_drift(self):
        original = fixture_config()
        proposed = proposed_config(original)
        self.assertFalse(proposed['resources']['monetary_budget_authorised'])
        with self.assertRaises(ValueError):
            authorised_config(original, proposed, 0)
        changed = copy.deepcopy(proposed)
        changed['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'undeclared scientific'):
            authorised_config(original, changed, 6)
        approved = authorised_config(original, proposed, 6)
        for key in ('seed', 'data', 'view', 'model', 'condition', 'diffusion', 'full_training', 'evaluation'):
            self.assertEqual(approved[key], original[key])

    def test_actual_resume_retains_state_after_precision_and_resource_binding(self):
        torch.set_num_threads(2)
        config, dataset = fixture_config(), SyntheticTraining()
        for variant in ('additive', 'scale_shift'):
            engine = TrainingEngine(config, variant, dataset, 'original')
            engine.train_batch()
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                parent, migrated = Path(folder)/'parent.pt', Path(folder)/'migrated.pt'
                engine.save(parent)
                approved = authorised_config(config, proposed_config(config), 6)
                original = torch.load(parent, map_location='cpu', weights_only=False)
                payload = continuation_payload(original, approved, 'new-protocol', digest(parent), 6)
                with migrated.open('xb') as handle:
                    torch.save(payload, handle)
                direct = TrainingEngine(config, variant, dataset, 'original', resume=parent)
                expected = direct.train_batch()
                expected_state = update_state(direct)
                resumed = TrainingEngine(approved, variant, dataset, 'new-protocol', resume=migrated)
                actual = resumed.train_batch()
                self.assertTrue(equal_state(expected_state, update_state(resumed)))
                self.assertEqual(expected['noise_sha256'], actual['noise_sha256'])
                self.assertEqual(expected['sample_indices'], actual['sample_indices'])


if __name__ == '__main__':
    unittest.main()
