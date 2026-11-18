"""Meaningful checks of observer transparency, draw identity and failure snapshots."""
import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from scripts.v8_generator_dev.io import ROOT
from scripts.v8_generator_dev.numerical_replay import (
    draw, equal_state, isolated_step, observe_clipping, rng_state, snapshot_before_failure, update_state,
)
from scripts.v8_generator_dev.test_full import SyntheticTraining, fixture_config


class NumericalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the actual original trainer, before the later precision-policy edit.
        path = ROOT / 'reports/v8_generator_dev/full_bundle_20261116_01/inputs.zip'
        with zipfile.ZipFile(path) as archive:
            code = archive.read('scripts/v8_generator_dev/full_train.py')
        namespace = {'__name__': 'frozen_diagnostic_training', '__file__': str(path) + '/full_train.py'}
        exec(compile(code, namespace['__file__'], 'exec'), namespace)
        cls.Engine = namespace['TrainingEngine']

    def setUp(self):
        torch.set_num_threads(2)
        self.config, self.data = fixture_config(), SyntheticTraining()

    def engine(self):
        return self.Engine(self.config, 'additive', self.data, 'diagnostic-test')

    def test_observer_preserves_exact_updates_draws_and_rng(self):
        direct = self.engine()
        expected = direct.train_batch()
        state = update_state(direct)
        observed = self.engine()
        records = []
        with observe_clipping(records):
            actual = observed.train_batch()
        for key in ('noise_sha256', 'sample_indices', 'timesteps', 'noise_mse', 'grad_norm'):
            self.assertEqual(actual[key], expected[key])
        self.assertTrue(equal_state(update_state(observed), state))
        self.assertEqual(records, [])

    def test_isolated_control_matches_original_raw_optimizer_update(self):
        direct = self.engine()
        expected = direct.train_batch()
        state = update_state(direct)
        isolated = self.engine()
        before = rng_state('cpu')
        draw(isolated)
        self.assertTrue(equal_state(before, rng_state('cpu')))
        actual = isolated_step(isolated)
        self.assertTrue(equal_state(update_state(isolated), state))
        for key in ('noise_sha256', 'sample_indices', 'timesteps'):
            self.assertEqual(actual['draw'][key], expected[key])
        self.assertEqual(actual['loss'], expected['noise_mse'])
        self.assertEqual(isolated.progress['global_step'], 0)

    def test_failed_gradient_snapshot_retains_pre_step_raw_optimizer_rng(self):
        engine = self.engine()
        before = update_state(engine)
        progress, scaler = dict(engine.progress), copy.deepcopy(engine.scaler.state_dict())
        random_state = rng_state('cpu')
        hook = list(engine.u.parameters())[-1].register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        observed = []
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite full-training gradient'):
            with observe_clipping(observed):
                engine.train_batch()
        hook.remove()
        self.assertFalse(observed[0]['all_individual_elements_finite'])
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'pre_step.pt'
            snapshot_before_failure(engine, path, random_state, progress, scaler)
            restored = self.Engine(self.config, 'additive', self.data, 'diagnostic-test', resume=path)
            self.assertTrue(equal_state(update_state(restored), before))
            self.assertEqual(restored.progress, progress)


if __name__ == '__main__':
    unittest.main()
