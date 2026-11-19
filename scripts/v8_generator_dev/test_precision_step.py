"""Actual CPU AMP overflow must replay dropout and apply exactly one update."""
import copy
import json
import unittest

import torch

from scripts.v8_generator_dev.precision_step import precision_step


class PrecisionStepTests(unittest.TestCase):
    def test_new_preparation_allows_only_declared_precision_policy_change(self):
        from scripts.v8_generator_dev.full_protocol import read_config
        from scripts.v8_generator_dev.io import ROOT
        from scripts.v8_generator_dev.prepare_full import validate_baseline
        config = read_config('configs/generator_full_v2_precision.json')
        baseline = json.loads((ROOT / 'configs/generator_dev_v1.json').read_text())
        policy = validate_baseline(config, baseline)
        self.assertEqual(policy['amp_gradient_retries_before'], 0)
        self.assertEqual(policy['amp_gradient_retries_after'], 8)
        config['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'verified baseline: training'):
            validate_baseline(config, baseline)

    def test_training_engine_retains_one_draw_and_ema_update_after_overflow(self):
        from scripts.v8_generator_dev.full_train import TrainingEngine
        from scripts.v8_generator_dev.test_full import SyntheticTraining, fixture_config
        torch.set_num_threads(2)
        config = fixture_config()
        config['training']['amp_gradient_retries'] = 16
        data = SyntheticTraining()
        engine = TrainingEngine(config, 'additive', data, 'precision-candidate')
        engine.amp = True
        engine.scaler = torch.amp.GradScaler('cpu', init_scale=2.0 ** 24)
        row = engine.train_batch()
        expected_rng = torch.random.get_rng_state()
        self.assertGreater(row['amp_retries'], 0)
        self.assertEqual(engine.progress['global_step'], 1)
        self.assertEqual(engine.progress['applied_steps'], 1)
        self.assertEqual(engine.progress['sample_offset'], 2)
        expected = copy.deepcopy(engine.eu.state_dict())
        direct = TrainingEngine(config, 'additive', data, 'precision-candidate')
        direct.amp = True
        direct.scaler = torch.amp.GradScaler('cpu', init_scale=row['amp_scale_after'])
        direct_row = direct.train_batch()
        for key in ('sample_indices', 'timesteps', 'noise_sha256', 'noise_mse'):
            self.assertEqual(row[key], direct_row[key])
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        for key, value in direct.eu.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)

    def test_real_amp_overflow_retries_same_rng_and_matches_safe_scale(self):
        torch.set_num_threads(2)
        model = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(4, 1))
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cpu', init_scale=2.0 ** 24)
        masks = []

        def loss():
            with torch.autocast('cpu', dtype=torch.float16):
                masked = model[0](torch.ones(4, 4))
                masks.append(masked.detach().clone())
                return (model[1](masked).float() - 2).square().mean()

        torch.manual_seed(55)
        _, _, result = precision_step(loss, model.parameters(), optimizer, scaler, 'cpu', 1.0, 16)
        expected_rng = torch.random.get_rng_state()
        trained = copy.deepcopy(model.state_dict())
        self.assertGreater(result['amp_retries'], 0)
        self.assertTrue(result['applied'])
        self.assertTrue(all(torch.equal(masks[0], mask) for mask in masks))
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        safe = torch.amp.GradScaler('cpu', init_scale=result['amp_scale_after'])
        torch.manual_seed(55)
        precision_step(loss, model.parameters(), optimizer, safe, 'cpu', 1.0)
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, trained[key]), key)

    def test_fp32_nonfinite_and_exhausted_retry_do_not_update_parameters(self):
        for enabled in (False, True):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            scaler = torch.amp.GradScaler('cpu', enabled=enabled, init_scale=2.0 ** 24)
            parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
            with self.assertRaisesRegex(FloatingPointError, 'Nonfinite full-training gradient'):
                precision_step(lambda: parameter.square().mean(), [parameter], optimizer, scaler, 'cpu', 1, 2)
            self.assertEqual(parameter.item(), 1)
            self.assertIsNone(parameter.grad)


if __name__ == '__main__':
    unittest.main()
