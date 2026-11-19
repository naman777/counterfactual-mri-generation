"""Full engine recovery, selection isolation, frame and statistical checks."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v7_geometry.orientation import apply_orientation, orientation_plan
from scripts.v8_generator_dev.evaluate_full import source_native
from scripts.v8_generator_dev.bundle_full import safe_archive_name
from scripts.v8_generator_dev.full_protocol import bootstrap_patients, plateau, read_config, select_setting
from scripts.v8_generator_dev.full_train import TrainingEngine
from scripts.v8_generator_dev.io import ROOT
from scripts.v8_generator_dev.test_pipeline import tiny_config


def fixture_config():
    config = read_config('configs/generator_full_v1.json')
    config.update(tiny_config())
    config['full_training']['batch_size'] = 2
    config['training']['mixed_precision'] = False
    return config


class SyntheticTraining:
    split = 'train'
    def __init__(self):
        self.frame = pd.DataFrame([dict(patient_id=f'p{i}', manufacturer_label=i % 2,
                                         field_strength_label=(i // 2) % 2) for i in range(5)])
        self.images = torch.randn((5, 1, 16, 16), generator=torch.Generator().manual_seed(5))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return dict(image=self.images[index], patient_id=row.patient_id,
                    manufacturer_label=int(row.manufacturer_label),
                    field_strength_label=int(row.field_strength_label), echo_time_norm=np.float32(0.3))


def equal_nested(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a.cpu(), b.cpu()))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            equal_nested(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal_nested(test, x, y)
    else:
        test.assertEqual(a, b)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_resume_exact_across_epoch_boundary_both_variants(self):
        config, data = fixture_config(), SyntheticTraining()
        for variant in config['variants']:
            uninterrupted = TrainingEngine(config, variant, data, 'frozen-protocol')
            direct_rows = [uninterrupted.train_batch() for _ in range(7)]
            expected = {key: copy.deepcopy(module.state_dict()) for key, module in
                        [('raw_c', uninterrupted.c), ('raw_u', uninterrupted.u),
                         ('ema_c', uninterrupted.ec), ('ema_u', uninterrupted.eu)]}
            expected_optimizer = copy.deepcopy(uninterrupted.optimizer.state_dict())
            first = TrainingEngine(config, variant, data, 'frozen-protocol')
            rows = [first.train_batch() for _ in range(2)]
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                checkpoint = Path(folder) / 'resume.pt'
                first.save(checkpoint, engineering=True)
                resumed = TrainingEngine(config, variant, data, 'frozen-protocol', resume=checkpoint)
                rows += [resumed.train_batch() for _ in range(5)]
                for key, module in [('raw_c', resumed.c), ('raw_u', resumed.u), ('ema_c', resumed.ec), ('ema_u', resumed.eu)]:
                    equal_nested(self, expected[key], module.state_dict())
                equal_nested(self, expected_optimizer, resumed.optimizer.state_dict())
                self.assertEqual(uninterrupted.progress, resumed.progress)
                for direct, continued in zip(direct_rows, rows):
                    for key in ('sample_indices', 'timesteps', 'noise_sha256', 'noise_mse', 'applied'):
                        self.assertEqual(direct[key], continued[key], key)

    def test_resume_rejects_protocol_variant_data_and_batch_changes(self):
        config, data = fixture_config(), SyntheticTraining()
        engine = TrainingEngine(config, 'additive', data, 'p')
        engine.train_batch()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'resume.pt'
            engine.save(path, engineering=True)
            for variant, protocol in [('scale_shift', 'p'), ('additive', 'changed')]:
                with self.assertRaisesRegex(ValueError, 'Resume incompatibility'):
                    TrainingEngine(config, variant, data, protocol, resume=path)
            with self.assertRaisesRegex(ValueError, 'effective_batch_size'):
                TrainingEngine(config, 'additive', data, 'p', resume=path, batch_size_override=1)
            changed = SyntheticTraining()
            changed.frame.loc[0, 'patient_id'] = 'different'
            with self.assertRaisesRegex(ValueError, 'data_identity'):
                TrainingEngine(config, 'additive', changed, 'p', resume=path)

    def test_validation_does_not_consume_dropout_rng_or_training_cursor(self):
        config, data = fixture_config(), SyntheticTraining()
        engine = TrainingEngine(config, 'additive', data, 'p')
        validation = SyntheticTraining()
        validation.split = 'val'
        state = torch.random.get_rng_state().clone()
        cursor = dict(engine.progress)
        _, first = engine.validate(validation)
        _, second = engine.validate(validation)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(cursor, engine.progress)

    def test_training_rejects_validation_dataset(self):
        data = SyntheticTraining()
        data.split = 'val'
        with self.assertRaisesRegex(ValueError, 'training split'):
            TrainingEngine(fixture_config(), 'additive', data, 'p')


class SelectionTests(unittest.TestCase):
    def test_archive_paths_cannot_escape_project(self):
        for value in ('../outside', '/absolute', 'C:/outside', 'data\\outside'):
            with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                safe_archive_name(value)
        self.assertEqual(str(safe_archive_name('data/processed/train.npy')), 'data/processed/train.npy')

    def rows(self):
        config = fixture_config()
        rows = []
        for t, delta in [(100, 0.3), (200, 0.5)]:
            for direction in ('0_to_1', '1_to_0'):
                for patient in range(2):
                    for judge in range(3):
                        rows.append(dict(axis='field_strength', generation_seed=config['evaluation']['selection_generation_seed'],
                                         variant='target', timestep=t, guidance=3.0, direction=direction,
                                         patient_id=f'{direction}-{patient}', judge=judge,
                                         condition_support_status='interpolation_only',
                                         ssim=0.8, other_axis_stable=1.0, paired_hit_delta=delta))
        return pd.DataFrame(rows), config

    def test_selection_excludes_outside_range_and_confirmation_seeds(self):
        rows, config = self.rows()
        rows.loc[rows.timestep == 200, 'condition_support_status'] = 'outside_target_echo_range'
        contaminant = rows.copy()
        contaminant['generation_seed'] = config['evaluation']['confirmation_generation_seeds'][0]
        contaminant['paired_hit_delta'] = 1.0
        chosen = select_setting(pd.concat([rows, contaminant]), 'field_strength', config)
        self.assertEqual(chosen['selected'], dict(timestep=100, guidance=3.0))

    def test_no_feasible_setting_is_explicit(self):
        rows, config = self.rows()
        for column, value in [('ssim', 0.2), ('paired_hit_delta', 0.0), ('other_axis_stable', 0.0)]:
            bad = rows.copy()
            bad[column] = value
            self.assertIsNone(select_setting(bad, 'field_strength', config)['selected'])
        bad = rows[rows.direction == '0_to_1']
        self.assertIsNone(select_setting(bad, 'field_strength', config)['selected'])

    def test_positive_delta_and_deterministic_tie_break(self):
        rows, config = self.rows()
        rows['paired_hit_delta'] = 0.4
        self.assertEqual(select_setting(rows, 'field_strength', config)['selected'], dict(timestep=100, guidance=3.0))

    def test_patient_bootstrap_averages_seeds_and_judges_before_resampling(self):
        frame = pd.DataFrame(dict(patient_id=['a', 'a', 'a', 'b'], hit=[0, 0, 0, 1]))
        result = bootstrap_patients(frame, 'hit', 5, 1000)
        self.assertEqual(result['patients'], 2)
        self.assertEqual(result['mean'], 0.5)
        self.assertEqual(result, bootstrap_patients(frame, 'hit', 5, 1000))

    def test_plateau_requires_minimum_epoch_and_patience(self):
        config = fixture_config()
        config['full_training'].update(minimum_stop_epoch=20, plateau_checks=2)
        self.assertFalse(plateau([dict(epoch=5, patient_macro_noise_mse=1)] * 3, config))
        self.assertTrue(plateau([dict(epoch=epoch, patient_macro_noise_mse=value)
                                 for epoch, value in [(10, 1.0), (15, 1.0), (20, 1.0)]], config))
        self.assertFalse(plateau([dict(epoch=epoch, patient_macro_noise_mse=value)
                                  for epoch, value in [(10, 1.0), (15, 0.9), (20, 0.8)]], config))

    def test_judges_receive_exact_inverse_source_geometry(self):
        source = np.arange(128 * 128).reshape(128, 128).astype(np.float32)
        for iop in ([-1, 0, 0, 0, -1, 0], [0, 1, 0, 1, 0, 0]):
            plan = orientation_plan(iop)
            canonical = apply_orientation(source, plan)
            np.testing.assert_array_equal(source_native(canonical[None], plan)[0], source)


if __name__ == '__main__':
    unittest.main()
