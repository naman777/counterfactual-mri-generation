"""Development isolation, sampling mathematics and architecture correctness checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v7_geometry.orientation import apply_orientation, orientation_plan
from scripts.v8_generator_dev.data import CanonicalDataset, PatientCellSampler, sampling_probabilities
from scripts.v8_generator_dev.io import digest
from scripts.v8_generator_dev.model import (
    ScaleShiftResidualBlock, build_models, clone_ema, update_ema,
)
from scripts.v8_generator_dev.prepare import validate_config
from scripts.v8_generator_dev.train_smoke import sample_wiring, validate


def tiny_config():
    return {
        'condition': dict(num_manufacturers=2, num_field_strengths=2,
                          manufacturer_embedding_dim=4, field_strength_embedding_dim=4,
                          echo_embedding_dim=4, condition_dim=8, dropout_prob=0.1),
        'model': dict(in_channels=1, out_channels=1, base_channels=8,
                      channel_multipliers=[1, 2, 4], time_embedding_dim=8,
                      condition_dim=8, dropout=0.1),
        'diffusion': dict(num_train_timesteps=1000, beta_start=0.0001,
                          beta_end=0.02, beta_schedule='linear', prediction_type='epsilon'),
    }


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.view = self.root / 'view'
        (self.view / 'splits').mkdir(parents=True)
        (self.view / 'view.json').write_text(json.dumps({'format': 'canonical_axial_view_v1'}))
        (self.root / 'data/processed/ax_t1_128/splits').mkdir(parents=True)
        self.source = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / (128 * 128)
        for split, pid in [('train', 'training'), ('val', 'validation')]:
            path = self.root / f'{split}.npy'
            np.save(path, self.source)
            iop = [-1, 0, 0, 0, -1, 0]
            row = dict(patient_id=pid, sop_uid=f'{pid}-sop', split=split,
                       manufacturer_label=0, field_strength_label=1, echo_time_norm=0.3,
                       source_image_path=path.name, source_array_sha256=digest(path),
                       orientation_plan=json.dumps(orientation_plan(iop)), source_iop=json.dumps(iop))
            pd.DataFrame([row]).to_csv(self.view / 'splits' / f'{split}.csv', index=False)
            original = dict(row, image_path=path.name)
            pd.DataFrame([original]).to_csv(self.root / 'data/processed/ax_t1_128/splits' / f'{split}.csv', index=False)
        # A sentinel test index must never be parsed by the loader.
        (self.root / 'data/processed/ax_t1_128/splits/test.csv').write_text('DO NOT READ TEST DATA')

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, field, value):
        path = self.view / 'splits/train.csv'
        frame = pd.read_csv(path)
        frame[field] = value
        frame.to_csv(path, index=False)

    def test_header_orientation_and_normalisation(self):
        data = CanonicalDataset(self.view, 'train', self.root)
        expected = apply_orientation(self.source, orientation_plan([-1, 0, 0, 0, -1, 0]))
        np.testing.assert_array_equal(data.image_01(0), expected)
        np.testing.assert_array_equal(data[0]['image'][0].numpy(), expected * 2 - 1)
        self.assertEqual(data[0]['echo_time_norm'], np.float32(0.3))

    def test_test_split_rejected_before_file_read(self):
        with self.assertRaisesRegex(ValueError, 'train/val only'):
            CanonicalDataset(self.view, 'test', self.root)

    def test_forged_split_membership_rejected(self):
        self.mutate('sop_uid', 'validation-sop')
        with self.assertRaisesRegex(ValueError, 'original declared split'):
            CanonicalDataset(self.view, 'train', self.root)

    def test_label_and_echo_changes_rejected(self):
        self.mutate('echo_time_norm', 0.7)
        with self.assertRaisesRegex(ValueError, 'EchoTime changed'):
            CanonicalDataset(self.view, 'train', self.root)
        self.mutate('echo_time_norm', 0.3)
        self.mutate('manufacturer_label', 1)
        with self.assertRaisesRegex(ValueError, 'manufacturer_label'):
            CanonicalDataset(self.view, 'train', self.root)

    def test_transform_not_chosen_from_vendor(self):
        plan = orientation_plan([-1, 0, 0, 0, -1, 0])
        plan['flip_rows'] = False
        self.mutate('orientation_plan', json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'DICOM IOP'):
            CanonicalDataset(self.view, 'train', self.root)

    def test_source_mutation_rejected(self):
        data = CanonicalDataset(self.view, 'train', self.root)
        np.save(self.root / 'train.npy', self.source + 0.001)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            data[0]


class SamplingTests(unittest.TestCase):
    @staticmethod
    def frame():
        # Cell 0 has two patients with unequal slice counts; cell 1 has one.
        return pd.DataFrame([
            dict(patient_id=pid, manufacturer_label=m, field_strength_label=0)
            for pid, m, slices in [('a', 0, 1), ('b', 0, 3), ('c', 1, 2)]
            for _ in range(slices)
        ])

    def test_supported_cells_balance_patients_not_slices(self):
        frame = self.frame()
        frame['p'] = sampling_probabilities(frame, 1)
        masses = frame.groupby('patient_id').p.sum().to_dict()
        self.assertAlmostEqual(masses['a'], 0.25)
        self.assertAlmostEqual(masses['b'], 0.25)
        self.assertAlmostEqual(masses['c'], 0.5)

    def test_sparse_cell_retained_without_forced_oversampling(self):
        frame = self.frame()
        frame['p'] = sampling_probabilities(frame, 2)
        for value in frame.groupby('patient_id').p.sum():
            self.assertAlmostEqual(value, 1 / 3)
        self.assertTrue((frame.p > 0).all())

    def test_sampler_seed_epoch_and_validation_exclusion(self):
        class Data:
            split = 'train'
            frame = SamplingTests.frame()
            def __len__(self):
                return len(self.frame)
        data = Data()
        sampler = PatientCellSampler(data, 123, 100, 1)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        data.split = 'val'
        with self.assertRaisesRegex(ValueError, 'restricted to training'):
            PatientCellSampler(data, 123)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initial_weights_and_new_projection(self):
        torch.manual_seed(73)
        c0, m0, _ = build_models(tiny_config(), 'additive')
        torch.manual_seed(73)
        c1, m1, _ = build_models(tiny_config(), 'scale_shift')
        for key, value in c0.state_dict().items():
            self.assertTrue(torch.equal(value, c1.state_dict()[key]))
        for key, value in m0.state_dict().items():
            other = m1.state_dict()[key]
            if value.shape == other.shape:
                self.assertTrue(torch.equal(value, other), key)
            else:
                self.assertTrue(torch.equal(value, other[other.shape[0] // 2:]), key)
                self.assertEqual(torch.count_nonzero(other[:other.shape[0] // 2]).item(), 0)

    def test_scale_shift_formula_and_condition_gradients(self):
        block = ScaleShiftResidualBlock(8, 8, 12, dropout=0).eval()
        x = torch.randn(2, 8, 8, 8)
        condition = torch.randn(2, 12, requires_grad=True)
        h = block.conv1(block.activation(block.norm1(x)))
        scale, shift = block.time_condition_proj(block.activation(condition)).chunk(2, dim=1)
        expected = block.conv2(block.activation(block.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None])) + x
        output = block(x, condition)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        self.assertGreater(condition.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(condition.grad).all())

    def test_both_axes_condition_outputs_and_update(self):
        c, model, scheduler = build_models(tiny_config(), 'scale_shift')
        c.eval()
        model.eval()
        x, t = torch.randn(1, 1, 16, 16), torch.tensor([100])
        m, f, e = torch.tensor([0]), torch.tensor([0]), torch.tensor([0.3])
        source = model(x, t, c(m, f, e))
        for a, b in [(1 - m, f), (m, 1 - f)]:
            self.assertFalse(torch.equal(source, model(x, t, c(a, b, e))))
        source.square().mean().backward()
        self.assertGreater(c.manufacturer_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(c.field_strength_embedding.weight.grad.abs().sum().item(), 0)
        for name, block in model.named_children():
            if isinstance(block, ScaleShiftResidualBlock):
                self.assertGreater(block.time_condition_proj.weight.grad.abs().sum().item(), 0, name)

    def test_ema_is_frozen_and_uses_expected_decay(self):
        raw = torch.nn.Linear(3, 2)
        ema = clone_ema(raw)
        before = ema.weight.detach().clone()
        with torch.no_grad():
            raw.weight.add_(2)
        update_ema(ema, raw, 0.75)
        torch.testing.assert_close(ema.weight, before + 0.5)
        self.assertFalse(ema.training)
        self.assertTrue(all(not p.requires_grad for p in ema.parameters()))

    def test_seeded_cfg_and_fixed_validation_rng(self):
        c, model, scheduler = build_models(tiny_config(), 'additive')
        images = torch.randn(1, 1, 16, 16)
        args = (c, model, scheduler, images, torch.tensor([0]), torch.tensor([1]), torch.tensor([0.3]), 93, 2, 3.0)
        state = torch.random.get_rng_state().clone()
        a, b = sample_wiring(*args), sample_wiring(*args)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(np.isfinite(a).all())


if __name__ == '__main__':
    unittest.main()
