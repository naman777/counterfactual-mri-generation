"""CPU-only B2 integration and budget checks. Synthetic models stay in tmp/."""
import argparse
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.v4_heldout import train_judge
from scripts.v4_heldout.evaluate_heldout import model_entries, wilson
from scripts.v4_heldout.launch import active_apps
from scripts.v4_heldout.budget import CPU_RATE, GPU_RATE, TIMEOUT
from scripts.v4_heldout.orientation_probe import intervals
from scripts.v4_convergence.core import digest

ROOT = Path(__file__).resolve().parents[2]


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = ROOT / 'tmp' / f'b2_cpu_preflight_{time.time_ns()}'
        cls.tmp.mkdir(parents=True)
        (cls.tmp / 'README.md').write_text('SYNTHETIC TEST FIXTURES ONLY. Checkpoints/metrics here are software integration tests, not MRI-trained judges or research results. No file here may be used for paper/report performance.\n')
        rng = np.random.default_rng(17)
        cls.synthetic = {}
        for s, n in [('train', 48), ('val', 16)]:
            cls.synthetic[f'{s}_x'] = rng.random((n, 128, 128), dtype=np.float32)
            cls.synthetic[f'{s}_mfg'] = np.arange(n) % 2
            cls.synthetic[f'{s}_fs'] = (np.arange(n) // 2) % 2
            cls.synthetic[f'{s}_patient'] = np.array([f'SYNTHETIC_{s}_{i}' for i in range(n)])
            cls.synthetic[f'{s}_sop'] = np.array([f'SYNTHETIC_SOP_{s}_{i}' for i in range(n)])

    def test_real_new_pack_exclusion(self):
        path = ROOT / 'reports/v4_heldout_judge/preprocess_preflight_20261016_01/heldout_train_val_float32.npz'
        data, receipt = train_judge.load_pack(path)
        self.assertEqual(len(set(data['train_patient'])), 48)
        self.assertEqual(len(set(data['val_patient'])), 16)
        self.assertTrue(set(data['train_patient']).isdisjoint(set(data['excluded_patient_ids'])))
        self.assertTrue(set(data['val_patient']).isdisjoint(set(data['excluded_patient_ids'])))
        self.assertFalse(any(k.startswith('test_') for k in data))
        self.assertTrue(receipt['complete'])

    def test_prepared_old_sets_and_hashes(self):
        base = ROOT / 'reports/v4_heldout_judge/b2_20261016_01'
        for row in json.loads((base / 'input_manifest.json').read_text()):
            self.assertEqual(digest(base / row['path']), row['sha256'])
        sets = json.loads((base / 'inputs/old_sets.json').read_text())
        self.assertEqual(len(sets), 13)
        with np.load(base / 'inputs/old_counterfactual_sets.npz', allow_pickle=False) as pack:
            for item in sets:
                self.assertEqual(pack[item['key']].shape, (len(item['test_indices']), 128, 128))
                self.assertEqual(len(set(item['test_indices'])), len(item['test_indices']))

    def test_full_cpu_training_and_state_reload(self):
        with patch.object(train_judge, 'load_pack', return_value=(self.synthetic, {'pack_sha256': 'SYNTHETIC_TEST_FIXTURE_NOT_RESEARCH'})):
            result = train_judge.train('SYNTHETIC', self.tmp / 'complete', 0, epochs=2, device='cpu')
        self.assertTrue(result['complete'])
        self.assertEqual(result['epochs_completed'], 2)
        self.assertFalse(result['test_evaluated'])
        from scripts.v2_quickwins.judge_model import load_judge
        model = load_judge(self.tmp / 'complete/judge_heldout_resnet18_seed0.pt', 'cpu')
        with torch.inference_mode():
            out = model(torch.from_numpy(self.synthetic['val_x'][:2]).unsqueeze(1))
        self.assertEqual(out['mfg'].shape, (2, 2))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))

    def test_deadline_saves_partial_without_claiming_completion(self):
        with patch.object(train_judge, 'load_pack', return_value=(self.synthetic, {'pack_sha256': 'SYNTHETIC_TEST_FIXTURE_NOT_RESEARCH'})):
            result = train_judge.train('SYNTHETIC', self.tmp / 'partial', 1, epochs=40, device='cpu', time_limit_s=0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['epochs_completed'], 1)
        self.assertTrue((self.tmp / 'partial/judge_heldout_resnet18_seed1.pt').is_file())
        self.assertFalse(result['test_evaluated'])

    def test_test_evaluation_rejects_missing_six_model_freeze(self):
        folder = self.tmp / 'invalid_freeze'
        folder.mkdir()
        (folder / 'frozen_checkpoints.json').write_text('[]')
        with self.assertRaises(ValueError):
            list(model_entries(folder))

    def test_degenerate_bootstrap_has_non_degenerate_wilson_context(self):
        self.assertLess(wilson(np.ones(100))['wilson_slice_lo'], 1)
        self.assertGreater(wilson(np.ones(100))['wilson_slice_lo'], .96)
        self.assertGreater(wilson(np.zeros(100))['wilson_slice_hi'], 0)
        self.assertLess(wilson(np.zeros(100))['wilson_slice_hi'], .04)

    def test_cluster_intervals_account_for_repeated_patients(self):
        values = np.repeat([0.] * 5 + [1.] * 5, 10)
        patient = np.repeat(np.arange(10), 10)
        result = intervals(values, patient)
        self.assertGreater(result['patient_hi'] - result['patient_lo'], result['slice_hi'] - result['slice_lo'])

    def test_nine_full_worker_reservations_fit_proposed_allowance(self):
        worst = 9 * (TIMEOUT + 30) * (GPU_RATE + CPU_RATE) + .25 + .05
        self.assertLess(worst, 5.0)

    def test_active_app_guard_uses_actual_modal_list_schema(self):
        fixture = [{'app_id': 'ACTIVE', 'state': 'ephemeral (detached)'}, {'app_id': 'STOPPED', 'state': 'stopped'}]
        response = type('Response', (), {'stdout': json.dumps(fixture)})()
        with patch('scripts.v4_heldout.launch.subprocess.run', return_value=response):
            self.assertEqual([a['app_id'] for a in active_apps()], ['ACTIVE'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Pipeline)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'cpu_only': True, 'gpu_or_modal_jobs_launched': False, 'synthetic_outputs_folder': str(Pipeline.tmp)}
    with Path(args.report).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
