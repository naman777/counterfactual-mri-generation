"""Freeze B2 cloud inputs locally. No upload, app creation or GPU work.

python -m scripts.v4_heldout.prepare_run --run-id b2_20261016_01
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v4_convergence.core import digest, json_write, load_pack

ROOT = Path(__file__).resolve().parents[2]


def valid_id(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in value):
        raise ValueError('Invalid run ID')
    return value


def prepare(run_id):
    valid_id(run_id)
    out = ROOT / 'reports/v4_heldout_judge' / run_id
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    prepared = ROOT / 'reports/v4_heldout_judge/preprocess_preflight_20261016_01'
    receipt = json.loads((prepared / 'summary.json').read_text())
    assert receipt['complete'] and digest(prepared / 'heldout_train_val_float32.npz') == receipt['pack_sha256']
    shutil.copyfile(prepared / 'heldout_train_val_float32.npz', inputs / 'heldout_train_val_float32.npz')
    shutil.copyfile(prepared / 'summary.json', inputs / 'summary.json')
    original_pack = ROOT / 'reports/v4_diffusion/b1_20261016_01/inputs/pack.npz'
    data = load_pack(original_pack, ['test'])
    norm = lambda p: str(p).replace('\\', '/')
    lookup = {norm(p): i for i, p in enumerate(data['test_path'])}
    sets, arrays = [], {}

    def add(name, axis, paths, images, provenance):
        indices = np.array([lookup[norm(p)] for p in paths], dtype=np.int64)
        assert len(indices) == len(images) and len(set(indices)) == len(indices)
        images = np.asarray(images, dtype=np.float32)
        assert images.shape[1:] == (128, 128) and np.isfinite(images).all()
        assert images.min() >= 0 and images.max() <= 1
        key = f'set{len(sets)}'
        arrays[key] = images
        sets.append({'name': name, 'axis': axis, 'key': key, 'test_indices': indices.tolist(), 'provenance': provenance})

    for name, axis, csvpath, folder in [
        ('old_fs_t100_w3', 'field_strength', 'reports/phase12/field_strength/field_strength_results.csv', 'reports/phase12/field_strength/counterfactuals'),
        ('old_fs_t100_w5', 'field_strength', 'reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv', 'reports/v2_quickwins/fs_t100_w5/field_strength/counterfactuals'),
        ('old_mfg_t300_w5', 'manufacturer', 'reports/phase12/manufacturer/manufacturer_results.csv', 'reports/phase12/manufacturer/counterfactuals'),
    ]:
        df = pd.read_csv(ROOT / csvpath)
        files = [ROOT / folder / f'{e}_counterfactual.npy' for e in df.example_id]
        images = np.stack([np.clip(np.load(p, allow_pickle=False).astype(np.float32), 0, 1) for p in files])
        add(name, axis, df.image_path, images, {'metadata': csvpath, 'metadata_sha256': digest(ROOT / csvpath), 'image_sha256': {p.name: digest(p) for p in files}})
    for axis in ['field_strength', 'manufacturer']:
        for baseline in ['identity', 'contrast_shift', 'gamma', 'gaussian_noise', 'histogram_matching']:
            p = ROOT / f'reports/v2_quickwins/baselines_regen/{axis}__{baseline}.npz'
            with np.load(p, allow_pickle=False) as z:
                add(f'old_{axis}_{baseline}', axis, z['image_path'], z['images'], {'file': str(p.relative_to(ROOT)), 'sha256': digest(p)})
    with (inputs / 'old_counterfactual_sets.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    json_write(inputs / 'old_sets.json', sets)
    protocol = {
        'run_id': run_id, 'status': 'prepared_no_cloud_launch', 'b1_run_id': 'b1_20261016_01',
        'b1_remote_root': '/data/v4_convergence/b1_20261016_01',
        'original_pack_sha256': digest(original_pack), 'new_pack_sha256': receipt['pack_sha256'],
        'seeds': [0, 1, 2], 'epochs': 40, 'families': ['orientation_robust', 'legacy_comparison'],
        'primary_augmentation': 'original robust recipe plus independently sampled vertical flip p=0.5',
        'selection': 'earliest best mean clean/noise03 validation slice balanced accuracy across the two heads; new 16-patient validation only',
        'test': 'all 934 original test slices; clean, noise03, horizontal/vertical flip and 180 rotation; no label changes',
        'validity_flags': {'new_val_clean_head_bacc_at_least': 0.8, 'max_new_val_orientation_bacc_drop': 0.05, 'purpose': 'descriptive flags only, never exclude judges or tune checkpoints using original test outcomes'},
        'regrade': 'all saved original diffusion/baseline sets and every completed B1/B3 test job; retain all judges, report validity flags',
        'bootstrap': {'n': 10000, 'seed': 42, 'methods': ['slice', 'patient-cluster'], 'boundary_note': 'degenerate bootstrap intervals at all-zero/all-one are supplemented by Wilson slice intervals; not certainty'},
        'execution': {'gpu': 'A10G', 'max_containers': 1, 'one_app_at_a_time': True, 'worker_timeout_seconds': 900, 'startup_allowance_seconds': 30, 'safety_margin': 1.2, 'suggested_total_allowance_usd': 5.0, 'controller_allocation_usd': 4.75, 'setup_reserve_usd': .25, 'pricing_source': 'https://modal.com/pricing', 'gpu_per_second': .000306, 'cpu_core_per_second': .0000131, 'memory_gib_per_second': .00000222},
        'permissions': {'data_download_and_preprocessing_approved': True, 'additional_cloud_allowance_approved': False},
    }
    json_write(out / 'protocol.json', protocol)
    json_write(out / 'input_manifest.json', [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in inputs.iterdir() if p.is_file()])
    (out / 'README.md').write_text('''# B2 held-out judge cloud run preparation

Generated locally by `python -m scripts.v4_heldout.prepare_run`. No cloud launch or upload occurs here. `protocol.json` freezes patient separation, six training jobs (three orientation-robust primary seeds plus three legacy-augmentation comparisons), checkpoint selection, post-freeze test/re-grade plan, descriptive judge-validity flags and budget assumptions. `input_manifest.json` hashes every input file.

`inputs/heldout_train_val_float32.npz` and `inputs/summary.json` copy the verified new-patient training pack and receipt. `inputs/old_counterfactual_sets.npz` stores the 13 existing diffusion/baseline image sets, while `inputs/old_sets.json` maps them to original test indices and immutable source files/hashes. No synthetic ablation or copied-paper numbers are used. The original B1 cloud pack is read by explicit namespace and hash; it is not overwritten.

Launch only through scripts/v4_heldout/launch.py, after explicit extra compute approval. It checks that B1's app has ended before creating the new app. The remote controller uses one A10 GPU sequentially, reserves each complete worker timeout with a safety margin, and preserves completed/partial outputs on budget exhaustion or failure. This is a conservative scheduling guard, not a hard account-wide billing limit. Current price assumptions come from https://modal.com/pricing. The proposed allowance is US$5, with US$4.75 for the controller and US$0.25 setup reserve. No approval is implied by this file.

Original test evaluation starts only after all six checkpoints are frozen. Re-grade outputs never change source labels/operating points. If B1/B3 stopped without its final marker, the available partial outputs are explicitly marked incomplete. Primary clean discrimination and orientation sensitivity must both be assessed before using the new judges as conclusive evidence.
''', encoding='utf-8')
    print(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    prepare(parser.parse_args().run_id)
