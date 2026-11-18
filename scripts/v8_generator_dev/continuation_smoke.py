"""Original-runtime strict resume/inference-update smoke; never advances efficacy parents."""
import argparse
import gc
import json

import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_train import CachedCanonical, TrainingEngine, deterministic_runtime
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def smoke(prepared, run_id):
    deterministic_runtime()
    prepared = project_path(prepared)
    protocol_hash = digest(prepared / 'protocol.json')
    config = json.loads((prepared / 'protocol.json').read_text())['config']
    pair = json.loads((prepared / 'prior_pair.json').read_text())
    train = CachedCanonical(CanonicalDataset(config['view'], 'train'))
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    rows = {}
    for variant, relative in pair['checkpoints'].items():
        path = project_path(relative)
        before_hash = digest(path)
        engine = TrainingEngine(config, variant, train, protocol_hash, 'cuda', path)
        if engine.progress != pair['progress'][variant]:
            raise ValueError('Resume cursor mismatch')
        rows[variant] = engine.train_batch()
        if digest(path) != before_hash or engine.progress['applied_steps'] != pair['progress'][variant]['applied_steps']+1:
            raise ValueError('Smoke changed parent bytes or skipped applied update')
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    for key in ('sample_indices', 'timesteps', 'noise_sha256', 'global_step'):
        if rows['additive'][key] != rows['scale_shift'][key]:
            raise ValueError('Runtime smoke has unmatched draws')
    json_write(output / 'steps.json', rows)
    json_write(output / 'complete.json', dict(complete=True, original_torch_runtime=str(torch.__version__),
        matched_draws=True, actual_strict_cuda_resume=True, parent_hashes_unchanged=True,
        smoke_updates_discarded=True, full_training_completed=False, anatomy_validated=False))
    (output / 'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.continuation_smoke`.\n'
        'steps.json retains isolated full-batch draws/losses/precision updates for both variants. '
        'complete.json records original-runtime strict resume; the smoke updates are discarded '
        'and the subsequent efficacy run starts from the unchanged matched parent files.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    smoke(args.prepared, args.run_id)
