"""Run B2 on the local GPU with the frozen b2_20261016_01 protocol and code.

Same trainer (train_judge.train), seeds, families, epochs, selection rule and
post-freeze evaluation (evaluate_heldout) as the Modal controller. Differences:
local RTX 3050 instead of A10G, no worker time limit, and B3 outputs are read
from the local fetch mirror of B1 instead of the Modal volume.

python -m scripts.v4_heldout.local_run --run-id b2_local_20261016_01 --stage train
python -m scripts.v4_heldout.local_run --run-id b2_local_20261016_01 --stage regrade_b3   # after fetching B1/B3
"""
import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.v4_convergence.core import digest, json_write
from scripts.v4_heldout.prepare_run import ROOT, valid_id

SOURCE_RUN = ROOT / 'reports/v4_heldout_judge/b2_20261016_01'
ORIGINAL_PACK = ROOT / 'reports/v4_diffusion/b1_20261016_01/inputs/pack.npz'
B1_LOCAL_MIRROR = ROOT / 'reports/v4_diffusion/b1_20261016_01/remote'


def setup(base):
    source_protocol = json.loads((SOURCE_RUN / 'protocol.json').read_text())
    if base.exists():
        protocol = json.loads((base / 'protocol.json').read_text())
        if protocol['source_protocol_sha256'] != digest(SOURCE_RUN / 'protocol.json'):
            raise ValueError('Source protocol changed since this local run was created')
        return protocol
    base.mkdir(parents=True)
    (base / 'inputs').mkdir()
    for name in ['old_sets.json', 'old_counterfactual_sets.npz']:
        shutil.copyfile(SOURCE_RUN / 'inputs' / name, base / 'inputs' / name)
    protocol = {**source_protocol,
                'run_id': base.name, 'status': 'local_gpu_run',
                'source_run': 'b2_20261016_01', 'source_protocol_sha256': digest(SOURCE_RUN / 'protocol.json'),
                'b1_remote_root': str(B1_LOCAL_MIRROR),
                'b1_root_note': 'local fetch mirror of the B1/B3 Modal outputs; same relative layout as /data/v4_convergence/b1_20261016_01',
                'execution': {'device': 'local', 'gpu': torch.cuda.get_device_name(0), 'worker_time_limit_seconds': None,
                              'cloud_spend_usd': 0.0, 'deviation_from_source': 'hardware only (A10G -> local GPU); no 900 s worker limit'},
                'permissions': {**source_protocol['permissions'], 'local_gpu_run_requested_by_user': True}}
    json_write(base / 'protocol.json', protocol)
    json_write(base / 'input_manifest.json', [
        {'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)}
        for p in [SOURCE_RUN / 'inputs/heldout_train_val_float32.npz', SOURCE_RUN / 'inputs/summary.json',
                  base / 'inputs/old_sets.json', base / 'inputs/old_counterfactual_sets.npz', ORIGINAL_PACK]])
    return protocol


def train_all(base, protocol):
    from scripts.v4_heldout.train_judge import train
    pack = SOURCE_RUN / 'inputs/heldout_train_val_float32.npz'
    if digest(pack) != protocol['new_pack_sha256']:
        raise ValueError('Held-out pack hash differs from protocol')
    for family in protocol['families']:
        for seed in protocol['seeds']:
            out = base / 'training' / f'{family}_seed{seed}'
            if (out / 'train.json').exists():
                print('skip (done)', out.name, flush=True)
                continue
            if out.exists():
                raise RuntimeError(f'{out} exists without train.json; inspect it before retrying')
            start = time.perf_counter()
            result = train(pack, out, seed, protocol['epochs'], 'cuda', vertical_flip=family == 'orientation_robust')
            json_write(base / 'timing' / f'{family}_seed{seed}.json', {'seconds': time.perf_counter() - start,
                       'complete': result['complete'], 'best_epoch': result['best_epoch']})
            print('trained', out.name, 'best epoch', result['best_epoch'], flush=True)
    frozen = base / 'frozen_checkpoints.json'
    if not frozen.exists():
        entries = []
        for family in protocol['families']:
            for seed in protocol['seeds']:
                folder = Path('training') / f'{family}_seed{seed}'
                path = folder / f'judge_heldout_resnet18_seed{seed}.pt'
                entries.append({'family': family, 'seed': seed, 'checkpoint': path.as_posix(),
                                'training_json': (folder / 'train.json').as_posix(), 'sha256': digest(base / path)})
        json_write(frozen, entries)


def evaluate(base, protocol, stages):
    from scripts.v4_heldout.evaluate_heldout import regrade, test_accuracy
    if digest(ORIGINAL_PACK) != protocol['original_pack_sha256']:
        raise ValueError('Original frozen evaluation pack changed')
    results = {}
    if 'test' in stages and not (base / 'test_accuracy/complete.json').exists():
        results['test'] = test_accuracy(base, ORIGINAL_PACK)
    if 'old' in stages and not (base / 'regrade_old/complete.json').exists():
        results['old_regrade'] = regrade(base, ORIGINAL_PACK, 'old')
    if 'b3' in stages:
        if (base / 'regrade_b3').exists():
            raise RuntimeError('regrade_b3 already exists; it is append-only')
        results['b3_regrade'] = regrade(base, ORIGINAL_PACK, 'b3')
    return results


def main(args):
    valid_id(args.run_id)
    base = ROOT / 'reports/v4_heldout_judge' / args.run_id
    protocol = setup(base)
    if args.stage == 'train':
        train_all(base, protocol)
        results = evaluate(base, protocol, ['test', 'old'])
    else:
        results = evaluate(base, protocol, ['b3'])
    json_write(base / f'status_{args.stage}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json',
               {'stage': args.stage, 'results': results, 'paper_integration_complete': False})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=['train', 'regrade_b3'], required=True)
    main(parser.parse_args())
