"""Append-only B2 downloads; no app creation. Run with --run-id <id>."""
import argparse
import shutil
from pathlib import Path
from uuid import uuid4

import modal

from scripts.v4_convergence.core import digest
from scripts.v4_heldout.prepare_run import ROOT, valid_id


def fetch(run_id):
    valid_id(run_id)
    volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
    prefix = f'v4_heldout/{run_id}/'
    entries = list(volume.iterdir('/' + prefix, recursive=True))
    names = {entry.path.lstrip('/') for entry in entries}
    root = ROOT / 'reports/v4_heldout_judge' / run_id / 'remote'
    count = 0
    for entry in entries:
        name = entry.path.lstrip('/')
        if entry.type.name != 'FILE' or '/inputs/' in name:
            continue
        relative = name.removeprefix(prefix)
        if relative == name or '..' in Path(relative).parts:
            raise ValueError('Unexpected remote path')
        parts = Path(relative).parts
        if parts[0] == 'training':
            family, seed = parts[1].rsplit('_seed', 1)
            if prefix + f'completed/train_{family}_{seed}.json' not in names:
                continue
        if parts[0] in ['test_accuracy', 'regrade_old', 'regrade_b3'] and prefix + f'completed/{parts[0]}.json' not in names:
            continue
        target = root / relative
        if target.exists():
            if target.stat().st_size != entry.size:
                raise ValueError('Existing immutable download differs')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.download-' + uuid4().hex)
        with partial.open('xb') as stream:
            for block in volume.read_file(entry.path):
                stream.write(block)
        if partial.stat().st_size != entry.size:
            raise ValueError('Remote output changed while downloading; partial retained')
        partial.rename(target)
        count += 1
    checkpoint_root = ROOT / 'checkpoints/judge/ax_t1_128_heldout' / run_id
    for source in (root / 'training').glob('*/*.pt'):
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        target = checkpoint_root / f'{source.parent.name}.pt'
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError('Exported checkpoint differs; refusing overwrite')
            continue
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    print('Fetched', count, 'completed files into', root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id', required=True)
    fetch(parser.parse_args().run_id)
