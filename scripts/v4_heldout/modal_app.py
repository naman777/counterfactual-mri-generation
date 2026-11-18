"""B2: one sequential A10 app; launch only via scripts.v4_heldout.launch.

No app is started by importing this file. The launcher enforces the one-app rule
BEFORE Modal creates this app. All data/checkpoint namespaces are run-specific.
"""
import json
import sys
import time
from pathlib import Path

import modal
from scripts.v4_heldout.budget import CPU_RATE, GPU_RATE, TIMEOUT

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root/project')
app = modal.App('capstone-v4-heldout-judges')
volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.5.1', 'torchvision==0.20.1', 'numpy==2.1.3', 'pandas==2.2.3',
    'scipy==1.14.1', 'scikit-image==0.24.0', 'pyyaml==6.0.2', 'pillow', 'matplotlib==3.9.2',
).env({'PYTHONPATH': '/root/project', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
if modal.is_local():
    image = image.add_local_dir(ROOT / 'src', '/root/project/src').add_local_dir(ROOT / 'scripts', '/root/project/scripts', ignore=['**/__pycache__/**'])

def remote_root(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in run_id):
        raise ValueError('Invalid run ID')
    return Path('/data/v4_heldout') / run_id


@app.function(image=image, gpu='A10G', cpu=2, memory=16384, timeout=TIMEOUT,
              max_containers=1, scaledown_window=2, retries=0, volumes={'/data': volume})
def worker(run_id: str, operation: str, family: str = '', seed: int = 0):
    sys.path.insert(0, '/root/project')
    import torch
    from scripts.v4_convergence.core import digest
    volume.reload()
    base = remote_root(run_id)
    protocol = json.loads((base / 'protocol.json').read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    if operation == 'train':
        from scripts.v4_heldout.train_judge import train
        if family not in protocol['families'] or seed not in protocol['seeds']:
            raise ValueError('Training outside frozen protocol')
        result = train(base / 'inputs/heldout_train_val_float32.npz', base / 'training' / f'{family}_seed{seed}', seed, protocol['epochs'], 'cuda', vertical_flip=family == 'orientation_robust', time_limit_s=TIMEOUT - 90)
        result = {'complete': result['complete'], 'family': family, 'seed': seed, 'best_epoch': result['best_epoch'], 'checkpoint_sha256': result['checkpoint_sha256']}
    else:
        from scripts.v4_heldout.evaluate_heldout import test_accuracy, regrade
        original_pack = Path(protocol['b1_remote_root']) / 'inputs/pack.npz'
        if digest(original_pack) != protocol['original_pack_sha256']:
            raise ValueError('Original frozen evaluation pack changed')
        if operation == 'test_accuracy':
            result = test_accuracy(base, original_pack)
        elif operation in ['regrade_old', 'regrade_b3']:
            result = regrade(base, original_pack, operation.removeprefix('regrade_'))
        else:
            raise ValueError(operation)
    result['worker_seconds'] = time.perf_counter() - start
    volume.commit()
    return result


@app.function(image=image, cpu=.25, memory=2048, timeout=4 * 3600, max_containers=1,
              scaledown_window=2, retries=0, volumes={'/data': volume})
def controller(run_id: str, budget_usd: float):
    sys.path.insert(0, '/root/project')
    from scripts.v4_convergence.core import digest, json_write
    volume.reload()
    base = remote_root(run_id)
    protocol = json.loads((base / 'protocol.json').read_text())
    if budget_usd <= 0:
        raise ValueError('A separately approved positive compute cap is required')
    ledger = base / 'ledger'
    ledger.mkdir(exist_ok=True)
    sessions = base / 'controller_sessions'
    sessions.mkdir(exist_ok=True)
    prior = sorted(ledger.glob('*.json'))
    spent = sum(json.loads(p.read_text())['conservative_usd'] for p in prior)
    spent += sum(json.loads(p.read_text())['conservative_usd'] for p in sessions.glob('*.json'))
    serial = len(prior)
    start = time.perf_counter()

    def call(operation, family='', seed=0):
        nonlocal spent, serial
        tag = f'{operation}_{family}_{seed}' if operation == 'train' else operation
        done = base / 'completed' / f'{tag}.json'
        if done.exists():
            return json.loads(done.read_text())['result']
        reserve = (TIMEOUT + 30) * GPU_RATE
        remaining = budget_usd - spent - (time.perf_counter() - start) * CPU_RATE
        if remaining < reserve + .05:
            raise RuntimeError(f'Allowance insufficient before {tag}; preserved outputs, remaining ${remaining:.2f}, worker reservation ${reserve:.2f}')
        json_write(base / 'reservations' / f'{serial:05d}.json', {'operation': tag, 'worst_case_usd': reserve})
        volume.commit()
        try:
            result = worker.remote(run_id, operation, family, seed)
        except Exception:
            volume.reload()
            spent += reserve
            json_write(ledger / f'{serial:05d}.json', {'operation': tag, 'failed': True, 'conservative_usd': reserve})
            serial += 1
            volume.commit()
            raise
        volume.reload()
        charge = (result['worker_seconds'] + 30) * GPU_RATE
        spent += charge
        json_write(ledger / f'{serial:05d}.json', {'operation': tag, 'worker_seconds': result['worker_seconds'], 'conservative_usd': charge})
        serial += 1
        if operation == 'train' and not result['complete']:
            json_write(base / 'partial' / f'{tag}.json', result)
            volume.commit()
            raise RuntimeError(f'{tag} saved a partial checkpoint before its timeout; no test data evaluated')
        json_write(done, {'result': result})
        volume.commit()
        return result

    try:
        for family in protocol['families']:
            for seed in protocol['seeds']:
                call('train', family, seed)
        frozen = base / 'frozen_checkpoints.json'
        if not frozen.exists():
            entries = []
            for family in protocol['families']:
                for seed in protocol['seeds']:
                    folder = Path('training') / f'{family}_seed{seed}'
                    path = folder / f'judge_heldout_resnet18_seed{seed}.pt'
                    entries.append({'family': family, 'seed': seed, 'checkpoint': path.as_posix(), 'training_json': (folder / 'train.json').as_posix(), 'sha256': digest(base / path)})
            json_write(frozen, entries)
            volume.commit()
        test = call('test_accuracy')
        old = call('regrade_old')
        b3 = call('regrade_b3')
        status = {'status': 'b2_experiments_complete' if b3['complete'] else 'b2_judges_ready_b3_incomplete', 'test': test, 'old_regrade': old, 'b3_regrade': b3, 'conservative_worker_usd': spent, 'budget_usd': budget_usd, 'paper_integration_complete': False}
        json_write(base / 'research_status.json', status)
        volume.commit()
        return status
    except Exception as exc:
        status = {'status': 'interrupted_outputs_preserved', 'error': repr(exc), 'conservative_worker_usd': spent, 'budget_usd': budget_usd}
        json_write(base / f'interrupted_{serial:05d}_{int(time.time())}.json', status)
        volume.commit()
        return status
    finally:
        json_write(sessions / f'session_{time.time_ns()}.json', {'elapsed_seconds': time.perf_counter() - start, 'conservative_usd': (time.perf_counter() - start) * CPU_RATE})
        volume.commit()


def upload(run_id):
    from scripts.v4_convergence.core import digest
    local = ROOT / 'reports/v4_heldout_judge' / run_id
    prefix = remote_root(run_id).relative_to('/data').as_posix()
    manifest = json.loads((local / 'input_manifest.json').read_text())
    for row in manifest:
        if digest(local / row['path']) != row['sha256']:
            raise ValueError('Prepared input hash differs')
    existing = []
    try:
        existing = list(volume.iterdir('/' + prefix))
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    if existing:
        remote_protocol = json.loads(b''.join(volume.read_file('/' + prefix + '/protocol.json')).decode())
        if remote_protocol != json.loads((local / 'protocol.json').read_text()):
            raise ValueError('Remote namespace belongs to a different protocol')
        return  # explicit resume; immutable inputs already present
    with volume.batch_upload(force=False) as batch:
        for row in manifest:
            batch.put_file(local / row['path'], '/' + prefix + '/' + row['path'])
        for name in ['protocol.json', 'input_manifest.json', 'source_manifest.json', 'source_snapshot.zip']:
            batch.put_file(local / name, '/' + prefix + '/' + name)


@app.local_entrypoint()
def main(run_id: str, budget_usd: float):
    if budget_usd <= 0:
        raise ValueError('Approved controller budget is required')
    upload(run_id)
    print(controller.remote(run_id, budget_usd), flush=True)
