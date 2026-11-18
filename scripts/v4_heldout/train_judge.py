"""Prepared B2 trainer; no automatic launch or cloud expenditure.

Matches the original architecture, intensity/noise augmentation and selection
recipe. Primary held-out judges also randomise vertical orientation, following
the documented vendor/orientation confound. The old recipe remains a control.
Requires the new completed training/validation pack; never loads original test
images. A Modal wrapper and approved allowance are still needed for GPU runs.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.v2_quickwins.judge_model import Judge


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def augment(x, vertical_flip=True):
    b = len(x)
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if vertical_flip:
        flip_y = torch.rand(b, device=x.device) < 0.5
        x = torch.where(flip_y[:, None, None, None], x.flip(-2), x)
    dx, dy = np.random.randint(-6, 7, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))
    alpha = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    beta = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.05, 0.05)
    x = (x * alpha + beta).clamp(0, 1)
    gamma = torch.exp(torch.empty(b, 1, 1, 1, device=x.device).uniform_(np.log(0.8), np.log(1.25)))
    x = x.clamp_min(1e-6) ** gamma
    sigma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0, 0.05)
    use = (torch.rand(b, 1, 1, 1, device=x.device) < 0.7).float()
    return (x + use * sigma * torch.randn_like(x)).clamp(0, 1)


def load_pack(path):
    path = Path(path)
    receipt = json.loads((path.parent / 'summary.json').read_text(encoding='utf-8'))
    if not receipt['complete'] or file_sha(path) != receipt['pack_sha256']:
        raise ValueError('Pack is incomplete or its SHA-256 differs from receipt')
    with np.load(path, allow_pickle=False) as stream:
        data = {k: stream[k] for k in stream.files}
    if any(k.startswith('test_') for k in data):
        raise ValueError('Training pack must not contain original test data')
    excluded = set(data['excluded_patient_ids'])
    assert len(excluded) == 103
    assert set(data['train_patient']).isdisjoint(set(data['val_patient']))
    for split, expected_patients in [('train', 48), ('val', 16)]:
        patient = data[f'{split}_patient']
        assert len(set(patient)) == expected_patients and set(patient).isdisjoint(excluded)
        x = data[f'{split}_x']
        assert x.dtype == np.float32 and x.shape == (len(patient), 128, 128)
        assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        for axis in ['mfg', 'fs']:
            y = data[f'{split}_{axis}']
            assert len(y) == len(x) and set(y) == {0, 1}
            assert all(len(set(y[patient == p])) == 1 for p in set(patient))
    return data, receipt


def bacc(y, predicted):
    return float(np.mean([(predicted[y == c] == c).mean() for c in [0, 1]]))


def train(pack_path, output, seed, epochs=40, device='cuda', vertical_flip=True, time_limit_s=None):
    if epochs <= 0 or seed not in [0, 1, 2]:
        raise ValueError('Use the prespecified three seeds and a positive epoch count')
    data, receipt = load_pack(pack_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Judge().to(device)
    x = {s: torch.from_numpy(data[f'{s}_x']).unsqueeze(1) for s in ['train', 'val']}
    y = {s: {a: torch.from_numpy(data[f'{s}_{a}']) for a in ['mfg', 'fs']} for s in x}
    losses = {}
    for axis in ['mfg', 'fs']:
        counts = torch.bincount(y['train'][axis], minlength=2).float()
        losses[axis] = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    batch = 64
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * math.ceil(len(x['train']) / batch), pct_start=0.15)

    @torch.no_grad()
    def validate(noise, transform='identity'):
        model.eval()
        probs = {axis: [] for axis in ['mfg', 'fs']}
        rng = torch.Generator(device=device).manual_seed(0)
        for i in range(0, len(x['val']), 256):
            xb = x['val'][i:i + 256].to(device)
            if transform == 'vertical_flip':
                xb = xb.flip(-2)
            elif transform == 'rotate_180':
                xb = xb.flip((-2, -1))
            if noise:
                xb = (xb + noise * torch.randn(xb.shape, device=device, generator=rng)).clamp(0, 1)
            outputs = model(xb)
            for axis in probs:
                probs[axis].append(F.softmax(outputs[axis], 1).cpu().numpy())
        probs = {axis: np.concatenate(values) for axis, values in probs.items()}
        scores = {}
        for axis, values in probs.items():
            truth = y['val'][axis].numpy()
            pred = values.argmax(1)
            scores[f'{axis}_acc'] = float((truth == pred).mean())
            scores[f'{axis}_bacc'] = bacc(truth, pred)
            scores[f'{axis}_patient_mean_slice_acc'] = float(np.mean([(truth[data['val_patient'] == p] == pred[data['val_patient'] == p]).mean() for p in np.unique(data['val_patient'])]))
        return scores, probs

    best, best_state, best_epoch, history = -1.0, None, None, []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x['train']))
        total = 0.0
        for start in range(0, len(perm), batch):
            indices = perm[start:start + batch]
            outputs = model(augment(x['train'][indices].to(device), vertical_flip=vertical_flip))
            loss = sum(losses[a](outputs[a], y['train'][a][indices].to(device)) for a in losses)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            total += loss.item() * len(indices)
        clean, _ = validate(0)
        noisy, _ = validate(0.03)
        score = float(np.mean([scores[f'{axis}_bacc'] for scores in [clean, noisy] for axis in ['mfg', 'fs']]))
        row = {'epoch': epoch, 'train_loss': total / len(perm), 'clean_val': clean, 'noisy_val': noisy, 'select': score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: tensor.detach().cpu().clone() for k, tensor in model.state_dict().items()}
        if time_limit_s is not None and time.perf_counter() - training_started >= time_limit_s:
            break  # preserve the best partial state before the outer cloud timeout
    model.load_state_dict(best_state)
    clean, clean_probs = validate(0)
    noisy, noisy_probs = validate(0.03)
    flip_scores, flip_probs = validate(0, 'vertical_flip')
    rotate_scores, rotate_probs = validate(0, 'rotate_180')
    checkpoint = out / f'judge_heldout_resnet18_seed{seed}.pt'
    with checkpoint.open('xb') as stream:
        torch.save(best_state, stream)
    result = {'complete': len(history) == epochs, 'seed': seed, 'epochs': epochs, 'epochs_completed': len(history), 'best_epoch': best_epoch, 'best_val_select': best, 'history': history, 'val_clean': clean, 'val_noise03': noisy, 'val_vertical_flip': flip_scores, 'val_rotate_180': rotate_scores, 'independent_vertical_flip_augmentation_p': 0.5 if vertical_flip else 0.0, 'judge_role': 'orientation_robust_heldout' if vertical_flip else 'legacy_augmentation_comparison', 'pack_sha256': receipt['pack_sha256'], 'source_sha256': file_sha(__file__), 'checkpoint_sha256': file_sha(checkpoint), 'test_evaluated': False, 'selection': 'mean clean/noise03 slice balanced accuracy across two heads; ties keep earliest epoch; orientation stress is reported after checkpoint freezing'}
    (out / 'train.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / 'validation_predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, patient=data['val_patient'], sop=data['val_sop'], mfg=data['val_mfg'], fs=data['val_fs'], clean_mfg=clean_probs['mfg'], clean_fs=clean_probs['fs'], noisy_mfg=noisy_probs['mfg'], noisy_fs=noisy_probs['fs'], flip_mfg=flip_probs['mfg'], flip_fs=flip_probs['fs'], rotate_mfg=rotate_probs['mfg'], rotate_fs=rotate_probs['fs'])
    (out / 'README.md').write_text('Generated by scripts/v4_heldout/train_judge.py. train.json records the frozen data/source/checkpoint hashes, orientation-augmentation role, training history and validation-selected epoch. judge_heldout_resnet18_seed*.pt is a state dict compatible with scripts/v2_quickwins/judge_model.py. validation_predictions.npz contains new-cohort clean/noisy and orientation-stress predictions, evaluated after freezing the checkpoint. Primary judges independently sample both horizontal and vertical flips; --legacy-horizontal-only preserves the old augmentation as a comparison. Original test patients are excluded from training and checkpoint selection; their accuracy and counterfactual re-grades remain separate future steps.\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pack', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--legacy-horizontal-only', action='store_true', help='Comparison arm only: omit the primary orientation-robust vertical flips')
    args = parser.parse_args()
    train(args.pack, args.output, args.seed, args.epochs, args.device, vertical_flip=not args.legacy_horizontal_only)
