"""Select matched checkpoints after a verified numerical failure, without promotion."""
import json
import math
from pathlib import Path


RATE_WITH_MARGIN = (0.000306 + 2 * 0.0000131 + 16 * 0.00000222) * 1.2


def select_failed_pair(training, failure, training_log):
    training = Path(training)
    if failure.get('complete') is not False or (training / 'complete.json').exists():
        raise ValueError('Expected an incomplete failed training run')
    if 'FloatingPointError: Nonfinite full-training gradient' not in training_log:
        raise ValueError('Numerical-gradient failure is not evidenced')
    if not any(row.get('phase') == 'train' and row.get('exit_code') == 1
               for row in failure.get('phases', [])):
        raise ValueError('Missing failed training phase receipt')
    candidates = []
    for path in sorted(training.glob('block_*/complete.json')):
        record = json.loads(path.read_text())
        progress = record['progress']
        if record['engineering'] or not record['matched_sources_timesteps_noise']:
            raise ValueError('Invalid paired provenance')
        if set(progress) != {'additive', 'scale_shift'} or progress['additive'] != progress['scale_shift']:
            raise ValueError('Unmatched training exposure')
        if progress['additive']['sample_offset'] != 0:
            raise ValueError('Partial epoch is not eligible')
        if set(record['checkpoints']) != set(progress) or set(record['checkpoint_hashes']) != set(progress):
            raise ValueError('Missing pair checkpoint identities')
        candidates.append((progress['additive']['epoch'], record, path))
    if not candidates:
        raise ValueError('No fully matched completed checkpoint block')
    epoch, record, path = max(candidates, key=lambda item: item[0])
    return dict(common_epoch=epoch, checkpoint_record=record, source_block=str(path),
                selection='Latest fully matched block before numerical failure; no outcome selection',
                full_training_completed=False, converged=False, anatomy_validated=False,
                comparison_role='numerically_interrupted_development_only')


def allowance(launch_seconds, failure_seconds, approved_total=6.0):
    values = [float(value) for value in launch_seconds]
    failure_seconds = float(failure_seconds)
    if not values or any(not math.isfinite(value) or value < 0 for value in values + [failure_seconds]):
        raise ValueError('Finite nonnegative launch and failure durations required')
    if approved_total != 6.0:
        raise ValueError('Only the original approved US$6 allowance is supported')
    debit_seconds = max(sum(values), failure_seconds) + 60
    estimated_prior = debit_seconds * RATE_WITH_MARGIN + 0.50
    remaining = max(0, math.floor((approved_total - estimated_prior) * 100) / 100)
    return dict(approved_total_usd=approved_total, launch_seconds=values,
                failure_seconds=failure_seconds, debit_seconds=debit_seconds,
                estimated_prior_usd=estimated_prior, remaining_usd=remaining,
                rate_with_margin=RATE_WITH_MARGIN, exact_bill_known=False,
                assumptions='All prior launch wall times, 60s pad, 20% rate margin, shared US$0.50 reserve',
                pricing_source='https://modal.com/pricing', pricing_verified_date='2026-11-16')
