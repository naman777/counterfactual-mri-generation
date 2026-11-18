"""Honest last-common-checkpoint recovery; never label a resource stop converged."""
import json
import math
from pathlib import Path


RATE_WITH_MARGIN = (0.000306 + 2 * 0.0000131 + 16 * 0.00000222) * 1.2


def remaining_allowance(failure, approved_total=6.0):
    seconds = float(failure['seconds'])
    if not math.isfinite(seconds) or seconds < 0 or approved_total != 6.0:
        raise ValueError('Invalid prior-run time or approval')
    estimated_used = (seconds + 60) * RATE_WITH_MARGIN + 0.50
    remaining = max(0, math.floor((approved_total - estimated_used) * 100) / 100)
    return dict(approved_total_usd=approved_total, estimated_prior_usd=estimated_used,
                remaining_usd=remaining, source='https://modal.com/pricing',
                assumptions='Same A10/2 CPU/16 GiB and 20% margin; prior runtime plus 60s pad and US$0.50 shared startup reserve',
                exact_bill_known=False)


def choose_common(training):
    training = Path(training)
    if (training / 'complete.json').exists() or not (training / 'resource_stop.json').exists():
        raise ValueError('Recovery is restricted to an explicitly incomplete resource-stop run')
    candidates = []
    for path in sorted(training.glob('block_*/complete.json')):
        row = json.loads(path.read_text())
        if row['engineering'] or not row['matched_sources_timesteps_noise']:
            raise ValueError('Invalid common-block provenance')
        a, b = row['progress']['additive'], row['progress']['scale_shift']
        if a != b or a['sample_offset'] != 0:
            raise ValueError('Unmatched/partial epoch is not a common primary checkpoint')
        candidates.append((a['epoch'], path, row))
    if not candidates:
        raise ValueError('No completed common block exists')
    epoch, path, row = max(candidates, key=lambda item: item[0])
    return dict(selection='latest completed common epoch before the resource stop; no outcome selection',
                common_epoch=epoch, source_block=str(path), checkpoint_record=row,
                full_training_completed=False, converged=False,
                comparison_role='resource_limited_development_only', anatomy_validated=False)
