"""Bounded AMP backoff on the same draw, retaining one applied update per batch."""
import torch


def precision_step(forward_loss, parameters, optimizer, scaler, device, max_norm, max_retries=0):
    if not isinstance(max_retries, int) or not 0 <= max_retries <= 16:
        raise ValueError('AMP retry count must be between zero and sixteen')
    parameters = list(parameters)
    device = torch.device(device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
    initial_scale = scaler.get_scale()
    for attempt in range(max_retries + 1):
        if attempt:
            torch.random.set_rng_state(cpu_rng)
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(cuda_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite full-training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        if torch.isfinite(norm):
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= scale
            if not applied:
                raise FloatingPointError('Finite-gradient optimiser update was unexpectedly skipped')
            return loss, norm, dict(applied=True, amp_retries=attempt,
                                    amp_scale_before=initial_scale, amp_scale_after=scaler.get_scale())
        if not scaler.is_enabled() or attempt == max_retries:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite full-training gradient')
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        state = scaler.state_dict()
        state['_growth_tracker'] = 0
        scaler.load_state_dict(state)
    raise AssertionError('Unreachable precision retry state')
