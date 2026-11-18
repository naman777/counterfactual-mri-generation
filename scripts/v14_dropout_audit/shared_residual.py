"""Candidate epsilon combiner only; not a full sampler or established MRI editor."""
import math
import torch


def parent_response_score(guided_unconditional,parent_conditional,parent_unconditional,guidance):
    """Share one guide residual across labels, keeping the parent's fixed-x CFG contrast.

    eps(c,g) = eps_parent(c) + [eps_guided(null,g)-eps_parent(null)].
    Therefore CFG = eps_guided(null,g) + w*(eps_parent(c)-eps_parent(null)).
    This algebra says nothing about reverse trajectories or clinical editing.
    """
    tensors=(guided_unconditional,parent_conditional,parent_unconditional)
    if not all(torch.is_tensor(t) and t.is_floating_point() for t in tensors):
        raise ValueError('Floating epsilon tensors required')
    if any(t.shape!=tensors[0].shape or t.device!=tensors[0].device or t.dtype!=tensors[0].dtype for t in tensors):
        raise ValueError('Matching epsilon shapes/dtypes/devices required')
    if not math.isfinite(guidance) or guidance<0 or not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError('Finite epsilons and nonnegative finite guidance required')
    return guided_unconditional+guidance*(parent_conditional-parent_unconditional)
