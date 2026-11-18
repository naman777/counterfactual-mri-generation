"""Isolated additive/scale-shift variants; historical architecture is untouched."""
import copy

import torch
from torch import nn

from src.models.conditioning import AcquisitionConditionEmbedding
from src.models.ddpm_scheduler import DDPMScheduler
from src.models.diffusion_unet import MRIConditionalUNet, ResidualBlock


class ScaleShiftResidualBlock(ResidualBlock):
    """Time/acquisition modulation after second GroupNorm, before SiLU.

    norm2(conv1(...)) * (1 + scale(time,condition)) + shift(time,condition).
    This is a new architecture, not a legacy checkpoint-compatible replacement.
    """

    def __init__(self, in_channels, out_channels, time_condition_dim, dropout=0.1):
        super().__init__(in_channels, out_channels, time_condition_dim, dropout)
        self.time_condition_proj = nn.Linear(time_condition_dim, 2 * out_channels)

    def forward(self, x, time_condition):
        h = self.conv1(self.activation(self.norm1(x)))
        scale, shift = self.time_condition_proj(self.activation(time_condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(self.activation(h)))
        return h + self.shortcut(x)


class ScaleShiftMRIUNet(MRIConditionalUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, block in list(self.named_children()):
            if isinstance(block, ResidualBlock):
                new = ScaleShiftResidualBlock(block.conv1.in_channels, block.conv1.out_channels,
                                               block.time_condition_proj.in_features, block.dropout.p)
                setattr(self, name, new)


def build_models(config, variant, device='cpu'):
    if variant not in {'additive', 'scale_shift'}:
        raise ValueError('Unknown conditioning variant')
    # Build common weights first with the same seed in both runs.
    conditioner = AcquisitionConditionEmbedding(**config['condition'])
    baseline = MRIConditionalUNet(**config['model'])
    if variant == 'additive':
        model = baseline
    else:
        model = ScaleShiftMRIUNet(**config['model'])
        common = baseline.state_dict()
        state = model.state_dict()
        for name, value in common.items():
            if state[name].shape == value.shape:
                state[name].copy_(value)
        for name, block in model.named_children():
            if isinstance(block, ScaleShiftResidualBlock):
                original = getattr(baseline, name).time_condition_proj
                channels = original.out_features
                state[f'{name}.time_condition_proj.weight'][:channels].zero_()
                state[f'{name}.time_condition_proj.bias'][:channels].zero_()
                state[f'{name}.time_condition_proj.weight'][channels:].copy_(original.weight)
                state[f'{name}.time_condition_proj.bias'][channels:].copy_(original.bias)
        model.load_state_dict(state)
    scheduler = DDPMScheduler(**config['diffusion'])
    return conditioner.to(device), model.to(device), scheduler.to(device)


def clone_ema(module):
    return copy.deepcopy(module).eval().requires_grad_(False)


@torch.no_grad()
def update_ema(ema, model, decay):
    if not 0 <= decay < 1:
        raise ValueError('EMA decay must be in [0,1)')
    state = ema.state_dict()
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            state[name].lerp_(value, 1 - decay)
        else:
            state[name].copy_(value)


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
