"""Differentiable local-order guide candidate, without raw clean-pixel bypass."""
import torch
import torch.nn.functional as F


def soft_structure(images):
    if images.ndim != 4 or images.shape[1:] != (1, 128, 128):
        raise ValueError('Expected source images [B,1,128,128]')
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Expected finite source intensities in [0,1]')
    image = images.float()
    padded = F.pad(image, (2, 2, 2, 2), mode='reflect')
    mean = F.avg_pool2d(padded, 5, stride=1)
    variance = (F.avg_pool2d(padded.square(), 5, stride=1)-mean.square()).clamp_min(0)
    scale = torch.sqrt(variance+1e-8)*.25
    order = torch.zeros_like(image)
    for y in range(5):
        for x in range(5):
            neighbour = padded[:, :, y:y+128, x:x+128]
            order = order + torch.sigmoid((image-neighbour)/scale)
    coarse = F.avg_pool2d(order/25, 4)
    gy, gx = torch.gradient(coarse, dim=(-2, -1))
    # Fixed coarse bottleneck; no raw intensity channel or acquisition label.
    return torch.cat([2*coarse-1, gx*2, gy*2], 1).clamp(-1, 1)
