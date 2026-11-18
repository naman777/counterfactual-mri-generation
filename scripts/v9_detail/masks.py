"""Source-only noise-aware mask candidate; no breast or tumour truth implied."""
import numpy as np
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import foreground_proxy,image_array

SETTINGS=dict(corner_width=12,corner_quantile=.99,threshold_floor=.03,
              noise_margin=.01,min_component_pixels=32,max_hole_pixels=64,
              dilation_pixels=2,strict_interior_margin=8)


def noise_aware_mask(source):
    source=image_array(source)
    if source.shape!=(128,128):
        raise ValueError('Candidate fixed to the canonical 128px source grid')
    smooth=ndi.gaussian_filter(source,1,mode='reflect')
    k=SETTINGS['corner_width']
    corners=[smooth[:k,:k],smooth[:k,-k:],smooth[-k:,:k],smooth[-k:,-k:]]
    # Deterministic lowest-median corner. It can contain tissue; report this risk.
    corner=int(np.argmin([np.median(c) for c in corners]))
    noise=float(np.quantile(corners[corner],SETTINGS['corner_quantile']))
    threshold=max(SETTINGS['threshold_floor'],noise+SETTINGS['noise_margin'])
    mask=smooth>threshold
    labels,n=ndi.label(mask)
    if n:
        sizes=np.bincount(labels.ravel())
        keep=sizes>=SETTINGS['min_component_pixels'];keep[0]=False
        mask=keep[labels]
    holes=ndi.binary_fill_holes(mask)&~mask
    labels,n=ndi.label(holes)
    if n:
        sizes=np.bincount(labels.ravel())
        keep=sizes<=SETTINGS['max_hole_pixels'];keep[0]=False
        mask=mask|keep[labels]
    mask=ndi.binary_dilation(mask,iterations=SETTINGS['dilation_pixels'])
    return mask,dict(source_threshold=threshold,corner_index=corner,corner_noise_quantile=noise)


def make_mask(source,variant):
    if variant=='historical_005':
        return foreground_proxy(source,.05),dict(source_threshold=.05,corner_index=-1,corner_noise_quantile=np.nan)
    if variant=='noise_aware_v1':
        return noise_aware_mask(source)
    raise ValueError('Unknown mask variant')


def geometry(source,mask):
    interior=ndi.binary_erosion(mask,iterations=SETTINGS['strict_interior_margin'],border_value=0)
    boundary=ndi.binary_dilation(mask,iterations=3)&~ndi.binary_erosion(mask,iterations=3,border_value=0)
    energy=np.hypot(*np.gradient(ndi.gaussian_filter(source,1)))**2
    total=float(energy.sum())
    return dict(mask_fraction=float(mask.mean()),strict_interior_pixels=int(interior.sum()),
        boundary_pixels=int(boundary.sum()),
        gradient_energy_inside=float(energy[mask].sum()/total) if total>1e-12 else np.nan,
        empty_or_no_strict_interior=int(interior.sum()<64),near_full_frame=int(mask.mean()>=.9))


def mask_iou(a,b):
    union=(a|b).sum()
    return float((a&b).sum()/union) if union else np.nan
