"""Geometry diagnostics, explicitly separated from verified anatomy labels.

Thresholded foreground is a contrast-sensitive proxy, not breast/tumour ground
truth. Measurements remain in the unregistered source pixel grid.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity


def image_array(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or min(value.shape) < 8:
        raise ValueError('Expected a 2D image with both dimensions >= 8')
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError('Images must be finite and already in [0,1]; no silent clipping')
    return value


def mask_array(value, shape):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Mask must be finite and match the native source grid')
    if not np.isin(value, [0, 1]).all():
        raise ValueError('Mask must be binary; do not resize annotations implicitly')
    return value.astype(bool)


def foreground_proxy(image, threshold=.05):
    image = image_array(image)
    mask = ndi.gaussian_filter(image, sigma=1, mode='reflect') > threshold
    labels, n = ndi.label(mask)
    if n:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= max(16, int(image.size * .002))
        keep[0] = False
        mask = keep[labels]
    return ndi.binary_fill_holes(mask)


def mask_metrics(source_mask, edit_mask, tolerance_px=2.):
    a, b = np.asarray(source_mask, bool), np.asarray(edit_mask, bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Masks must share a 2D grid')
    na, nb = int(a.sum()), int(b.sum())
    # Empty masks do not certify preservation, even when both are empty.
    if na == 0 or nb == 0:
        return dict(valid=False, dice=None, iou=None, area_ratio=None,
                    centroid_shift_px=None, boundary_hd95_px=None, boundary_f1=None)
    intersection = int((a & b).sum())
    ca, cb = np.asarray(ndi.center_of_mass(a)), np.asarray(ndi.center_of_mass(b))
    sa = a & ~ndi.binary_erosion(a, border_value=0)
    sb = b & ~ndi.binary_erosion(b, border_value=0)
    ab = ndi.distance_transform_edt(~sb)[sa]
    ba = ndi.distance_transform_edt(~sa)[sb]
    # Conservative symmetric HD95: maximum of the two directed percentiles.
    precision = float((ba <= tolerance_px).mean())
    recall = float((ab <= tolerance_px).mean())
    return dict(valid=True, dice=2*intersection/(na+nb), iou=intersection/int((a|b).sum()),
                area_ratio=nb/na, centroid_shift_px=float(np.linalg.norm(cb-ca)),
                boundary_hd95_px=float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
                boundary_f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def landmark_metrics(source, edit, shape):
    if not isinstance(source, dict) or not source or set(source) != set(edit):
        raise ValueError('Landmark maps must contain identical non-empty named points')
    distances = []
    for name in sorted(source):
        a, b = np.asarray(source[name], float), np.asarray(edit[name], float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            raise ValueError('Landmarks must be finite [x,y] coordinates')
        if any(p[0] < 0 or p[1] < 0 or p[0] >= shape[1] or p[1] >= shape[0] for p in [a, b]):
            raise ValueError('Landmark outside native source pixel grid')
        distances.append(float(np.linalg.norm(a-b)))
    return dict(landmark_mean_shift_px=float(np.mean(distances)),
                landmark_max_shift_px=float(max(distances)), landmark_count=len(distances))


def diagnostics(source, edit, thresholds=(.03, .05, .1), gates=None):
    source, edit = image_array(source), image_array(edit)
    if source.shape != edit.shape:
        raise ValueError('Source and edit must share a grid; no registration/resizing')
    result = dict(ssim=float(structural_similarity(source, edit, data_range=1., win_size=7)),
                  mad=float(np.abs(source-edit).mean()), anatomy_validated=False)
    masks = [mask_metrics(foreground_proxy(source, t), foreground_proxy(edit, t)) for t in thresholds]
    result['proxy_masks_valid'] = all(m['valid'] for m in masks)
    for key in ['dice', 'iou', 'area_ratio', 'centroid_shift_px', 'boundary_hd95_px', 'boundary_f1']:
        values = [m[key] for m in masks if m[key] is not None]
        result['proxy_' + key + '_min'] = float(min(values)) if values else None
        result['proxy_' + key + '_max'] = float(max(values)) if values else None
    source_coarse = ndi.gaussian_filter(source, 2)
    edit_coarse = ndi.gaussian_filter(edit, 2)
    ga = np.hypot(*np.gradient(source_coarse)).ravel()
    gb = np.hypot(*np.gradient(edit_coarse)).ravel()
    result['coarse_gradient_ncc'] = float(np.corrcoef(ga, gb)[0, 1]) if ga.std() > 1e-8 and gb.std() > 1e-8 else None
    foreground = foreground_proxy(source, .05)
    src_hf = float(np.mean(ndi.laplace(source)[foreground]**2)) if foreground.any() else 0.
    edit_hf = float(np.mean(ndi.laplace(edit)[foreground]**2)) if foreground.any() else 0.
    result['foreground_hf_ratio_to_source'] = edit_hf/src_hf if src_hf > 1e-12 else None
    if gates:
        result['proxy_screen_pass'] = bool(result['proxy_masks_valid'] and
            result['proxy_dice_min'] >= gates['min_dice'] and
            result['proxy_boundary_hd95_px_max'] <= gates['max_boundary_hd95_px'] and
            result['proxy_centroid_shift_px_max'] <= gates['max_centroid_shift_px'] and
            result['proxy_area_ratio_min'] >= 1-gates['max_area_fraction_change'] and
            result['proxy_area_ratio_max'] <= 1+gates['max_area_fraction_change'])
    return result
