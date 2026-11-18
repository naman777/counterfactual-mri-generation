"""Structural consistency proxies and synthetic controls; no clinical anatomy labels."""
import numpy as np
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import diagnostics, foreground_proxy, image_array

METRICS = ('rank_mad_roi', 'rank_patch_max', 'edge_chamfer_px')
GRID_CONTROLS = ('gamma_07', 'gamma_13', 'gain_08', 'bias_003', 'noise_0005', 'blur_05')
DAMAGE_CONTROLS = ('translate_2px', 'translate_4px', 'local_warp_4px', 'erase_5px', 'erase_12px')


def local_rank(image, radius=2):
    image = image_array(image)
    pad = np.pad(image, radius, mode='reflect')
    result = np.zeros_like(image)
    for y in range(2*radius+1):
        for x in range(2*radius+1):
            neighbour = pad[y:y+image.shape[0], x:x+image.shape[1]]
            result += (neighbour < image) + .5*(neighbour == image)
    return result / (2*radius+1)**2


def edge_map(image, domain):
    gradient = np.hypot(*np.gradient(ndi.gaussian_filter(image, 1, mode='reflect')))
    if not domain.any() or gradient[domain].max() < 1e-6:
        return np.zeros_like(domain)
    threshold = max(1e-6, float(np.quantile(gradient[domain], .8)))
    return (gradient >= threshold) & domain


def measurements(source, edit):
    source, edit = image_array(source), image_array(edit)
    if source.shape != edit.shape:
        raise ValueError('Source and edit must share the native source grid')
    result = diagnostics(source, edit)
    roi = foreground_proxy(source, .05)
    domain = ndi.binary_dilation(roi, iterations=4)
    a, b = edge_map(source, domain), edge_map(edit, domain)
    valid = bool(roi.sum() >= 64 and a.any() and b.any())
    result.update(screen_measurements_valid=valid,
                  rank_mad_roi=None, rank_patch_max=None, edge_chamfer_px=None)
    if valid:
        difference = np.abs(local_rank(source) - local_rank(edit))
        patch = ndi.uniform_filter(difference, size=8, mode='reflect')
        result.update(rank_mad_roi=float(difference[roi].mean()),
            rank_patch_max=float(patch[roi].max()),
            edge_chamfer_px=float((ndi.distance_transform_edt(~b)[a].mean()+
                                   ndi.distance_transform_edt(~a)[b].mean())/2))
    return result


def screen(measured, thresholds):
    if not measured['screen_measurements_valid']:
        return dict(screen_score=None, within_calibrated_envelope=None, screen_flag=None)
    value = max(measured[name]/thresholds[name] for name in METRICS)
    return dict(screen_score=float(value), within_calibrated_envelope=int(value <= 1), screen_flag=int(value > 1))


def controls(source, seed):
    source = image_array(source)
    rng = np.random.default_rng(seed)
    values = dict(gamma_07=source**.7, gamma_13=source**1.3,
        gain_08=source*.8, bias_003=np.clip(source+.03, 0, 1),
        noise_0005=np.clip(source+rng.normal(0, .005, source.shape), 0, 1),
        blur_05=ndi.gaussian_filter(source, .5, mode='reflect'))
    for pixels in (2, 4):
        values[f'translate_{pixels}px'] = ndi.shift(source, (pixels, pixels), order=1, mode='constant', cval=0)
    yy, xx = np.indices(source.shape, dtype=float)
    envelope = np.exp(-((yy-source.shape[0]/2)**2+(xx-source.shape[1]/2)**2)/(2*18**2))
    values['local_warp_4px'] = ndi.map_coordinates(source, [yy+4*envelope, xx-4*envelope], order=1, mode='reflect')
    roi = foreground_proxy(source, .05)
    gradient = np.hypot(*np.gradient(ndi.gaussian_filter(source, 1)))
    domain = ndi.binary_erosion(roi, iterations=6)
    if not domain.any():
        domain = roi
    y, x = np.unravel_index(np.argmax(np.where(domain, gradient, -1)), source.shape)
    for size in (5, 12):
        value = source.copy()
        y0, x0 = np.clip(y-size//2, 0, source.shape[0]-size), np.clip(x-size//2, 0, source.shape[1]-size)
        value[y0:y0+size, x0:x0+size] = 0
        values[f'erase_{size}px'] = value
    return {name: np.asarray(value, np.float32) for name, value in values.items()}
