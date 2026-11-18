"""Additional local texture-loss proxy; validation-informed benchmark development."""
import numpy as np
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import foreground_proxy
from scripts.v8_generator_dev.preservation_screen import (
    GRID_CONTROLS, DAMAGE_CONTROLS, controls,
    measurements as previous_measurements, METRICS as PREVIOUS_METRICS,
)

METRICS = (*PREVIOUS_METRICS, 'local_texture_deficit')


def local_variance(image):
    image = np.asarray(image, np.float64)
    mean = ndi.uniform_filter(image, size=5, mode='reflect')
    return np.maximum(0, ndi.uniform_filter(image**2, size=5, mode='reflect')-mean**2)


def measurements(source, edit):
    result = previous_measurements(source, edit)
    source_var, edit_var = local_variance(source), local_variance(edit)
    roi = foreground_proxy(source, .05)
    positive = source_var[roi & (source_var > 1e-6)]
    result['local_texture_deficit'] = None
    if not len(positive):
        result['screen_measurements_valid'] = False
        return result
    textured = roi & (source_var >= max(1e-6, float(np.quantile(positive, .5))))
    ratio = edit_var[textured] / source_var[textured]
    typical = float(np.median(ratio))
    if typical <= 1e-12:
        result['local_texture_deficit'] = 1.
    else:
        result['local_texture_deficit'] = float(np.clip(1-np.min(ratio/typical), 0, 1))
    return result


# The base screen reads its module's metric tuple. Avoid changing that shared
# module; this version explicitly scores the additional metric itself.
def screen(measured, thresholds):
    if not measured['screen_measurements_valid']:
        return dict(screen_score=None, within_calibrated_envelope=None, screen_flag=None)
    value = max(measured[name]/thresholds[name] for name in METRICS)
    return dict(screen_score=float(value), within_calibrated_envelope=int(value <= 1), screen_flag=int(value > 1))
