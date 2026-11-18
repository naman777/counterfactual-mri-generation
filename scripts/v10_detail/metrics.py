"""Source-fixed strict regional metrics and predeclared warning handling."""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from scripts.v5_anatomy.metrics import diagnostics,image_array
from scripts.v8_generator_dev.preservation_screen import local_rank
from scripts.v9_detail.masks import noise_aware_mask,geometry,mask_iou
from scripts.v8_generator_dev.preservation_screen import controls,GRID_CONTROLS


def warning(source,thresholds):
    mask,info=noise_aware_mask(source)
    geom=geometry(source,mask)
    # The original per-source calibration seed is supplied by callers via
    # warning_for_index; do not silently change the noise-control draw.
    return mask,info,geom


def warning_for_index(source,index,thresholds):
    mask,info,geom=warning(source,thresholds)
    transformed=controls(source,583110+int(index))
    stable=[mask_iou(mask,noise_aware_mask(transformed[name])[0]) for name in GRID_CONTROLS]
    minimum=float(np.nanmin(stable))
    reasons=[]
    if geom['mask_fraction']<thresholds['area_low'] or geom['mask_fraction']>thresholds['area_high']:reasons.append('train_area_outlier')
    if geom['gradient_energy_inside']<thresholds['gradient_low']:reasons.append('train_gradient_capture_outlier')
    if minimum<thresholds['stability_low']:reasons.append('train_control_stability_outlier')
    if geom['empty_or_no_strict_interior']:reasons.append('strict_interior_under64')
    return dict(qc_warning=int(bool(reasons)),qc_reasons=';'.join(reasons),min_control_iou=minimum,**info,**geom)


def measure(source,edit):
    source,edit=image_array(source),image_array(edit)
    result=diagnostics(source,edit)
    mask,_=noise_aware_mask(source)
    interior=ndi.binary_erosion(mask,iterations=8,border_value=0)
    boundary=ndi.binary_dilation(mask,iterations=3)&~ndi.binary_erosion(mask,iterations=3,border_value=0)
    # Finite 7x7 high-pass kernel support (radius 3); margin 8 keeps SSIM,
    # local rank and HF windows inside the source mask.
    high_source=source-ndi.gaussian_filter(source,1,mode='reflect',truncate=3.)
    high_edit=edit-ndi.gaussian_filter(edit,1,mode='reflect',truncate=3.)
    _,ssim=structural_similarity(source,edit,data_range=1.,win_size=7,full=True)
    ranks=np.abs(local_rank(source)-local_rank(edit))
    for name,domain in [('interior',interior),('boundary',boundary),('whole',np.ones_like(mask))]:
        valid=domain.sum()>=64
        result[name+'_pixels']=int(domain.sum());result[name+'_valid']=bool(valid)
        maps=dict(mad=np.abs(source-edit),ssim=ssim,rank_mad=ranks,
                  hf_residual_mad=np.abs(high_source-high_edit))
        for metric,values in maps.items():result[name+'_'+metric]=float(values[domain].mean()) if valid else np.nan
    return result
