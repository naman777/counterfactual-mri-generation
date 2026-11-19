"""Development-only fine-detail stress controls and source-domain measurements."""
import numpy as np
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import image_array
from scripts.v8_generator_dev.preservation_screen import local_rank
from scripts.v8_generator_dev.preservation_screen_v2 import local_variance
from scripts.v9_detail.masks import noise_aware_mask,SETTINGS

METRICS=('rank_patch_max','texture_deficit','hf_patch_max','boundary_change')


def domains(source):
    mask,_=noise_aware_mask(source)
    interior=ndi.binary_erosion(mask,iterations=SETTINGS['strict_interior_margin'],border_value=0)
    boundary=ndi.binary_dilation(mask,iterations=3)&~ndi.binary_erosion(mask,iterations=3,border_value=0)
    return mask,interior,boundary


def measure(source,edit):
    source,edit=image_array(source),image_array(edit)
    if source.shape!=edit.shape:
        raise ValueError('No implicit resize, registration or clipping')
    mask,interior,boundary=domains(source)
    result=dict(valid=bool(interior.sum()>=64 and boundary.sum()>=64),
        strict_interior_pixels=int(interior.sum()),boundary_pixels=int(boundary.sum()),
        **{k:np.nan for k in METRICS})
    if not result['valid']:
        return result
    ranks=np.abs(local_rank(source)-local_rank(edit))
    result['rank_patch_max']=float(ndi.uniform_filter(ranks,size=3,mode='reflect')[interior].max())
    va,vb=local_variance(source),local_variance(edit)
    candidates=va[interior & (va>1e-6)]
    if not len(candidates):
        result['valid']=False
        return result
    textured=interior & (va>=max(1e-6,float(np.median(candidates))))
    ratio=vb[textured]/va[textured]
    typical=float(np.median(ratio))
    result['texture_deficit']=float(np.clip(1-np.min(ratio)/max(typical,1e-12),0,1))
    ha=source-ndi.gaussian_filter(source,1,mode='reflect')
    hb=edit-ndi.gaussian_filter(edit,1,mode='reflect')
    result['hf_patch_max']=float(ndi.uniform_filter(np.abs(ha-hb),size=3,mode='reflect')[interior].max())
    delta=edit-source
    result['boundary_change']=float(np.quantile(np.hypot(*np.gradient(delta))[boundary],.99))
    if not all(np.isfinite(result[k]) for k in METRICS):
        result['valid']=False
    return result


def score(measured,thresholds):
    if not measured['valid']:
        return dict(flag=np.nan,score=np.nan)
    value=max(measured[k]/thresholds[k] for k in METRICS)
    return dict(flag=int(value>1),score=float(value))


def damage_controls(source,seed):
    source=image_array(source)
    mask,interior,boundary=domains(source)
    if interior.sum()<64:
        return []
    variance=local_variance(source)
    gradient=np.hypot(*np.gradient(ndi.gaussian_filter(source,1)))
    rng=np.random.default_rng(seed)
    textured=np.argwhere(interior & (variance>=np.median(variance[interior])))
    if not len(textured):
        return []
    random_location=textured[int(rng.integers(len(textured)))]
    locations=dict(variance_peak=np.unravel_index(np.argmax(np.where(interior,variance,-1)),source.shape),
        gradient_peak=np.unravel_index(np.argmax(np.where(interior,gradient,-1)),source.shape),
        seeded_texture=tuple(random_location))
    output=[]
    for location,(y,x) in locations.items():
        for size in (1,2,3,5):
            y0,x0=int(y-size//2),int(x-size//2)
            edit=source.copy();edit[y0:y0+size,x0:x0+size]=0
            output.append((f'erase_{size}px',location,edit,dict(y=int(y),x=int(x),width_px=size)))
        y0,x0=int(y-3),int(x-3)
        edit=source.copy();blur=ndi.gaussian_filter(source,1.5)
        edit[y0:y0+7,x0:x0+7]=blur[y0:y0+7,x0:x0+7]
        output.append(('local_smooth_7px',location,edit,dict(y=int(y),x=int(x),width_px=7)))
        edit=source.copy()
        patch=source[y0:y0+7,x0:x0+7]
        edit[y0:y0+7,x0:x0+7]=patch[:,::-1]
        output.append(('local_texture_mirror_7px',location,edit,dict(y=int(y),x=int(x),width_px=7)))
    # A deliberately hard final composite creates a boundary appearance seam.
    edit=source.copy();edit[mask]=np.clip(source[mask]*.8,0,1)
    output.append(('boundary_gain_seam','source_mask',edit,dict(y=-1,x=-1,width_px=0)))
    edit=source.copy();moved=ndi.shift(source,(2,2),order=1,mode='constant',cval=0)
    edit[boundary]=moved[boundary]
    output.append(('boundary_translate_2px','source_mask',edit,dict(y=-1,x=-1,width_px=2)))
    return output
