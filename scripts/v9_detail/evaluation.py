"""Explicit engineering evaluator routes for guide and mask ablations."""
import numpy as np
import torch
from scipy import ndimage as ndi

from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.detail_candidates import multiscale_guide,masked_sample
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v9_detail.masks import noise_aware_mask

GUIDE_CONTROLS=('correct','zero','absent','wrong','coarse_only','fine_only')
MASK_CONTROLS=('unmasked','noise_aware','full','empty','translated','wrong','final_composite')


@torch.inference_mode()
def evaluate_case(trainer,item,source,axis,seed,guide_control='correct',mask_control='unmasked',
                  wrong_source=None,timestep=2,guidance=3.):
    if axis not in ('manufacturer','field_strength') or guide_control not in GUIDE_CONTROLS or mask_control not in MASK_CONTROLS:
        raise ValueError('Undeclared engineering route')
    if timestep!=2 or guidance!=3.:
        raise ValueError('This evaluator is capped at engineering t=2,w=3; no efficacy generation')
    device=trainer.device
    image=item['image'].unsqueeze(0).to(device)
    source=np.asarray(source,np.float32)
    if not np.allclose(((image[0,0]+1)/2).cpu().numpy(),source,atol=6e-8,rtol=0):
        raise ValueError('Guide/source image identity mismatch')
    if guide_control=='wrong' or mask_control=='wrong':
        if wrong_source is None or np.array_equal(source,wrong_source):
            raise ValueError('Distinct preselected wrong source required')
    guide_image=wrong_source if guide_control=='wrong' else source
    guide=multiscale_guide(torch.from_numpy(guide_image.copy())[None,None].to(device),trainer.mode)
    if guide_control=='zero':guide=torch.zeros_like(guide)
    if guide_control=='coarse_only':guide[:,3:]=0
    if guide_control=='fine_only':guide[:,:3]=0
    present=torch.zeros(1,device=device) if guide_control=='absent' else torch.ones(1,device=device)
    bound=BoundStructureModel(trainer.model,guide,present)
    mask,_=noise_aware_mask(wrong_source if mask_control=='wrong' else source)
    if mask_control=='full':mask=np.ones_like(mask)
    if mask_control=='empty':mask=np.zeros_like(mask)
    if mask_control=='translated':mask=ndi.shift(mask.astype(np.float32),(4,4),order=0,mode='constant')>0
    mask_tensor=torch.from_numpy(mask.astype(np.float32))[None,None].to(device)
    m,f=item['manufacturer_label'],item['field_strength_label']
    tm,tf=(1-m,f) if axis=='manufacturer' else (m,1-f)
    arrays=dict(source=source)
    sampler_source=((image+1)/2).cpu().numpy()[0,0]
    for name,mfg,fs in [('null',m,f),('target',tm,tf)]:
        args=(trainer.c,bound,trainer.scheduler,image,torch.tensor([mfg],device=device),
            torch.tensor([fs],device=device),torch.tensor([item['echo_time_norm']],device=device),int(seed),timestep,guidance)
        if mask_control in ('unmasked','final_composite'):
            value=sample_wiring(*args)[0,0]
            if mask_control=='final_composite':value=np.where(mask,value,sampler_source)
        else:value=masked_sample(*args,mask_tensor)[0,0]
        arrays[name]=value
    return arrays,dict(source_echo_time_norm=float(item['echo_time_norm']),
        target_echo_time_norm=float(item['echo_time_norm']),target_manufacturer=tm,target_field_strength=tf,
        guide_control=guide_control,mask_control=mask_control,actual_sampling_seed=int(seed),
        guide_mode=trainer.mode,timestep=timestep,guidance=guidance,
        copied_pixel_fraction=float((~mask).mean()) if mask_control!='unmasked' else 0.,
        engineering=True)
