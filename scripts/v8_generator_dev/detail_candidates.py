"""Untrained guide and mask-ablation candidates; separate from historical routes.

No candidate is wired into an existing evaluator or given efficacy status.
"""
import torch
from torch import nn
import torch.nn.functional as F

from scripts.v6_structure.model import StructureAdapter, StructureConditionedUNet


def local_order_native(images):
    if images.ndim != 4 or images.shape[1:] != (1,128,128):
        raise ValueError('Expected [B,1,128,128] source in [0,1]')
    if not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
        raise ValueError('Invalid source values')
    image=images.float()
    padded=F.pad(image,(2,2,2,2),mode='reflect')
    mean=F.avg_pool2d(padded,5,stride=1)
    variance=(F.avg_pool2d(padded.square(),5,stride=1)-mean.square()).clamp_min(0)
    scale=torch.sqrt(variance+1e-8)*.25
    order=torch.zeros_like(image)
    for y in range(5):
        for x in range(5):
            order=order+torch.sigmoid((image-padded[:,:,y:y+128,x:x+128])/scale)
    return order/25


def order_maps(order, factor):
    coarse=F.avg_pool2d(order,factor)
    gy,gx=torch.gradient(coarse,dim=(-2,-1))
    return torch.cat([2*coarse-1,2*gx,2*gy],1).clamp(-1,1)


def multiscale_guide(images, mode='coarse_fine'):
    """Matched six-channel inputs; fine detail may also carry acquisition labels."""
    if mode not in ('coarse_fine','coarse_control','learned_zero'):
        raise ValueError('Undeclared guide candidate')
    order=local_order_native(images)
    coarse=F.interpolate(order_maps(order,4),size=(64,64),mode='bilinear',align_corners=False)
    fine=order_maps(order,2) if mode=='coarse_fine' else coarse
    guide=torch.cat([coarse,fine],1)
    return torch.zeros_like(guide) if mode=='learned_zero' else guide


class MatchedDetailAdapter(StructureAdapter):
    """Identical capacity for coarse/coarse and coarse/fine inputs."""
    def __init__(self,base_channels=(64,128,256),width=16):
        super().__init__(base_channels,width)
        self.blocks[0][0]=nn.Conv2d(6,width,3,padding=1)

    def forward(self,structure):
        if structure.ndim != 4 or structure.shape[1:] != (6,64,64):
            raise ValueError('Expected six local-order channels at 64x64')
        h=F.interpolate(structure,size=(128,128),mode='bilinear',align_corners=False)
        controls=[]
        for block,output in zip(self.blocks,self.outputs):
            h=block(h)
            controls.append(output(h))
        return controls


class DetailConditionedUNet(StructureConditionedUNet):
    def __init__(self,base,width=16):
        super().__init__(base,width)
        channels=(base.input_conv.out_channels,base.down2_res2.conv2.out_channels,
                  base.down3_res2.conv2.out_channels)
        self.adapter=MatchedDetailAdapter(channels,width)


@torch.inference_mode()
def masked_sample(conditioner,model,scheduler,images,mfg,fs,echo,seed,timestep,guidance,mask):
    """Ancestral CFG with source-consistent exterior; one coupled forward path.

    Inputs/images are [-1,1]. mask=1 edits, mask=0 retains the source path.
    The exterior forward Markov path has marginal q(x_t|source) matching the
    historical initial q_sample. Its own RNG cannot change reverse noise draws.
    Noised exterior is projected after each reverse step; final exterior is
    exactly source in [-1,1] before the historical output conversion.
    """
    if images.ndim!=4 or images.shape[1:]!=(1,128,128) or mask.shape!=images.shape:
        raise ValueError('Expected image and mask [B,1,128,128]')
    if not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
        raise ValueError('Expected finite [-1,1] source')
    if not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Mask must be binary; no label- or target-derived resizing')
    if mask.device!=images.device or mask.dtype!=images.dtype:
        raise ValueError('Mask device/dtype must match image')
    if not 0 <= timestep < scheduler.num_train_timesteps:
        raise ValueError('Invalid timestep')
    conditioner.eval();model.eval()
    device=images.device
    generator=torch.Generator(device=device).manual_seed(seed)
    path_generator=torch.Generator(device=device).manual_seed(seed+719003)
    shape=images.shape
    noise=torch.randn(shape,device=device,generator=generator)
    # Project outside along a coherent Markov path, while preserving the exact
    # historical initial noisy image/noise stream inside the mask. The forward
    # bridge is sampled backwards conditional on that initial x_t and source.
    ts=torch.full((len(images),),timestep,device=device,dtype=torch.long)
    x=scheduler.q_sample(images,ts,noise)
    exterior={timestep:x.clone()}
    for t in range(timestep,0,-1):
        beta,alpha,abar=scheduler.betas[t],scheduler.alphas[t],scheduler.alphas_cumprod[t]
        previous_abar=scheduler.alphas_cumprod[t-1]
        mean=(torch.sqrt(previous_abar)*beta/(1-abar)*images +
              torch.sqrt(alpha)*(1-previous_abar)/(1-abar)*exterior[t])
        variance=beta*(1-previous_abar)/(1-abar)
        exterior[t-1]=mean+torch.sqrt(variance)*torch.randn(shape,device=device,generator=path_generator)
    ce=conditioner(mfg,fs,echo)
    ue=conditioner(mfg,fs,echo,force_unconditional=True)
    for t in range(timestep,-1,-1):
        ts=torch.full((len(images),),t,device=device,dtype=torch.long)
        uncond=model(x,ts,ue)
        cond=model(x,ts,ce)
        eps=uncond+guidance*(cond-uncond)
        beta,alpha,abar=scheduler.betas[t],scheduler.alphas[t],scheduler.alphas_cumprod[t]
        x=torch.rsqrt(alpha)*(x-beta/torch.sqrt(1-abar)*eps)
        if t:
            x=x+torch.sqrt(beta)*torch.randn(shape,device=device,generator=generator)
        x=torch.where(mask.bool(),x,exterior[t-1] if t else images)
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite masked sampling')
    return ((x.clamp(-1,1)+1)/2).cpu().numpy().astype('float32')
