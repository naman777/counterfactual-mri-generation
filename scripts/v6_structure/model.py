"""Frozen DDPM with a zero-initialised, spatial source-structure adapter.

The guide is a low-resolution contrast-normalised gradient proxy. It is not
an expert annotation, an invariant anatomical representation, or a guarantee
of pathology preservation. Raw clean images are never passed to the adapter.
"""
import torch
from torch import nn
import torch.nn.functional as F
from scripts.v5_anatomy.anchored_sampler import lowpass


def structure_maps(images, size=32):
    """[B,1,128,128] in [0,1] -> [B,3,32,32] bounded coarse gradients.

    Removes raw intensity and full-resolution texture. Normalisation reduces
    positive affine intensity sensitivity, but orientation/scanner leakage must
    still be measured. Also differentiable for a proxy consistency objective.
    """
    if images.ndim != 4 or images.shape[1:] != (1,128,128):
        raise ValueError('Expected [B,1,128,128]')
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Expected finite image intensities in [0,1]')
    if size != 32:
        raise ValueError('Version 1 uses a fixed 32x32 structure bottleneck')
    coarse = F.interpolate(lowpass(images.float(), 2.),size=(size,size),mode='bilinear',align_corners=False)
    gy, gx = torch.gradient(coarse, dim=(-2,-1))
    energy = gx.square()+gy.square()
    local = F.avg_pool2d(F.pad(energy,(2,2,2,2),mode='reflect'),5,stride=1)
    floor = energy.mean((-2,-1),keepdim=True)*.01+1e-10
    normaliser = torch.sqrt(local+floor)
    magnitude = (torch.sqrt(energy+floor)-torch.sqrt(floor))/normaliser
    return torch.cat([gx/normaliser,gy/normaliser,magnitude],1).clamp(-3,3)/3


def photometric_view(images, generator):
    """Geometry-preserving appearance perturbation for structure extraction only."""
    shape=(len(images),1,1,1)
    gamma=.7+.6*torch.rand(shape,device=images.device,generator=generator)
    gain=.8+.4*torch.rand(shape,device=images.device,generator=generator)
    offset=.04*(torch.rand(shape,device=images.device,generator=generator)-.5)
    return (images.clamp_min(1e-6).pow(gamma)*gain+offset).clamp(0,1)


class StructureAdapter(nn.Module):
    def __init__(self,base_channels=(64,128,256),width=16):
        super().__init__()
        self.blocks=nn.ModuleList([
            nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU()),
            nn.Sequential(nn.Conv2d(width,width*2,3,stride=2,padding=1),nn.SiLU()),
            nn.Sequential(nn.Conv2d(width*2,width*4,3,stride=2,padding=1),nn.SiLU())])
        self.outputs=nn.ModuleList([nn.Conv2d(ch,out,1) for ch,out in zip([width,width*2,width*4],base_channels)])
        for layer in self.outputs:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,structure):
        if structure.ndim != 4 or structure.shape[1:] != (3,32,32):
            raise ValueError('The adapter accepts only [B,3,32,32] proxy maps, not clean MRI pixels')
        h=F.interpolate(structure,size=(128,128),mode='bilinear',align_corners=False)
        controls=[]
        for block,output in zip(self.blocks,self.outputs):
            h=block(h);controls.append(output(h))
        return controls


class StructureConditionedUNet(nn.Module):
    def __init__(self,base,width=16):
        super().__init__()
        self.base=base.requires_grad_(False).eval()
        channels=(base.input_conv.out_channels,base.down2_res2.conv2.out_channels,base.down3_res2.conv2.out_channels)
        self.adapter=StructureAdapter(channels,width)

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()  # frozen base dropout must not drift during adaptation
        return self

    def forward(self,x,t,condition,structure=None,present=None):
        if structure is None:
            return self.base(x,t,condition)
        if len(structure)!=len(x) or not torch.isfinite(structure).all():
            raise ValueError('Invalid structure map batch')
        if present is None:
            present=torch.ones(len(x),device=x.device,dtype=x.dtype)
        if present.shape!=(len(x),) or not torch.isfinite(present).all() or present.min()<0 or present.max()>1:
            raise ValueError('Invalid source-presence gate')
        controls=[value*present[:,None,None,None] for value in self.adapter(structure)]
        b=self.base
        tc=torch.cat([b.time_embedding(t),condition],dim=1)
        h=b.input_conv(x)+controls[0]
        h=b.down1_res1(h,tc);h=b.down1_res2(h,tc);skip1=h
        h=b.downsample1(h);h=b.down2_res1(h,tc);h=b.down2_res2(h,tc)+controls[1];skip2=h
        h=b.downsample2(h);h=b.down3_res1(h,tc);h=b.down3_res2(h,tc)+controls[2]
        h=b.mid_res1(h,tc);h=b.mid_res2(h,tc);h=b.upsample2(h)
        if h.shape[-2:]!=skip2.shape[-2:]:h=F.interpolate(h,size=skip2.shape[-2:],mode='nearest')
        h=b.up2_res1(torch.cat([h,skip2],1),tc);h=b.up2_res2(h,tc);h=b.upsample1(h)
        if h.shape[-2:]!=skip1.shape[-2:]:h=F.interpolate(h,size=skip1.shape[-2:],mode='nearest')
        h=b.up1_res1(torch.cat([h,skip1],1),tc);h=b.up1_res2(h,tc)
        return b.output_conv(b.output_activation(b.output_norm(h)))


class BoundStructureModel(nn.Module):
    """Expose the historical sampler interface with an identical guide in both CFG arms."""
    def __init__(self,model,structure,present=None):
        super().__init__();self.model=model;self.structure=structure;self.present=present

    def forward(self,x,t,condition):
        n=len(self.structure)
        if len(x) not in [n,2*n]:
            raise ValueError('Unexpected sampler/CFG batch size')
        repeat=len(x)//n
        maps=self.structure.repeat(repeat,1,1,1)
        present=self.present.repeat(repeat) if self.present is not None else None
        return self.model(x,t,condition,maps,present)
