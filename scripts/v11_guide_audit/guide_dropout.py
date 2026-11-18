"""Engineering-only source-guide dropout candidate; no existing route is changed."""
import torch
from torch import nn

STATE_NAMES=('full','coarse_only','second_block_only','zero_input','absent_adapter')
STATE_PROBABILITIES=(.4,.2,.2,.1,.1)


def draw_states(batch,seed,step,device='cpu'):
    if batch<1 or step<0:raise ValueError('Invalid gate draw')
    generator=torch.Generator(device=device).manual_seed(int(seed)+5000000+int(step))
    return torch.multinomial(torch.tensor(STATE_PROBABILITIES,device=device),batch,replacement=True,generator=generator)


def apply_states(guide,states):
    if guide.ndim!=4 or guide.shape[1:]!=(6,64,64) or not torch.isfinite(guide).all():
        raise ValueError('Six-channel finite source guide required')
    if states.shape!=(len(guide),) or states.dtype!=torch.long or states.device!=guide.device:
        raise ValueError('Matching integer gate state required')
    if states.min()<0 or states.max()>4:raise ValueError('Unknown gate state')
    first=~((states==2)|(states==3));second=~((states==1)|(states==3))
    gated=torch.cat([guide[:,:3]*first[:,None,None,None],guide[:,3:]*second[:,None,None,None]],dim=1)
    present=(states!=4).to(guide.dtype)
    return gated,present


class EngineeringGate(nn.Module):
    """Wrap a fresh adapter for capped integration only; no checkpoint/resume API."""
    def __init__(self,inner,seed,force_coverage=False):
        super().__init__();self.inner=inner;self.seed=seed;self.step=0;self.force_coverage=force_coverage;self.last_states=None

    @property
    def adapter(self):return self.inner.adapter

    def forward(self,x,t,condition,guide,present=None):
        if present is not None:raise ValueError('Engineering wrapper owns the presence gate')
        states=((torch.arange(len(guide),device=guide.device)+self.step)%5 if self.force_coverage else
                draw_states(len(guide),self.seed,self.step,guide.device))
        self.last_states=states.detach().cpu().tolist()
        gated,presence=apply_states(guide,states)
        return self.inner(x,t,condition,gated,presence)
