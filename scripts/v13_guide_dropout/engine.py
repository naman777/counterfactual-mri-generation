"""Same objective/draw engine with explicit guide-state policy and strict checkpointing."""
import copy
import hashlib

import torch
from torch import nn

from scripts.v10_detail.engine import Trainer as ReferenceTrainer
from scripts.v10_detail.protocol import canonical_hash
from scripts.v11_guide_audit.guide_dropout import apply_states, draw_states
from scripts.v13_guide_dropout.protocol import validate, CHECKPOINT_FORMAT, MODES


class GuideDropoutModel(nn.Module):
    def __init__(self,inner,seed):
        super().__init__()
        self.inner,self.seed,self.step = inner,int(seed),0
        self.last_states = None

    @property
    def adapter(self):
        return self.inner.adapter

    def forward(self,x,t,condition,guide,present=None):
        if present is not None:
            raise ValueError('Training dropout owns the presence gate')
        states = draw_states(len(guide),self.seed,self.step,guide.device)
        self.last_states = states.detach().cpu().tolist()
        gated,presence = apply_states(guide,states)
        return self.inner(x,t,condition,gated,presence)


def finite_nested(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite_nested(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_nested(v) for v in value)
    if isinstance(value,float):
        import math
        return math.isfinite(value)
    return True


class Trainer(ReferenceTrainer):
    def __init__(self,protocol,dataset,mode,seed,device='cpu',engineering=False,batch_override=None):
        validate(protocol)
        if mode not in MODES:
            raise ValueError('Only the two proposed guided arms are trained')
        if not engineering and batch_override is not None:
            raise ValueError('Full exposure cannot change the batch size')
        super().__init__(protocol['reference_protocol'],dataset,mode,seed,device,engineering,batch_override)
        self.protocol = copy.deepcopy(protocol)
        self.protocol_hash = canonical_hash(protocol)
        self.model = GuideDropoutModel(self.model,seed)
        self.gate_counts = [0]*5
        self.draw_chain = hashlib.sha256(b'guide_dropout_actual_draw_chain_v1').hexdigest()
        wanted = {id(p) for p in self.model.adapter.parameters()}
        actual = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if wanted != actual:
            raise ValueError('Wrapped model lost optimiser parameter identity')

    def batch(self):
        self.model.step = self.step
        row = super().batch()
        row['gate_states'] = self.model.last_states
        row['gate_policy_sha256'] = canonical_hash(self.protocol['guide_dropout'])
        draw = {k:row[k] for k in ('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states')}
        self.draw_chain = canonical_hash(dict(previous=self.draw_chain,draw=draw))
        row['actual_draw_chain_sha256'] = self.draw_chain
        for state in row['gate_states']:
            self.gate_counts[state] += 1
        return row

    def identity(self):
        return dict(format=CHECKPOINT_FORMAT,protocol=self.protocol,protocol_sha256=self.protocol_hash,
                    parent_sha256=self.protocol['reference_protocol']['parent_sha256'],mode=self.mode,
                    training_seed=self.seed,engineering=self.engineering,total_steps=self.total_steps,
                    batch_size=self.batch_size,data_sha256=self.data_hash,runtime=self.runtime,frozen=self.frozen)

    def save(self,path):
        self.frozen_check()
        if not finite_nested(self.model.adapter.state_dict()) or not finite_nested(self.optimizer.state_dict()):
            raise ValueError('Nonfinite checkpoint refused')
        payload = dict(self.identity(),step=self.step,gate_counts=self.gate_counts,actual_draw_chain_sha256=self.draw_chain,
                       adapter={k:v.detach().cpu().clone() for k,v in self.model.adapter.state_dict().items()},
                       optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),
                       rng_cpu=torch.random.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])
        with path.open('xb') as handle:
            torch.save(payload,handle)

    def restore(self,path):
        ck = torch.load(path,map_location='cpu',weights_only=False)
        expected = self.identity()
        if not isinstance(ck,dict) or any(ck.get(k) != v for k,v in expected.items()):
            raise ValueError('Strict policy/science/data/runtime checkpoint mismatch')
        required = {'step','gate_counts','actual_draw_chain_sha256','adapter','optimizer','scaler','rng_cpu','rng_cuda'}
        if set(ck) != set(expected)|required:
            raise ValueError('Checkpoint fields missing or unexpected')
        cap = min(2,self.total_steps) if self.engineering else self.total_steps
        if type(ck['step']) is not int or not 0 <= ck['step'] <= cap:
            raise ValueError('Invalid applied-update cursor')
        counts = ck['gate_counts']
        if not isinstance(counts,list) or len(counts)!=5 or any(type(v) is not int or v<0 for v in counts):
            raise ValueError('Invalid gate exposure ledger')
        expected_samples = sum(min(self.batch_size,len(self.dataset)-(i % ((len(self.dataset)+self.batch_size-1)//self.batch_size))*self.batch_size)
                               for i in range(ck['step']))
        if sum(counts) != expected_samples:
            raise ValueError('Gate exposure disagrees with applied cursor')
        chain = ck['actual_draw_chain_sha256']
        if not isinstance(chain,str) or len(chain)!=64 or any(c not in '0123456789abcdef' for c in chain):
            raise ValueError('Invalid draw-chain identity')
        if not finite_nested(ck):
            raise ValueError('Nonfinite saved state refused')
        current = self.model.adapter.state_dict()
        if set(ck['adapter']) != set(current) or any(ck['adapter'][k].shape != v.shape for k,v in current.items()):
            raise ValueError('Adapter state schema mismatch')
        self.model.adapter.load_state_dict(ck['adapter'],strict=True)
        self.optimizer.load_state_dict(ck['optimizer'])
        self.scaler.load_state_dict(ck['scaler'])
        torch.random.set_rng_state(ck['rng_cpu'])
        if self.device.type=='cuda':
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        self.step,self.gate_counts,self.draw_chain = ck['step'],list(counts),chain
        self.model.step = self.step
