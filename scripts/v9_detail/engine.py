"""Bounded engineering trainer for six-input matched guide candidates.

This version cannot run efficacy training. Full training requires a separately
frozen protocol, replicas, decision criteria and execution package.
"""
import hashlib
import json

import torch

from scripts.v6_structure.model import photometric_view
from scripts.v8_generator_dev.adapter_pilot import draw,independent_condition,objective,state_digest,pilot_settings
from scripts.v8_generator_dev.data import sampling_probabilities
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet,multiscale_guide
from scripts.v8_generator_dev.full_train import load_ema
from scripts.v8_generator_dev.io import digest,project_path
from scripts.v8_generator_dev.model import cpu_state
from scripts.v8_generator_dev.precision_step import precision_step
from scripts.v8_generator_dev.soft_structure import soft_structure

MODES=('coarse_control','coarse_fine','learned_zero')


class EngineeringTrainer:
    def __init__(self,parent,dataset,mode,seed=583201,batch_size=2,device='cpu'):
        if mode not in MODES or not 1<=batch_size<=2 or dataset.split!='train':
            raise ValueError('Engineering trainer requires a declared mode, training split and batch <=2')
        self.parent=project_path(parent)
        self.parent_hash=digest(self.parent)
        self.mode,self.seed,self.batch_size,self.device=mode,int(seed),batch_size,torch.device(device)
        self.dataset=dataset
        self.c,self.base,self.scheduler,payload=load_ema(self.parent,device)
        self.config,self.backbone=payload['config'],payload['variant']
        del payload
        self.c.eval().requires_grad_(False)
        torch.manual_seed(seed)
        self.model=DetailConditionedUNet(self.base,width=16).to(device).train()
        self.weights=torch.from_numpy(sampling_probabilities(dataset.frame,self.config['data']['minimum_cell_patients']))
        self.settings=pilot_settings()
        self.settings.update(seed=int(seed),batch_size=batch_size,guide='matched_six_input_detail_v1')
        self.optimizer=torch.optim.AdamW(self.model.adapter.parameters(),lr=self.settings['learning_rate'],
                                        weight_decay=self.settings['weight_decay'])
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.device.type=='cuda')
        self.frozen=dict(base=state_digest(self.base),conditioner=state_digest(self.c))
        self.initial_adapter=state_digest(self.model.adapter)
        self.step=0

    def batch(self):
        if self.step>=2:
            raise ValueError('This engine is capped at two discarded engineering updates per copy')
        items,clean,ts,noise,dropped,record=draw(self.dataset,self.weights,self.step,self.batch_size,self.seed)
        clean,ts,noise=[x.to(self.device) for x in (clean,ts,noise)]
        image=(clean+1)/2
        rng=torch.Generator(device=self.device).manual_seed(self.seed+4000000+self.step)
        with torch.no_grad():
            guide=multiscale_guide(photometric_view(image,rng),self.mode)
            true_structure=soft_structure(image)
        condition=independent_condition(self.c,items,dropped,self.device)
        xt=self.scheduler.q_sample(clean,ts,noise)
        self.model.train()
        pieces={}
        def forward():
            with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                prediction=self.model(xt,ts,condition,guide)
            total,mse,structure=objective(prediction,noise,xt,ts,self.scheduler,true_structure,self.settings)
            pieces.update(noise_mse=float(mse.detach()),coarse_structure_loss=float(structure.detach()))
            return total
        loss,norm,precision=precision_step(forward,list(self.model.adapter.parameters()),self.optimizer,
            self.scaler,self.device,self.settings['gradient_clip'],self.settings['max_same_batch_precision_retries'])
        self.step+=1
        if any(p.grad is not None for p in [*self.base.parameters(),*self.c.parameters()]):
            raise AssertionError('Frozen parent received gradients')
        if state_digest(self.base)!=self.frozen['base'] or state_digest(self.c)!=self.frozen['conditioner']:
            raise AssertionError('Frozen parent weights changed')
        return dict(record,mode=self.mode,backbone=self.backbone,loss=float(loss.detach()),
                    norm=float(norm),**pieces,**precision,engineering=True)

    def save(self,path):
        with path.open('xb') as handle:
            torch.save(dict(format='matched_detail_engineering_v1',engineering=True,
                parent_sha256=self.parent_hash,parent=str(self.parent),backbone=self.backbone,
                mode=self.mode,seed=self.seed,batch_size=self.batch_size,settings=self.settings,step=self.step,
                adapter=cpu_state(self.model.adapter),optimizer=self.optimizer.state_dict(),
                scaler=self.scaler.state_dict(),frozen=self.frozen,
                cpu_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None),handle)

    def restore(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        required=dict(format='matched_detail_engineering_v1',engineering=True,parent_sha256=self.parent_hash,
            backbone=self.backbone,mode=self.mode,seed=self.seed,batch_size=self.batch_size,settings=self.settings,
            frozen=self.frozen)
        if any(ck.get(k)!=v for k,v in required.items()):
            raise ValueError('Strict engineering checkpoint/protocol identity mismatch')
        if type(ck['step']) is not int or not 0<=ck['step']<=2:
            raise ValueError('Invalid engineering cursor')
        self.model.adapter.load_state_dict(ck['adapter'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.scaler.load_state_dict(ck['scaler'])
        self.step=ck['step']
        torch.random.set_rng_state(ck['cpu_rng'])
        if self.device.type=='cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
