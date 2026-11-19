"""Full-exposure adapter engine with strict scientific/runtime/data resume identity."""
import math
import time

import torch

from scripts.v6_structure.model import photometric_view
from scripts.v8_generator_dev.adapter_pilot import draw,independent_condition,objective,state_digest
from scripts.v8_generator_dev.data import sampling_probabilities
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet,multiscale_guide
from scripts.v8_generator_dev.full_train import load_ema
from scripts.v8_generator_dev.io import digest,project_path
from scripts.v8_generator_dev.model import cpu_state
from scripts.v8_generator_dev.precision_step import precision_step
from scripts.v8_generator_dev.soft_structure import soft_structure
from scripts.v10_detail.protocol import validate,canonical_hash,MODES


class Trainer:
    def __init__(self,protocol,dataset,mode,seed,device='cpu',engineering=False,batch_override=None):
        validate(protocol)
        if dataset.split!='train' or mode not in MODES or seed not in protocol['training_seeds']:
            raise ValueError('Declared training source/copy required')
        if batch_override is not None and (not engineering or not 1<=batch_override<=32):
            raise ValueError('Batch overrides are engineering only')
        self.protocol,self.mode,self.seed=protocol,mode,seed
        self.protocol_hash=canonical_hash(protocol)
        self.dataset,self.device,self.engineering=dataset,torch.device(device),bool(engineering)
        self.settings=dict(protocol['training'])
        self.batch_size=batch_override or self.settings['batch_size']
        self.settings.update(seed=seed,batch_size=self.batch_size)
        parent=project_path(protocol['parent'])
        if digest(parent)!=protocol['parent_sha256']:raise ValueError('Frozen parent bytes changed')
        self.c,self.base,self.scheduler,payload=load_ema(parent,device)
        if payload['variant']!=protocol['backbone']:raise ValueError('Parent backbone mismatch')
        self.config=payload['config'];del payload
        self.c.eval().requires_grad_(False)
        torch.manual_seed(seed)
        self.model=DetailConditionedUNet(self.base,self.settings['adapter_width']).to(device).train()
        self.weights=torch.from_numpy(sampling_probabilities(dataset.frame,self.config['data']['minimum_cell_patients']))
        self.optimizer=torch.optim.AdamW(self.model.adapter.parameters(),lr=self.settings['learning_rate'],
                                        weight_decay=self.settings['weight_decay'])
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.device.type=='cuda')
        self.step=0
        self.total_steps=math.ceil(len(dataset)/self.batch_size)*self.settings['epochs']
        self.frozen=dict(base=state_digest(self.base),conditioner=state_digest(self.c))
        self.data_hash=canonical_hash(dataset.frame.to_csv(index=False))
        self.runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'cpu',
            batch_size=self.batch_size,amp=self.scaler.is_enabled())

    def batch(self):
        cap=min(2,self.total_steps) if self.engineering else self.total_steps
        if self.step>=cap:raise ValueError('Declared exposure exhausted')
        items,clean,ts,noise,dropped,record=draw(self.dataset,self.weights,self.step,self.batch_size,self.seed)
        clean,ts,noise=[x.to(self.device) for x in (clean,ts,noise)]
        with torch.no_grad():
            images=(clean+1)/2
            rng=torch.Generator(device=self.device).manual_seed(self.seed+4000000+self.step)
            guide=multiscale_guide(photometric_view(images,rng),self.mode)
            truth=soft_structure(images)
            condition=independent_condition(self.c,items,dropped,self.device)
        xt=self.scheduler.q_sample(clean,ts,noise)
        self.model.train();parts={};started=time.monotonic()
        def loss_fn():
            with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                prediction=self.model(xt,ts,condition,guide)
            total,mse,structure=objective(prediction,noise,xt,ts,self.scheduler,truth,self.settings)
            parts.update(noise_mse=float(mse.detach()),coarse_structure_loss=float(structure.detach()))
            return total
        loss,norm,precision=precision_step(loss_fn,self.model.adapter.parameters(),self.optimizer,self.scaler,
            self.device,self.settings['gradient_clip'],self.settings['max_same_batch_precision_retries'])
        self.step+=1
        if any(p.grad is not None for p in [*self.base.parameters(),*self.c.parameters()]):
            raise AssertionError('Frozen parameters received gradients')
        return dict(record,mode=self.mode,training_seed=self.seed,total_loss=float(loss.detach()),
            gradient_norm=float(norm),seconds=time.monotonic()-started,**parts,**precision,
            engineering=self.engineering,actual_batch_size=len(items),fixed_echo_time=True)

    def frozen_check(self):
        if self.frozen!={'base':state_digest(self.base),'conditioner':state_digest(self.c)}:
            raise AssertionError('Frozen parent or conditioner changed')

    def save(self,path):
        self.frozen_check()
        with path.open('xb') as handle:
            torch.save(dict(format='replicated_detail_checkpoint_v1',protocol=self.protocol,
                protocol_sha256=self.protocol_hash,parent_sha256=self.protocol['parent_sha256'],
                mode=self.mode,training_seed=self.seed,engineering=self.engineering,
                step=self.step,total_steps=self.total_steps,batch_size=self.batch_size,
                data_sha256=self.data_hash,runtime=self.runtime,frozen=self.frozen,
                adapter=cpu_state(self.model.adapter),optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),
                rng_cpu=torch.random.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []),handle)

    def restore(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        expected=dict(format='replicated_detail_checkpoint_v1',protocol=self.protocol,
            protocol_sha256=self.protocol_hash,parent_sha256=self.protocol['parent_sha256'],
            mode=self.mode,training_seed=self.seed,engineering=self.engineering,total_steps=self.total_steps,
            batch_size=self.batch_size,data_sha256=self.data_hash,runtime=self.runtime,frozen=self.frozen)
        if any(ck.get(k)!=v for k,v in expected.items()):raise ValueError('Strict checkpoint identity/runtime mismatch')
        cap=min(2,self.total_steps) if self.engineering else self.total_steps
        if type(ck['step']) is not int or not 0<=ck['step']<=cap:raise ValueError('Invalid cursor')
        self.model.adapter.load_state_dict(ck['adapter']);self.optimizer.load_state_dict(ck['optimizer'])
        self.scaler.load_state_dict(ck['scaler']);self.step=ck['step']
        torch.random.set_rng_state(ck['rng_cpu'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(ck['rng_cuda'])
