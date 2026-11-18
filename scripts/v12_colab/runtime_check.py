"""Disposable synthetic adapter updates on the existing project architecture."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from pathlib import Path

import torch

from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet, multiscale_guide
from scripts.v8_generator_dev.precision_step import precision_step
from scripts.v8_generator_dev.soft_structure import soft_structure
from scripts.v11_guide_audit.guide_dropout import EngineeringGate
from scripts.v12_colab.objective_snapshot import objective

ROOT = Path(__file__).resolve().parents[2]


def digest_module(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_bundle():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text())
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundle identity failed: {name}')
    return hashlib.sha256((ROOT / 'bundle_manifest.json').read_bytes()).hexdigest()


def run(output, batch_size=32, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'README.md').write_text(
        'Producer: scripts.v12_colab.runtime_check. environment.json: actual runtime; '
        'journal.jsonl: every disposable synthetic update; result.json: pass/failure, '
        'frozen-weight and memory checks. Randomly initialised parent and synthetic pixels only. '
        'No MRI data, trained weights, efficacy checkpoint, resume test or scientific result.\n')
    result = dict(complete=False,engineering_only=True,synthetic_only=True,
                  trained_parent_loaded=False,efficacy_established=False,
                  full_training_ready=False,checkpoints_saved=False)
    started = time.monotonic()
    try:
        result['bundle_manifest_sha256'] = verify_bundle()
        cfg = json.loads((ROOT / 'compatibility_config.json').read_text())
        if not 1 <= batch_size <= 32:
            raise ValueError('Batch size must be between 1 and 32')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('GPU runtime required; no silent CPU fallback')
        if device == 'cuda' and batch_size != 32:
            result['intended_batch_32_checked'] = False
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        packages = {}
        for name in ('torch','torchvision','numpy','pandas'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        environment = dict(python=platform.python_version(),packages=packages,
                           torch_cuda=torch.version.cuda,device=device,batch_size=batch_size,
                           gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                           gpu_bytes=torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else None,
                           cudnn=torch.backends.cudnn.version(),tf32=False,amp=device == 'cuda')
        (output / 'environment.json').write_text(json.dumps(environment,indent=2))
        reference = []
        results = []
        with (output / 'journal.jsonl').open('x') as journal:
            for mode in ('coarse_control','coarse_fine'):
                torch.manual_seed(583301)
                conditioner, base, scheduler = build_models(cfg['model_config'],'additive',device)
                conditioner.eval().requires_grad_(False)
                model = EngineeringGate(DetailConditionedUNet(base,16),583301,force_coverage=True).to(device).train()
                frozen = (digest_module(base),digest_module(conditioner))
                adapter_before = digest_module(model.adapter)
                optimizer = torch.optim.AdamW(model.adapter.parameters(),lr=1e-4,weight_decay=.01)
                scaler = torch.amp.GradScaler('cuda',enabled=device == 'cuda')
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                draws = []
                for step in range(2):
                    if time.monotonic()-started > 480:
                        raise TimeoutError('Compatibility check time cap reached')
                    rng = torch.Generator().manual_seed(583301+step)
                    images = torch.rand(batch_size,1,128,128,generator=rng).to(device)
                    noise = torch.randn(batch_size,1,128,128,generator=rng).to(device)
                    # Exercise the structure-loss branch and the high-timestep branch.
                    ts = (torch.arange(batch_size,device=device)%2)*700+100
                    with torch.no_grad():
                        labels = torch.arange(batch_size,device=device)%2
                        echo = torch.full((batch_size,),.5,device=device)
                        condition = conditioner(labels,labels,echo)
                        guide = multiscale_guide(images,mode)
                        truth = soft_structure(images)
                    xt = scheduler.q_sample(images*2-1,ts,noise)
                    model.step = step
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    before = time.monotonic()

                    def loss_fn():
                        with torch.autocast(device,dtype=torch.float16,enabled=device == 'cuda'):
                            prediction = model(xt,ts,condition,guide)
                        return objective(prediction,noise,xt,ts,scheduler,truth,cfg['objective_settings'])[0]

                    loss,norm,precision = precision_step(loss_fn,model.adapter.parameters(),optimizer,
                                                        scaler,device,1.,8)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    row = dict(mode=mode,step=step+1,batch_size=batch_size,
                               loss=float(loss.detach()),gradient_norm=float(norm),
                               seconds=time.monotonic()-before,gate_states=model.last_states,
                               noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                               **precision)
                    if not precision['applied'] or not torch.isfinite(norm):
                        raise ValueError('Finite applied update required')
                    if batch_size >= 5 and set(model.last_states) != set(range(5)):
                        raise ValueError('All five states were not exercised')
                    if any(p.grad is not None for p in [*base.parameters(),*conditioner.parameters()]):
                        raise ValueError('Frozen parameter received a gradient')
                    draws.append((row['noise_sha256'],row['gate_states']))
                    journal.write(json.dumps(row)+'\n');journal.flush()
                    print(json.dumps(row),flush=True)
                    del loss,xt,images,noise,guide,truth,condition
                if reference and reference != draws:
                    raise ValueError('Matched synthetic draws changed between modes')
                reference = draws
                if frozen != (digest_module(base),digest_module(conditioner)):
                    raise ValueError('Frozen parent/conditioner changed')
                if adapter_before == digest_module(model.adapter):
                    raise ValueError('Adapter did not change')
                results.append(dict(mode=mode,frozen_weights=True,adapter_changed=True,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None))
                del model,base,conditioner,scheduler,optimizer,scaler,loss_fn
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        result.update(complete=True,copies=results,actual_updates=4,matched_synthetic_draws=True,
                      intended_batch_32_checked=batch_size == 32 and device == 'cuda')
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['seconds'] = time.monotonic()-started
        with (output / 'result.json').open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    run(args.output,args.batch_size,args.device)
