"""Run the frozen verification library with the same CUDA math policy as grading."""
import argparse

import torch

from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import json_write
from scripts.v8_generator_dev.verify_full import verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('prepared', 'first', 'resumed', 'direct', 'run-id'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--evaluation', action='append', default=[])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    deterministic_runtime()
    output = verify(args.prepared, args.first, args.resumed, args.direct,
                    args.evaluation, args.run_id, args.device)
    json_write(output / 'runtime.json', dict(
        torch=str(torch.__version__), cpu_threads=torch.get_num_threads(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        reason='Match the training/evaluation runtime when reproducing CUDA probabilities',
    ))
    (output / 'RUNTIME_README.md').write_text(
        'Producer entrypoint: `python -m scripts.v8_generator_dev.verify_full_runtime`.\n'
        'It configures the same deterministic/TF32/thread settings as full_train/evaluate_full, '
        'calls the frozen verify_full library, and records those settings in runtime.json. '
        'The main README maps the underlying test/artifact verification files.\n', encoding='utf-8')
