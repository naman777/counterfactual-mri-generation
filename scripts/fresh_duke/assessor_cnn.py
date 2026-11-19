"""Train the original four-block CNN with the independent judge recipe."""

import argparse
import time
from pathlib import Path

import torch

from scripts.fresh_duke import replication as r
from scripts.v4_heldout import train_judge
from src.models.predictor import AcquisitionPredictor


REPORT = r.ROOT / 'reports/v26_assessor_controls'
PACK = r.HELDOUT.parent / 'b2_20261016_01/inputs/heldout_train_val_float32.npz'
CHECKPOINT_STEM = 'judge_heldout_smallcnn'
CNN = 'heldout_robust_smallcnn_seed0'


class RobustCNN(AcquisitionPredictor):
    """Original predictor architecture with robust-judge scaling and two supervised heads."""

    def forward(self, x):
        output = super().forward((x - .5) / .25)
        return {'mfg': output['manufacturer_logits'], 'fs': output['field_strength_logits']}


def train(out):
    resnet = next(row for row in r.load_json(r.HELDOUT / 'frozen_checkpoints.json')
                  if row['family'] == 'orientation_robust' and row['seed'] == 0)
    reference = r.load_json(r.HELDOUT / resnet['training_json'])
    if not reference['complete'] or reference['epochs'] != 40 or reference['pack_sha256'] != r.digest(PACK):
        raise ValueError('Robust ResNet reference and CNN training pack differ')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'seed': 0, 'epochs': 40, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'resnet_reference': resnet,
        'architecture': 'Original four-block predictor and its shared layer with two supervised classification heads',
        'recipe': 'Frozen held-out ResNet-18 recipe including input scaling, augmentations, loss and selection',
        'inputs': {r.rel(path): r.digest(path) for path in
                   (PACK, PACK.parent / 'summary.json', r.HELDOUT / resnet['training_json'],
                    Path(__file__), Path(train_judge.__file__), r.ROOT / 'src/models/predictor.py')},
    }
    r.json_write(out / 'protocol.json', manifest)
    started = time.perf_counter()
    result = train_judge.train(PACK, out / 'training', 0, 40, 'cuda',
                               model_factory=RobustCNN, checkpoint_stem=CHECKPOINT_STEM)
    if not result['complete']:
        raise ValueError('Incomplete CNN training')
    r.json_write(out / 'training_receipt.json', {'seconds': time.perf_counter() - started,
                                                'checkpoint_sha256': result['checkpoint_sha256'],
                                                'selected_epoch': result['best_epoch']})
    (out / 'README.md').write_text(
        '# Held-out robust four-block CNN\n\n'
        'Run `python -m scripts.fresh_duke.assessor_cnn --out <new-folder>`. '
        'protocol.json and training_receipt.json are produced by assessor_cnn.py. '
        'The training subfolder is produced by scripts/v4_heldout/train_judge.py. '
        'Only the existing independent training and validation pack is used.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPORT / 'cnn_seed0_20261119_01')
    train(parser.parse_args().out)
