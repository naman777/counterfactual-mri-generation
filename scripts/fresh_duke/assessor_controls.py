"""Compare both replay controls and the robust CNN on frozen fresh field-strength edits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fresh_duke import replication as r
from scripts.fresh_duke.assessor_cnn import CHECKPOINT_STEM, CNN, PACK, REPORT, RobustCNN
from scripts.fresh_duke.image_diagnostics import RUN, check_digest
from scripts.fresh_duke.paired_stats import PRIMARY_JUDGES, patient_estimate
from scripts.fresh_duke.verify_replication import read_images
from scripts.v2_quickwins.judge_model import judge_probs


KEYS = ['slice_id', 'patient_id', 'mfg', 'fs']
EPOCHS = (10, 110)


def check(path, expected, inputs):
    check_digest(path, expected)
    resolved = path.resolve()
    key = r.rel(resolved) if resolved.is_relative_to(r.ROOT) else resolved.as_posix()
    inputs[key] = expected


def frozen_inputs():
    receipt = r.load_json(RUN / 'complete.json')
    inputs = {}
    check(RUN / 'protocol.json', receipt['protocol_sha256'], inputs)
    protocol = r.load_json(RUN / 'protocol.json')
    check(RUN / 'inputs.csv', protocol['input_sha256'], inputs)
    cohort = pd.read_csv(RUN / 'inputs.csv')
    with np.load(PACK, allow_pickle=False) as pack:
        if set(cohort.patient_id) & (set(pack['train_patient']) | set(pack['val_patient'])):
            raise ValueError('Fresh patients overlap the independent training or selection pack')
    if not cohort.slice_id.is_unique or not cohort.groupby('patient_id').size().eq(5).all():
        raise ValueError('Invalid fresh identifier or patient coverage')
    real_path = RUN / 'real/per_slice.csv'
    check(real_path, r.load_json(RUN / 'real/complete.json')['per_slice_sha256'], inputs)
    real = pd.read_csv(real_path).query("axis == 'field_strength' and judge in @PRIMARY_JUDGES")
    real = real[KEYS + ['judge', 'orientation', 'correct']]
    phases = {}
    for epoch in EPOCHS:
        name = f'epoch{epoch}_field_strength'
        folder = RUN / name
        complete = receipt['phases'][name]
        if (complete != r.load_json(folder / 'complete.json') or complete['status'] != 'complete'
                or complete['failed_arm_slices'] or complete['paired_target_slices'] != len(cohort)):
            raise ValueError(f'Incomplete archived phase: {name}')
        path = folder / 'per_slice.csv'
        check(path, complete['per_slice_sha256'], inputs)
        saved = pd.read_csv(path).query('judge in @PRIMARY_JUDGES')
        np.testing.assert_array_equal(saved.hit, saved.prediction.eq(1 - saved.fs))
        phases[epoch] = (folder, complete, saved[KEYS + ['judge', 'arm', 'hit']])
    return cohort, real, phases, inputs


def evaluate_cnn(run, cohort, phases, inputs):
    manifest = r.load_json(run / 'protocol.json')
    receipt = r.load_json(run / 'training_receipt.json')
    training = r.load_json(run / 'training/train.json')
    if (not training['complete'] or training['seed'] != 0 or training['epochs_completed'] != 40
            or training['checkpoint_sha256'] != receipt['checkpoint_sha256']
            or training['pack_sha256'] != manifest['inputs'][r.rel(PACK)]):
        raise ValueError('Incomplete or mismatched CNN training receipt')
    for path, sha in manifest['inputs'].items():
        check(r.ROOT / path, sha, inputs)
    checkpoint = run / 'training' / f'{CHECKPOINT_STEM}_seed0.pt'
    check(checkpoint, training['checkpoint_sha256'], inputs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustCNN().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    native = np.stack([r.image_array(r.ROOT / row.image_path, row.image_sha256) for row in cohort.itertuples()])
    rows = []

    def predict(images, arm, epoch=None, orientation=None):
        _, prob = judge_probs(model, images, device, bs=64)
        decision = prob.argmax(axis=1)
        return cohort[KEYS].assign(judge=CNN, arm=arm, epoch=epoch, orientation=orientation,
                                   p0=prob[:, 0], p1=prob[:, 1], prediction=decision,
                                   hit=decision == 1 - cohort.fs, correct=decision == cohort.fs)

    rows.append(predict(native, 'real', orientation='native'))
    rows.append(predict(np.ascontiguousarray(native[:, ::-1]), 'real', orientation='vertical_flip'))
    for epoch, (folder, receipt, _) in phases.items():
        check(folder / 'generation.jsonl', receipt['generation_log_sha256'], inputs)
        images, _ = read_images(folder, {}, len(cohort))
        if images.keys() != set(r.GENERATED):
            raise ValueError(f'Missing CNN evaluation arms at epoch {epoch}')
        for arm, array in images.items():
            rows.append(predict(array, arm, epoch=epoch))
    return pd.concat(rows, ignore_index=True)


def subsets(table):
    return [('pooled', table), *table.groupby('fs')]


def summaries(cohort, real, phases, cnn):
    controls, gaps, accuracy = [], [], []
    judges = list(PRIMARY_JUDGES)
    if cnn is not None:
        judges.append(CNN)
        real = pd.concat([real, cnn[cnn.arm.eq('real')][real.columns]], ignore_index=True)
    for judge in judges:
        wide = real[real.judge.eq(judge)].pivot(index=KEYS, columns='orientation', values='correct').reset_index()
        check_coverage(cohort, wide, ['native', 'vertical_flip'])
        wide['flip_minus_native'] = wide.vertical_flip.astype(int) - wide.native.astype(int)
        for metric in ('native', 'vertical_flip', 'flip_minus_native'):
            accuracy.append(dict(judge=judge, metric=metric, **patient_estimate(wide, metric, True)))
    for epoch, (_, _, saved) in phases.items():
        if cnn is not None:
            saved = pd.concat([saved, cnn[cnn.epoch.eq(epoch)][saved.columns]], ignore_index=True)
        paired = saved[saved.arm.isin(r.GENERATED)].pivot(index=KEYS, columns=['judge', 'arm'], values='hit')
        paired.columns = [f'{judge}:{arm}' for judge, arm in paired.columns]
        paired = paired.reset_index()
        check_coverage(cohort, paired, [f'{judge}:{arm}' for judge in judges for arm in r.GENERATED])
        for direction, group in subsets(paired):
            for judge in judges:
                target = group[f'{judge}:target'].astype(int)
                for arm in r.GENERATED:
                    contrast = group[f'{judge}:{arm}'].astype(int)
                    row = group[KEYS].assign(hit=contrast, target_minus_control=target - contrast)
                    for metric in ('hit', 'target_minus_control') if arm != 'target' else ('hit',):
                        controls.append(dict(epoch=epoch, source_label=direction, judge=judge, arm=arm,
                                             metric=metric, **patient_estimate(row, metric, True)))
            pairs = [PRIMARY_JUDGES]
            if cnn is not None:
                pairs += [(PRIMARY_JUDGES[0], CNN), (CNN, PRIMARY_JUDGES[1])]
            for left, right in pairs:
                for arm in r.GENERATED:
                    estimate = group[f'{left}:{arm}'].astype(int) - group[f'{right}:{arm}'].astype(int)
                    row = group[KEYS].assign(gap=estimate)
                    gaps.append(dict(epoch=epoch, source_label=direction, arm=arm, left=left, right=right,
                                     **patient_estimate(row, 'gap', True)))
    return (pd.DataFrame(table) for table in (controls, gaps, accuracy))


def check_coverage(cohort, table, measures):
    expected = cohort[KEYS].set_index(KEYS).sort_index()
    actual = table.set_index(KEYS).sort_index()
    if not expected.index.equals(actual.index) or actual[measures].isna().any().any():
        raise ValueError('Missing or mismatched paired slice coverage')


def main(out, cnn_run):
    cohort, real, phases, inputs = frozen_inputs()
    cnn = evaluate_cnn(cnn_run, cohort, phases, inputs) if cnn_run is not None else None
    control, gaps, accuracy = summaries(cohort, real, phases, cnn)
    out.mkdir(parents=True, exist_ok=False)
    r.save_csv(out / 'control_contrasts.csv', control)
    r.save_csv(out / 'assessor_pairs.csv', gaps)
    r.save_csv(out / 'native_flip.csv', accuracy)
    if cnn is not None:
        r.save_csv(out / 'cnn_predictions.csv', cnn)
    r.json_write(out / 'inputs.json', {'inputs': inputs, 'script_sha256': r.digest(Path(__file__)),
                                     'cnn_run': str(cnn_run) if cnn_run is not None else None})
    (out / 'README.md').write_text(
        '# Fresh field-strength replay and assessor comparisons\n\n'
        'All files are generated by `python -m scripts.fresh_duke.assessor_controls --out <new-folder>` '
        '(pass `--cnn-run <training-folder>` to include the robust CNN). '
        'The positive class in the replay evaluation is the requested opposite field strength.\n\n'
        '- `control_contrasts.csv` gives target-hit rates and paired target-minus-control differences.\n'
        '- `assessor_pairs.csv` gives paired signed hit-rate gaps (left minus right), including the original-minus-CNN '
        'change and CNN-minus-ResNet residual when the CNN is available.\n'
        '- `native_flip.csv` gives field-strength accuracy and the paired flip-minus-native difference.\n'
        '- `cnn_predictions.csv` records the CNN probabilities and decisions, when run.\n'
        '- `inputs.json` records the verified input and script hashes.\n\n'
        'The intervals resample whole patients within fixed scanner cells. Accuracy on real acquisitions and '
        'agreement on edits address different outcomes. The single-seed CNN comparison is descriptive.\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cnn-run', type=Path)
    args = parser.parse_args()
    main(args.out, args.cnn_run)
