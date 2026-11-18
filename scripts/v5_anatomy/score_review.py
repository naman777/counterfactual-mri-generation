"""Score supplied reader labels/native-grid annotations, never proxy substitutes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.v4_convergence.core import digest, json_write
from scripts.v5_anatomy.metrics import mask_array, mask_metrics, landmark_metrics
from scripts.v5_anatomy.csv_io import read_table


def label(value):
    value = str(value).strip().lower()
    if value in ['', 'nan', 'none']:
        return None
    if value in ['yes', 'true', '1']:
        return True
    if value in ['no', 'false', '0']:
        return False
    raise ValueError('Reader labels must be yes/no or blank; never infer missing answers')


def score(pilot, output, review_csv=None, annotations=None):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    if not review_csv and not annotations:
        raise ValueError('Supply reader responses and/or actual annotations')
    complete = json.loads((pilot/'complete.json').read_text())
    if not complete['complete']:
        raise ValueError('Pilot is incomplete')
    mapping = read_table(pilot/'review/private_mapping.csv', dtype=str).set_index('review_id')
    readers, measured, inputs = [], [], {str(pilot/'complete.json'): digest(pilot/'complete.json')}
    if review_csv:
        review_csv = Path(review_csv).resolve()
        inputs[str(review_csv)] = digest(review_csv)
        frame = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
        if frame.duplicated(['review_id','reviewer']).any():
            raise ValueError('Duplicate reader/case response')
        for row in frame.to_dict('records'):
            rid = row['review_id']
            if rid not in mapping.index:
                raise ValueError('Unknown review ID')
            anatomy = label(row.get('anatomy_preserved',''))
            assessable = label(row.get('lesion_assessable',''))
            lesion = label(row.get('lesion_preserved',''))
            if lesion is not None and assessable is not True:
                raise ValueError('Lesion judgement requires lesion_assessable=yes')
            if any(x is not None for x in [anatomy, assessable, lesion]):
                if not row.get('reviewer','').strip():
                    raise ValueError('Actual judgements require a reviewer identifier')
                readers.append(dict(review_id=rid, reviewer=row['reviewer'], anatomy_preserved=anatomy,
                                    lesion_assessable=assessable, lesion_preserved=lesion,
                                    comment=row.get('comment',''), **mapping.loc[rid].to_dict()))
    if annotations:
        annotations = Path(annotations).resolve()
        inputs[str(annotations)] = digest(annotations)
        content = json.loads(annotations.read_text(encoding='utf-8'))
        seen = set()
        for row in content['cases']:
            rid = row['review_id']
            if rid not in mapping.index or (rid,row.get('reviewer')) in seen:
                raise ValueError('Unknown/duplicate annotation')
            seen.add((rid,row.get('reviewer')))
            mask_present = [row.get(k) is not None for k in ['source_mask','edit_mask']]
            points_present = [row.get(k) is not None for k in ['source_landmarks','edit_landmarks']]
            if any(mask_present) and not all(mask_present) or any(points_present) and not all(points_present):
                raise ValueError('Both source and edit annotations are required')
            if not any(mask_present+points_present):
                continue
            if not row.get('reviewer'):
                raise ValueError('Annotations require reviewer attribution')
            info = mapping.loc[rid]
            with np.load(pilot/'arrays'/f'{info.case_id}.npz') as pack:
                shape = pack['source'].shape
            result = dict(review_id=rid, reviewer=row['reviewer'], **info.to_dict())
            if all(mask_present):
                masks = []
                for key in ['source_mask','edit_mask']:
                    path = Path(row[key])
                    path = path.resolve() if path.is_absolute() else (annotations.parent/path).resolve()
                    inputs[str(path)] = digest(path)
                    masks.append(mask_array(np.load(path, allow_pickle=False), shape))
                result.update({'annotated_'+k:v for k,v in mask_metrics(*masks).items()})
            if all(points_present):
                result.update(landmark_metrics(row['source_landmarks'],row['edit_landmarks'],shape))
            measured.append(result)
    output.mkdir(parents=True, exist_ok=False)
    (pd.DataFrame(readers) if readers else pd.DataFrame(columns=[
        'review_id','reviewer','case_id','variant','anatomy_preserved','lesion_assessable','lesion_preserved','comment'])).to_csv(
            output/'reader_responses.csv', index=False)
    (pd.DataFrame(measured) if measured else pd.DataFrame(columns=[
        'review_id','reviewer','case_id','variant','annotated_valid','annotated_dice','landmark_mean_shift_px'])).to_csv(
            output/'annotation_metrics.csv', index=False)
    # Per-reader judgements remain separate; do not invent consensus.
    joint = []
    predictions = read_table(pilot/'predictions.csv')
    for reader in readers:
        for pred in predictions[predictions.review_id == reader['review_id']].to_dict('records'):
            anatomy = reader['anatomy_preserved']
            lesion_required = reader['lesion_assessable'] is True
            lesion = reader['lesion_preserved']
            preservation = (anatomy and lesion) if anatomy is not None and lesion_required and lesion is not None else (
                anatomy if anatomy is not None and reader['lesion_assessable'] is False else None)
            joint.append(dict(review_id=reader['review_id'], reviewer=reader['reviewer'], judge=pred['judge'],
                              axis=pred['axis'], variant=pred['variant'], anatomy_preserved=anatomy,
                              preservation_with_lesion_assessment=preservation,
                              target_hit_and_reviewed_preservation=bool(pred['target_hit'] and preservation) if preservation is not None else None))
    (pd.DataFrame(joint) if joint else pd.DataFrame(columns=[
        'review_id','reviewer','judge','axis','variant','anatomy_preserved',
        'preservation_with_lesion_assessment','target_hit_and_reviewed_preservation'])).to_csv(
            output/'joint_reader_judge.csv', index=False)
    json_write(output/'coverage.json', dict(total_output_cases=len(mapping),
               cases_with_anatomy_judgement=len({r['review_id'] for r in readers if r['anatomy_preserved'] is not None}),
               cases_with_annotations=len({r['review_id'] for r in measured}),
               reader_response_rows=len(readers), consensus_inferred=False,
               missing_judgements_replaced=False, clinical_validation_claimed=False))
    json_write(output/'inputs.json', inputs)
    (output/'README.md').write_text('Generated by scripts/v5_anatomy/score_review.py.\n'
        'reader_responses.csv preserves actual answers by reader; annotation_metrics.csv measures native-grid labels.\n'
        'joint_reader_judge.csv separates classifier hits from reviewed preservation. coverage.json records missingness.\n'
        'inputs.json hashes supplied evidence. No clinical qualification or consensus is inferred.\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review-csv')
    parser.add_argument('--annotations')
    args = parser.parse_args()
    print(score(args.pilot,args.output,args.review_csv,args.annotations))
