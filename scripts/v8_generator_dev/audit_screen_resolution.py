"""Frozen-screen small-erasure sensitivity and physical pixel-scale audit."""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import foreground_proxy
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import source_native
from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,project_path,verify_inputs
from scripts.v8_generator_dev.preservation_screen_v2 import local_variance,measurements,screen


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new audit ID required')
    base=ROOT/'reports/v8_generator_dev'
    benchmark=base/'preservation_benchmark_v2_20261117_01'
    verify_inputs(json.loads((benchmark/'inputs.json').read_text()))
    thresholds=json.loads((benchmark/'thresholds.json').read_text())['thresholds']
    receipt=json.loads((base/'failed_pair_results_20261117_01/complete.json').read_text())
    request_path=project_path(receipt['evaluations']['additive'])/'requests.csv'
    indices=sorted(pd.read_csv(request_path).dataset_index.unique())
    dataset=CanonicalDataset('data/processed/ax_t1_128_canonical/view_v1_20261116_01','val')
    output=base/run_id
    output.mkdir(exist_ok=False)
    rows,scales=[],[]
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in (
        benchmark/'thresholds.json',request_path,ROOT/'scripts/v8_generator_dev/audit_screen_resolution.py',
        ROOT/'scripts/v8_generator_dev/preservation_screen_v2.py')}
    for index in indices:
        row=dataset.frame.iloc[index]
        source=source_native(dataset.image_01(int(index))[None],dataset.plans[index])[0]
        path=project_path(row.source_image_path)
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        row_spacing=float(row.native_row_spacing_mm*row.native_rows/128)
        col_spacing=float(row.native_column_spacing_mm*row.native_columns/128)
        scales.append(dict(patient_id=row.patient_id,dataset_index=int(index),
            processed_source_row_mm=row_spacing,processed_source_col_mm=col_spacing,
            one_pixel_erasure_row_mm=row_spacing,one_pixel_erasure_col_mm=col_spacing,
            five_pixel_erasure_row_mm=5*row_spacing,five_pixel_erasure_col_mm=5*col_spacing,
            assumption='Whole-image centre-aligned resize and lossless source/native inverse; not lesion size'))
        roi=ndi.binary_erosion(foreground_proxy(source,.05),iterations=6)
        if not roi.any():
            raise ValueError('Insufficient source ROI for synthetic sensitivity test')
        gradient=np.hypot(*np.gradient(ndi.gaussian_filter(source,1)))
        for location,feature in [('gradient_peak',gradient),('variance_peak',local_variance(source))]:
            y,x=np.unravel_index(np.argmax(np.where(roi,feature,-1)),source.shape)
            for size in (1,2,3):
                edit=source.copy()
                y0,x0=np.clip(y-size//2,0,128-size),np.clip(x-size//2,0,128-size)
                edit[y0:y0+size,x0:x0+size]=0
                values=measurements(source,edit)
                rows.append(dict(patient_id=row.patient_id,dataset_index=int(index),location=location,
                    erasure_pixels=size,patch_row_mm=size*row_spacing,patch_col_mm=size*col_spacing,
                    source_selected_y=int(y),source_selected_x=int(x),**screen(values,thresholds),
                    clinical_anatomy_validated=False))
    table=pd.DataFrame(rows)
    csv_write(output/'sensitivity_cases.csv',table)
    csv_write(output/'physical_scale.csv',pd.DataFrame(scales))
    summaries=[]
    for size,group in table.groupby('erasure_pixels'):
        summaries.append(dict(erasure_pixels=int(size),source_locations_per_patient=2,
            **bootstrap_patients(group,'screen_flag',482944)))
    csv_write(output/'sensitivity_intervals.csv',pd.DataFrame(summaries))
    lines=['# Resolution and small-damage sensitivity','',
        'This audit keeps the existing screen thresholds fixed and checks smaller artificial erasures '
        'at two source-chosen locations per patient. It is development sensitivity analysis on the same '
        'validation patients, not a clinical lesion benchmark.','',
        '| Erasure width on processed grid | Flagged, % [95% patient CI] |','|---|---|']
    for row in summaries:
        lines.append(f'| {row["erasure_pixels"]} pixel(s) | {100*row["mean"]:.1f} [{100*row["ci95_lo"]:.1f}, {100*row["ci95_hi"]:.1f}] |')
    lines+=['','physical_scale.csv gives actual per-source millimetres per processed pixel. '
        'A five-pixel erased square on a resized MRI is not a five-millimetre lesion. '
        'The original 128-pixel preprocessing already limits detail; upsampling the generated '
        'image cannot restore information lost during preprocessing.','',
        'Do not interpret detection of an artificial square as sensitivity to true lesions, '
        'absence of a flag as preserved anatomy, or a degenerate bootstrap interval as population certainty. '
        'The lower-resolution structure adapter addresses coarse correspondence; its effect on '
        'fine pathology remains unverified. Native-derived higher resolution remains a separately '
        'gated experiment after editing/preservation comparisons.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,thresholds_unchanged=True,
        validation_patients=len(indices),clinical_anatomy_validated=False,lesion_sensitivity_established=False,
        image_upsampled=False,test_pixels_used=False))
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.audit_screen_resolution`.\n'
        'sensitivity_cases.csv/intervals.csv retain every artificial erasure and patient CIs; '
        'physical_scale.csv records source-grid millimetres and assumptions. RESULTS.md explains limits; '
        'inputs.json hashes code/cutoffs/source; complete.json denies clinical lesion sensitivity.\n',encoding='utf-8')
    print('RESOLUTION SCREEN AUDIT',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    audit(parser.parse_args().run_id)
