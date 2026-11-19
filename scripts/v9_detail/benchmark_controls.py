"""Train-calibrated fine-control screen: retain false flags and missed erasures."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import csv_write,json_write
from scripts.v8_generator_dev.preservation_screen import controls,GRID_CONTROLS
from scripts.v9_detail.common import start,sources,finish,interval,display
from scripts.v9_detail.controls import METRICS,measure,score,damage_controls


def benchmark(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.benchmark_controls','Fine-detail synthetic control benchmark')
    json_write(output/'protocol.json',dict(metrics=list(METRICS),source_domain='Fixed noise-aware v1 mask; 8px-eroded interior and +/-3px boundary',
        threshold_rule='95th percentile of training-patient maximum over all six grid-preserving controls; any metric exceedance flags',
        grid_controls=list(GRID_CONTROLS),damage_controls=['erase1/2/3/5px','local_smooth7px','local_texture_mirror7px','boundary_gain_seam','boundary_translate2px'],
        locations=['variance_peak','gradient_peak','seeded_texture'],frame='Canonical 128px source grid; no resizing/registration',
        history='Validation-informed metric development; calibration uses training only; not independent confirmation',
        no_clinical_lesion_labels=True,mask_is_proxy=True,invalid_policy='Retain invalid measurements; do not count as screened or preserved'))
    selection,items=sources(inputs)
    calibration,validation=[],[]
    for split in ('train','val'):
        for row,source in items:
            if row.split!=split:continue
            values=controls(source,583120+row.dataset_index)
            for name in GRID_CONTROLS:
                record=dict(patient_id=row.patient_id,dataset_index=row.dataset_index,control=name,
                    location='whole',family='grid_preserving',y=-1,x=-1,width_px=0,**measure(source,values[name]))
                (calibration if split=='train' else validation).append(record)
            if split=='val':
                validation.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,control='identity',
                    location='whole',family='identity',y=-1,x=-1,width_px=0,**measure(source,source)))
                damaged=damage_controls(source,583130+row.dataset_index)
                for name,location,image,info in damaged:
                    validation.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,control=name,
                        location=location,family='injected_damage',**info,**measure(source,image)))
                if not damaged:
                    validation.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,
                        control='invalid_source_domain',location='unavailable',family='invalid_source',
                        y=-1,x=-1,width_px=0,valid=False,**{k:np.nan for k in METRICS}))
        if split=='train':
            trained=pd.DataFrame(calibration)
            valid=trained[trained.valid]
            maxima=valid.groupby('patient_id')[list(METRICS)].max()
            if len(maxima)<10:raise ValueError('Insufficient calibration')
            thresholds={k:max(1e-6,float(maxima[k].quantile(.95))) for k in METRICS}
            json_write(output/'thresholds.json',dict(thresholds=thresholds,training_patients=len(maxima),
                invalid_training_rows=int((~trained.valid).sum()),validation_retuned=False))
        print('FINE CONTROLS',split,flush=True)
    csv_write(output/'calibration.csv',trained)
    frame=pd.DataFrame([dict(row,**score(row,thresholds)) for row in validation])
    csv_write(output/'validation_controls.csv',frame)
    summaries=[]
    for (family,control),g in frame.groupby(['family','control']):
        for location,part in [('all_locations',g),*list(g.groupby('location'))]:
            summaries.append(dict(family=family,control=control,location=location,
                invalid_cases=int(part.flag.isna().sum()),**interval(part,'flag')))
    summary=pd.DataFrame(summaries);csv_write(output/'control_intervals.csv',summary)
    csv_write(output/'patient_selection.csv',selection)
    points=summary[summary.location.eq('all_locations')]
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    ax.errorbar(points.estimate*100,np.arange(len(points)),
        xerr=[(points.estimate-points.ci_low)*100,(points.ci_high-points.estimate)*100],fmt='o',capsize=3)
    ax.set_yticks(range(len(points)),points.control);ax.invert_yaxis();ax.set_xlim(-3,103)
    ax.set_xlabel('Flagged control cases, patient mean % [95% patient CI]')
    ax.set_title('Train-calibrated synthetic sensitivity and false flags\nReused development patients; no clinical lesion ground truth')
    fig.savefig(output/'control_detection.png',dpi=140);plt.close(fig)
    lines=['# Fine-detail engineering benchmark','',
        'This is an expanded development screen. Its source-only mask and metrics are informed by earlier validation '
        'diagnostics. Numerical cutoffs are fitted only on training patients and frozen before validation. '
        'No synthetic erasure is a clinical lesion or anatomy label. All locations, failures and false flags remain.','',
        '| Control | Flag fraction [95% patient CI] | Valid patients / total |','|---|---|---|']
    for row in points.itertuples():
        lines.append(f'| {row.control} | {display(row)} | {row.patients_valid} / {row.patients_total} |')
    lines += ['', 'For small erasures/local texture controls the headline is the mean fraction of source-selected '
        'locations flagged within each patient, not a patient-any-detection rate. Location-specific estimates are '
        'in control_intervals.csv. Grid-preserving transforms can change visibility; their flags are engineering '
        'false flags relative to a grid-preservation null, not clinical specificity. Tiny erasures are widths '
        'on the processed 128px grid; they are not millimetre lesion sizes. Degenerate intervals are not '
        'population certainty. Boundary metrics deliberately assess a separate boundary domain; interior '
        'measurements do not credit copied exterior pixels. Historical screens/cutoffs are unchanged. '
        'This screen is not an independent preservation test and no edited output is declared preserved.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'OUTPUTS.md').write_text('Producer: scripts.v9_detail.benchmark_controls. calibration.csv/thresholds.json: '
        'training metrics and fixed cutoffs; validation_controls.csv: every validation transform/location/metric; '
        'control_intervals.csv: patient intervals and missingness; control_detection.png: sensitivity/false flags; '
        'patient_selection.csv: source membership; RESULTS.md: interpretation.\n',encoding='utf-8')
    finish(output,inputs,thresholds_train_only=True,validation_informed_metric_development=True,
        screen_promoted=False,clinical_lesion_sensitivity_established=False)
    print('FINE CONTROL BENCHMARK COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    benchmark(p.parse_args().run_id)
