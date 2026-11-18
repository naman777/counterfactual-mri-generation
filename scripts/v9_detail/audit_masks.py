"""Freeze source-mask candidate, fit train-only warning cutoffs, retain validation failures."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import csv_write,json_write,digest,ROOT
from scripts.v8_generator_dev.preservation_screen import controls,GRID_CONTROLS
from scripts.v9_detail.common import start,sources,finish,interval,display
from scripts.v9_detail.masks import SETTINGS,make_mask,geometry,mask_iou

VARIANTS=('historical_005','noise_aware_v1')


def audit(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.audit_masks','Source-only mask development audit')
    json_write(output/'protocol.json',dict(settings=SETTINGS,variants=list(VARIANTS),
        selection='Historical 72 training guide sources; frozen 15 editor-validation sources (different validation slices from earlier guide probe)',
        calibration='Warning cutoffs fit to training masks/photometric control stability only; no edit outputs or validation tuning',
        warning_rule='Area outside train 1st--99th percentiles, gradient capture below train 1st percentile, control minimum IoU below train 5th percentile, or strict interior <64 px',
        controls=list(GRID_CONTROLS),source_only=True,clinical_mask_validation=False,
        caveat='Noise corner can contain tissue; mask is contrast-sensitive; low-signal tissue may be omitted. No automatic mask promotion.'))
    selection,items=sources(inputs)
    rows,sensitivity,panels=[],[],[]
    # Training rows/controls are fully computed before cutoffs, then fixed for val.
    thresholds={}
    for split in ('train','val'):
        for row,source in items:
            if row.split!=split:
                continue
            transformed=controls(source,583110+row.dataset_index)
            masks={}
            for variant in VARIANTS:
                mask,info=make_mask(source,variant);masks[variant]=mask
                stable=[]
                for control in GRID_CONTROLS:
                    other,_=make_mask(transformed[control],variant)
                    iou=mask_iou(mask,other);stable.append(iou)
                    sensitivity.append(dict(patient_id=row.patient_id,split=split,variant=variant,
                        control=control,mask_iou=iou,**geometry(transformed[control],other)))
                rows.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,split=split,
                    variant=variant,min_control_iou=float(np.nanmin(stable)),**info,**geometry(source,mask)))
            if split=='val':
                panels.append((row,source,masks))
        if split=='train':
            trained=pd.DataFrame(rows)
            for variant,g in trained.groupby('variant'):
                thresholds[variant]=dict(area_low=float(g.mask_fraction.quantile(.01)),
                    area_high=float(g.mask_fraction.quantile(.99)),
                    gradient_low=float(g.gradient_energy_inside.quantile(.01)),
                    stability_low=float(g.min_control_iou.quantile(.05)))
            json_write(output/'thresholds.json',dict(training_patients=selection[selection.split=='train'].patient_id.nunique(),
                thresholds=thresholds,selection_or_optimisation_on_validation=False))
        print('MASK AUDIT',split,flush=True)
    frame=pd.DataFrame(rows)
    flags=[]
    for row in frame.itertuples():
        limits=thresholds[row.variant]
        reasons=[]
        if row.mask_fraction<limits['area_low'] or row.mask_fraction>limits['area_high']:
            reasons.append('train_area_outlier')
        if row.gradient_energy_inside<limits['gradient_low']:
            reasons.append('train_gradient_capture_outlier')
        if row.min_control_iou<limits['stability_low']:
            reasons.append('train_control_stability_outlier')
        if row.empty_or_no_strict_interior:
            reasons.append('strict_interior_under64')
        flags.append(dict(qc_warning=int(bool(reasons)),qc_reasons=';'.join(reasons)))
    frame=pd.concat([frame,pd.DataFrame(flags)],axis=1)
    csv_write(output/'mask_cases.csv',frame)
    csv_write(output/'control_cases.csv',pd.DataFrame(sensitivity))
    csv_write(output/'patient_selection.csv',selection)
    summaries=[]
    for (split,variant),group in frame.groupby(['split','variant']):
        for metric in ('mask_fraction','gradient_energy_inside','min_control_iou','near_full_frame',
                       'empty_or_no_strict_interior','qc_warning'):
            summaries.append(dict(split=split,variant=variant,metric=metric,**interval(group,metric)))
    summary=pd.DataFrame(summaries);csv_write(output/'patient_intervals.csv',summary)
    fig,axes=plt.subplots(len(panels),3,figsize=(8,2*len(panels)),squeeze=False)
    for i,(row,source,masks) in enumerate(sorted(panels,key=lambda x:x[0].patient_id)):
        for j,(name,img) in enumerate([('Source',source),*masks.items()]):
            axes[i,j].imshow(img,cmap='gray',vmin=0,vmax=1);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0: axes[i,j].set_title(name)
        axes[i,0].set_ylabel(row.patient_id,fontsize=8)
    fig.suptitle('Every validation source; masks are unvalidated intensity proxies')
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/'validation_masks.png',dpi=110);plt.close(fig)
    lines=['# Source-only mask audit','',
        'The noise-aware candidate is a fixed engineering revision, not an anatomical segmentation. '
        'It estimates background from the lowest-median corner, removes small components and fills only small holes. '
        'Its constants were frozen before the train/validation computations. Warning cutoffs use training sources only. '
        'All validation sources and warnings remain; no generated output is used to fit or choose masks.','',
        '| Variant / validation metric | Estimate [95% patient CI] |','|---|---|']
    for row in summary[summary.split.eq('val')].itertuples():
        lines.append(f'| {row.variant} / {row.metric} | {display(row)} |')
    lines += ['', 'All intervals are equal-patient percentile bootstraps. Zero observed warnings or failures can '
        'produce degenerate intervals; this is not population certainty or an anatomical accuracy estimate. '
        'Gradient capture is an appearance proxy and can be inflated by background noise. '
        'The corner can contain tissue and both variants can omit low-signal tissue. Training cutoffs measure '
        'engineering outliers rather than clinical mask validity.','',
        'The old near-full-frame failure is retained alongside the candidate, not erased. A changed area alone '
        'does not establish a corrected tissue boundary. Neither mask is selected automatically for efficacy. '
        'Any prospective masked evaluation must freeze its failure policy and retain whole/interior/boundary results.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'OUTPUTS.md').write_text('All outputs produced by scripts.v9_detail.audit_masks. '
        'mask_cases.csv: source masks/QC; control_cases.csv: photometric stability; thresholds.json: train cutoffs; '
        'patient_intervals.csv: all patient CIs; patient_selection.csv: unchanged source selection; '
        'validation_masks.png: all validation patients; RESULTS.md: interpretation.\n',encoding='utf-8')
    finish(output,inputs,train_calibrated=True,validation_tuned=False,mask_selected=False)
    print('MASK AUDIT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    audit(p.parse_args().run_id)
