"""Check constructed gain-composite severity relative to the gain null; no retuning."""
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scripts.v8_generator_dev.io import csv_write,json_write,digest
from scripts.v9_detail.common import BASE,start,sources,finish,interval,display
from scripts.v9_detail.controls import domains,measure


def audit(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.audit_boundary_control','Gain-composite control severity audit')
    json_write(output/'protocol.json',dict(role='Post-result diagnostic; no new detector/threshold or generated MRI editing',
        comparison='Constructed masked .8 gain versus source and uniform .8 gain',
        hypothesis='An undetected masked-gain control may resemble an allowed gain transform in dark exterior pixels',
        mask='Fixed noise_aware_v1 source mask',cases='Every original editor-validation source',
        caveat='No synthetic-control quality or anatomical severity ground truth'))
    _,items=sources(inputs)
    rows=[]
    for row,source in items:
        if row.split!='val':continue
        mask,_,boundary=domains(source)
        whole=source*.8
        composite=source.copy();composite[mask]=whole[mask]
        compared=measure(source,composite);gain=measure(source,whole)
        difference=composite-whole
        extra=np.hypot(*np.gradient(difference))
        rows.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,
            exterior_mean_source=float(source[~mask].mean()),
            composite_minus_uniform_mad=float(np.abs(difference).mean()),
            composite_minus_uniform_boundary_gradient_p99=float(np.quantile(extra[boundary],.99)),
            composite_source_boundary_change=compared['boundary_change'],
            uniform_gain_source_boundary_change=gain['boundary_change'],
            paired_boundary_change_difference=compared['boundary_change']-gain['boundary_change']))
    frame=pd.DataFrame(rows);csv_write(output/'control_severity.csv',frame)
    summaries=[]
    for metric in frame.columns:
        if metric not in ('patient_id','dataset_index'):
            summaries.append(dict(metric=metric,**interval(frame,metric)))
    summary=pd.DataFrame(summaries);csv_write(output/'patient_intervals.csv',summary)
    lines=['# Interpreting the missed masked-gain control','',
        'This diagnostic follows the expanded screen results. It compares the constructed masked gain '
        'with the grid-preserving full-image gain control. Source masks commonly place their boundary '
        'in dim tissue/background; copying that exterior can introduce little additional change. '
        'The earlier screen and its failed control results remain unchanged.','',
        '| Severity proxy | Estimate [95% patient CI] |','|---|---|']
    for row in summary.itertuples():lines.append(f'| {row.metric} | {display(row)} |')
    lines += ['', 'The constructed control has no independent visible-seam or anatomical-damage label. '
        'Failure to flag it is an observed stress-test outcome, but is not sufficient evidence of a '
        'boundary-damage detector failure. Training a new metric solely to flag this construction would '
        'risk penalising an allowed appearance change. Before a new boundary metric, specify a control '
        'with measured added discontinuity and keep matched global/local photometric nulls. No cutoff '
        'is retuned, this diagnosis is not an independent test, and no edited tissue is declared preserved.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'OUTPUTS.md').write_text('Producer: scripts.v9_detail.audit_boundary_control. control_severity.csv: '
        'every masked/global gain contrast; patient_intervals.csv: equal-patient uncertainty; '
        'RESULTS.md: limits on interpreting a missed artificial gain-composite control.\n',encoding='utf-8')
    finish(output,inputs,cutoffs_changed=False,new_metric_selected=False)
    print('BOUNDARY CONTROL AUDIT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    audit(p.parse_args().run_id)
