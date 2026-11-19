"""Quantify current joint-cell gaps and freeze candidate-ingestion requirements."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.data import CanonicalDataset, sampling_probabilities
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write


def plan(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new planning ID required')
    view='data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    train=CanonicalDataset(view,'train').frame
    val=CanonicalDataset(view,'val').frame
    probability=sampling_probabilities(train,5)
    weighted=train[['patient_id','manufacturer_label','field_strength_label']].copy()
    weighted['slice_sampling_probability']=probability
    patient=weighted.groupby(['patient_id','manufacturer_label','field_strength_label'],as_index=False).slice_sampling_probability.sum()
    rows=[]
    for (m,f),group in train.groupby(['manufacturer_label','field_strength_label']):
        validation=val[(val.manufacturer_label==m)&(val.field_strength_label==f)]
        weights=patient[(patient.manufacturer_label==m)&(patient.field_strength_label==f)]
        rows.append(dict(manufacturer_label=int(m),field_strength_label=int(f),
            training_patients=group.patient_id.nunique(),training_slices=len(group),
            validation_patients=validation.patient_id.nunique(),
            source_echo_time_ms_min=float(group.echo_time.min()),source_echo_time_ms_max=float(group.echo_time.max()),
            fixed_normalised_echo_min=float(group.echo_time_norm.min()),fixed_normalised_echo_max=float(group.echo_time_norm.max()),
            total_sampling_mass=float(weights.slice_sampling_probability.sum()),
            per_patient_sampling_mass=float(weights.slice_sampling_probability.mean())))
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    csv_write(output/'joint_cells.csv',pd.DataFrame(rows))
    csv_write(output/'patient_sampling_mass.csv',patient)
    effective=float(1/np.square(patient.slice_sampling_probability).sum())
    json_write(output/'sampling_summary.json',dict(training_patients=int(train.patient_id.nunique()),
        effective_patient_count_from_sampling_weights=effective,
        interpretation='Concentration of the intended sampler weights only; not an independent-subject count or confidence interval',
        duplicate_slices_count_as_new_patients=False))
    gates=[
        ('patient_membership','Exclude all existing train/validation/test patients and duplicate source SOPs before any prospective split assignment'),
        ('sequence','Axial non-DCE ax-t1 profile compatible with current task; string matching alone is insufficient'),
        ('condition_headers','Verify manufacturer, field strength and EchoTime from source DICOM headers; do not infer them from image appearance'),
        ('fixed_echo','Require joint target-cell support at source EchoTime for the intended axis/direction; never change EchoTime to force overlap'),
        ('geometry','Audit full-series IOP/IPP/grid/spacing/FOV; only validated lossless canonical transforms in the current profile'),
        ('normalisation','Retain the existing documented transforms for comparisons, or create a separately versioned data/model profile'),
        ('source_annotations','If labels are obtained, require exact source instance/series/grid references and verified annotation provenance; no resized DCE or box-derived tissue truth'),
        ('split_policy','Freeze new patient-separated development and untouched evaluation membership before model-specific outcome selection'),
        ('support_adequacy','The current two-patient direction rule is an engineering minimum, not proof of adequate statistical/clinical sample size'),
        ('execution_permission','A new large download or paid data/training job requires its own concrete approval; this plan launches none')]
    csv_write(output/'candidate_ingestion_gates.csv',pd.DataFrame(gates,columns=['gate','requirement']))
    inputs={}
    for split in ('train','val'):
        for path in (ROOT/view/'splits'/(split+'.csv'),ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv')):
            inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for path in (ROOT/'scripts/v8_generator_dev/plan_data_expansion.py',ROOT/'scripts/v8_generator_dev/data.py'):
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',inputs)
    (output/'README.md').write_text(
        '# Compatible-data expansion requirements\n\nProducer: `python -m scripts.v8_generator_dev.plan_data_expansion`.\n'
        'joint_cells.csv records patient counts, fixed-conditioning ranges and sampler mass; '
        'patient_sampling_mass.csv/sampling_summary.json expose concentration without counting repeated slices '
        'as new independent subjects. candidate_ingestion_gates.csv fixes candidate compatibility/provenance '
        'checks before any new dataset action. inputs.json hashes source metadata/code. '
        'Prioritise rare joint cells and source-fixed-EchoTime overlap, rather than more slices from the same '
        'patients. The unused-local audit found no exact current ax-t1 candidate. No additional cohort, '
        'split reassignment, threshold relaxation or new data download is performed here.\n',encoding='utf-8')
    json_write(output/'complete.json',dict(complete=True,metadata_only=True,data_downloaded=False,
        source_selection_changed=False,test_acquisition_fields_used=False,image_pixels_loaded=False,
        clinical_sample_size_adequacy_established=False))
    print('DATA EXPANSION REQUIREMENTS',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    plan(parser.parse_args().run_id)
