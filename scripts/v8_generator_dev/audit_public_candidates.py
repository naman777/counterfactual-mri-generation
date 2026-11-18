"""Bounded public metadata queries only; no image/manifest download endpoint exists here."""
import argparse
import hashlib
import json
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from scripts.v7_geometry.orientation import orientation_plan
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write

BASE='https://services.cancerimagingarchive.net/nbia-api'
TAGS={'(0008,0070)':'manufacturer','(0018,0087)':'field_strength_t',
      '(0018,0081)':'echo_time_ms','(0018,0080)':'repetition_time_ms',
      '(0018,0020)':'scanning_sequence','(0018,0023)':'acquisition_type',
      '(0020,0037)':'iop','(0028,0010)':'rows','(0028,0011)':'columns',
      '(0028,0030)':'pixel_spacing','(0018,1314)':'flip_angle',
      '(0018,0010)':'contrast_bolus_agent','(0008,0018)':'sample_sop_uid'}


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new metadata audit ID required')
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'raw').mkdir()
    json_write(output/'protocol.json',dict(collection='Duke-Breast-Cancer-MRI',sample_patients=12,
        selection_seed=482950,exclude_all_local_raw_patients=True,metadata_only=True,
        maximum_total_response_bytes=4*1024**2,maximum_response_bytes=1024**2,
        endpoints=['v1/getPatient','v1/getSeries','getDicomTags'],
        image_download_endpoint_enabled=False,model_outputs_used_for_selection=False,
        compatibility='Representative metadata only; full-series/contrast/image validity unresolved',
        documentation=['https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/',
        'https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide',
        'https://wiki.cancerimagingarchive.net/display/Public/NBIA+Advanced+REST+API+Guide']))
    session=requests.Session()
    byte_count=0
    requests_log=[]

    def get(endpoint,params):
        nonlocal byte_count
        if endpoint not in ('v1/getPatient','v1/getSeries','getDicomTags'):
            raise ValueError('Only whitelisted metadata endpoints are supported')
        response=session.get(BASE+'/services/'+endpoint,params=params,timeout=30,stream=True)
        if urlparse(response.url).hostname!='services.cancerimagingarchive.net':
            raise ValueError('Metadata endpoint redirected to another host')
        response.raise_for_status()
        chunks=[]
        size=0
        for chunk in response.iter_content(65536):
            size+=len(chunk)
            byte_count+=len(chunk)
            if size>1024**2 or byte_count>4*1024**2:
                response.close()
                raise ValueError('Metadata byte cap reached; no image download attempted')
            chunks.append(chunk)
        content=b''.join(chunks)
        requests_log.append(dict(endpoint=endpoint,params=params,status=response.status_code,
            bytes=size,sha256=hashlib.sha256(content).hexdigest()))
        time.sleep(.15)
        return json.loads(content)

    # The guest flow is documented by TCIA. Tokens remain only in memory and
    # are never emitted, stored in provenance or placed in command arguments.
    token=session.post(BASE+'/oauth/token',data=dict(username='nbia_guest',password='',
        client_id='NBIA',grant_type='password'),timeout=30)
    token.raise_for_status()
    session.headers['Authorization']='Bearer '+token.json()['access_token']
    collection=get('v1/getPatient',dict(Collection='Duke-Breast-Cancer-MRI',format='json'))
    identifiers=sorted({row['PatientId'] for row in collection})
    local={path.name for path in (ROOT/'data/raw/duke_breast_cancer_mri').glob('Breast_MRI_*') if path.is_dir()}
    for split in ('train','val','test'):
        local.update(pd.read_csv(ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv'),usecols=['patient_id']).patient_id)
    pool=sorted(set(identifiers)-local)
    selected=np.random.default_rng(482950).choice(pool,size=min(12,len(pool)),replace=False).tolist()
    json_write(output/'patient_selection.json',dict(public_patient_ids=identifiers,excluded_local_patient_ids=sorted(local),
        selected_unseen_patient_ids=selected,selection_before_series_or_header_queries=True))
    train=CanonicalDataset('data/processed/ax_t1_128_canonical/view_v1_20261116_01','train').frame
    slope,offset=np.polyfit(train.echo_time,train.echo_time_norm,1)
    if not np.allclose(train.echo_time*slope+offset,train.echo_time_norm,rtol=0,atol=1e-8):
        raise ValueError('Current fixed EchoTime transform is not the expected affine mapping')
    candidates,coverage=[],[]
    for patient in selected:
        series=get('v1/getSeries',dict(Collection='Duke-Breast-Cancer-MRI',PatientID=patient,format='json'))
        json_write(output/'raw'/(patient+'_series.json'),series)
        matching=sorted([row for row in series if row.get('Modality')=='MR' and
            row.get('SeriesDescription','').strip().lower()=='ax t1'],key=lambda row:row['SeriesInstanceUID'])
        coverage.append(dict(patient_id=patient,series_metadata_rows=len(series),exact_ax_t1_series=len(matching)))
        if not matching:
            continue
        chosen=matching[0]
        data=get('getDicomTags',dict(SeriesUID=chosen['SeriesInstanceUID']))
        json_write(output/'raw'/(patient+'_sample_tags.json'),data)
        tag_values={row['element'].upper().replace(' ',''):row.get('data','') for row in data}
        values={name:str(tag_values.get(tag,'')) for tag,name in TAGS.items()}
        m=0 if 'GE' in values['manufacturer'].upper() else 1 if 'SIEMENS' in values['manufacturer'].upper() else None
        f_value=float(values['field_strength_t'])
        f=0 if f_value==1.5 else 1 if f_value==3 else None
        echo=float(values['echo_time_ms'])
        norm=float(echo*slope+offset)
        geometry=False
        try:
            orientation_plan([float(value) for value in values['iop'].split('\\')],
                             (int(values['rows']),int(values['columns'])))
            geometry=True
        except (ValueError,TypeError):
            pass
        record=dict(patient_id=patient,series_uid=chosen['SeriesInstanceUID'],
            series_description=chosen['SeriesDescription'],image_count=chosen.get('ImageCount'),
            declared_series_bytes=chosen.get('FileSize'),manufacturer_label=m,field_strength_label=f,
            source_echo_time_norm=norm,normalisation_within_current_global_range=0<=norm<=1,
            sample_geometry_compatible=geometry,
            sample_2d_spin_echo_family=values['acquisition_type']=='2D' and 'SE' in values['scanning_sequence'],
            contrast_status_proven=False,full_series_compatibility_proven=False,
            image_downloaded=False,**values)
        if m is not None and f is not None:
            for axis,target_m,target_f in [('manufacturer',1-m,f),('field_strength',m,1-f)]:
                support=request_support(train,target_m,target_f,norm,5)
                record[axis+'_direction']=f'{m if axis=="manufacturer" else f}_to_{1-m if axis=="manufacturer" else 1-f}'
                record[axis+'_target_support']=support['condition_support_status']
        candidates.append(record)
    csv_write(output/'series_coverage.csv',pd.DataFrame(coverage))
    csv_write(output/'candidate_headers.csv',pd.DataFrame(candidates))
    json_write(output/'requests.json',requests_log)
    inputs={'scripts/v8_generator_dev/audit_public_candidates.py':digest(__file__)}
    for split in ('train','val','test'):
        path=ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv')
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,metadata_only=True,sampled_unseen_patients=len(selected),
        exact_current_series_candidates=len(candidates),public_metadata_bytes=byte_count,
        image_series_downloaded=False,local_or_test_acquisition_fields_used=False,
        data_split_changed=False,compatibility_established=False,credentials_persisted=False))
    (output/'README.md').write_text(
        '# Small public metadata candidate audit\n\nProducer: `python -m scripts.v8_generator_dev.audit_public_candidates`.\n'
        'protocol.json freezes the read-only endpoints/caps and primary documentation. patient_selection.json '
        'records exclusion of all local patients before metadata-only sampling; raw/ retains public series/header '
        'responses. series_coverage.csv/candidate_headers.csv show exact-name series, representative tags and '
        'current target-condition support. requests.json hashes bounded API responses without tokens. '
        'inputs.json fingerprints code/local membership; complete.json denies image downloads, new splits '
        'or established compatibility. A candidate/header is not a compatible new training patient. '
        'All image downloads require a separately reviewed size/scope.\n',encoding='utf-8')
    print('PUBLIC METADATA AUDIT',len(candidates),'candidates;',byte_count,'metadata bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    audit(parser.parse_args().run_id)
