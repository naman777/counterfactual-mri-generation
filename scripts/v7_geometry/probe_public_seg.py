"""Inspect one <=1 MB public SEG sample; never download a collection or apply its masks."""
import argparse
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import pydicom
import requests
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path
from scripts.v7_geometry.annotation_compatibility import check_source,seg_references

ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'reports/v4_downstream/mask_feasibility_20261016_01/official_2d_segmentation_catalog_matches.csv'


def bounded_get(url,limit):
    content=bytearray()
    with requests.get(url,stream=True,timeout=30) as response:
        response.raise_for_status()
        if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('Payload exceeds declared small-sample cap')
        for chunk in response.iter_content(32768):
            content.extend(chunk)
            if len(content)>limit:raise ValueError('Payload exceeds declared small-sample cap')
    return bytes(content)


def probe(output):
    output=project_path(output)
    if not output.is_relative_to(ROOT/'reports'):raise ValueError('Use a new reports directory')
    if output.exists():raise FileExistsError(output)
    val_path=ROOT/'data/processed/ax_t1_128/splits/val.csv';val=pd.read_csv(val_path);catalog=pd.read_csv(CATALOG)
    candidates=catalog[catalog.PatientID.isin(val.patient_id)].sort_values('PatientID')
    if candidates.empty:raise ValueError('No original-validation patient has an official SEG entry')
    chosen=candidates.iloc[0]
    if float(chosen.series_size_MB)>1.:raise ValueError('Declared SEG exceeds the 1 MB sample cap')
    output.mkdir(parents=True)
    json_write(output/'protocol.json',dict(patient=str(chosen.PatientID),selection='first sorted original-validation patient in official SEG catalogue',
        max_seg_bytes=1000000,max_listing_bytes=65536,mask_pixels_decoded=False,apply_annotations=False,
        collection_download=False,source_catalogue_sha256=digest(CATALOG)))
    base=f'https://{chosen.aws_bucket}.s3.amazonaws.com'
    listing_url=base+'/?list-type=2&prefix='+quote(str(chosen.crdc_series_uuid)+'/',safe='')
    try:
        listing=bounded_get(listing_url,65536)
        with (output/'listing.xml').open('xb') as handle:handle.write(listing)
        tree=ET.fromstring(listing)
        objects=[e for e in tree.iter() if e.tag.endswith('Contents')]
        files=[]
        for obj in objects:
            fields={e.tag.rsplit('}',1)[-1]:e.text for e in obj}
            if fields.get('Key','').endswith('.dcm'):files.append(fields)
        if len(files)!=1 or int(files[0]['Size'])>1000000:raise ValueError('Expected exactly one small SEG object')
        url=base+'/'+quote(files[0]['Key'],safe='/')
        payload=bounded_get(url,1000000)
        with (output/'public_seg.dcm').open('xb') as handle:handle.write(payload)
        seg=pydicom.dcmread(io.BytesIO(payload),stop_before_pixels=True)
        if str(seg.SOPClassUID)!='1.2.840.10008.5.1.4.1.1.66.4':raise ValueError('Object is not DICOM Segmentation Storage')
        references=seg_references(seg);comparisons=[];inputs={str(CATALOG.relative_to(ROOT)):digest(CATALOG),str(val_path.relative_to(ROOT)):digest(val_path)}
        for row in val[val.patient_id==chosen.PatientID].itertuples():
            source=project_path(row.slice_path);ds=pydicom.dcmread(source,stop_before_pixels=True)
            result=check_source(seg,ds)
            comparisons.append(dict(patient_id=row.patient_id,source_sop_uid=row.sop_uid,source_series_uid=row.series_uid,
                                    source_series_description=str(getattr(ds,'SeriesDescription','')),**{k:v for k,v in result.items() if k not in references}))
            inputs[str(source.relative_to(ROOT))]=digest(source)
        pd.DataFrame(comparisons).to_csv(output/'current_ax_t1_comparisons.csv',index=False)
        # Cached IDC catalogue only: produce a source-series download proposal,
        # never fetch its MRI instances or assume it is the current ax-t1 series.
        import idc_index_data
        full=pd.read_parquet(idc_index_data.IDC_INDEX_PARQUET_FILEPATH,filters=[('collection_id','==','duke_breast_cancer_mri')])
        series=full[full.SeriesInstanceUID.isin(references['referenced_series_uids'])]
        keep=[k for k in ['PatientID','SeriesInstanceUID','SeriesDescription','Modality','instanceCount','series_size_MB','crdc_series_uuid','series_aws_url'] if k in series]
        series[keep].to_csv(output/'referenced_series_metadata.csv',index=False)
        json_write(output/'reference_inventory.json',references);json_write(output/'inputs.json',inputs)
        result=dict(complete=True,small_public_sample_only=True,patient=str(chosen.PatientID),
            seg_bytes=len(payload),seg_url=url,seg_sha256=digest(output/'public_seg.dcm'),
            current_ax_t1_sources_checked=len(comparisons),
            matching_current_sop_uids=sum(r['source_sop_referenced'] for r in comparisons),
            matching_current_series=sum(r['source_series_referenced'] for r in comparisons),
            direct_transfer_approved=False,mask_pixels_decoded=False,external_messages_sent=False)
        json_write(output/'complete.json',result);print(result,flush=True)
    except Exception as error:
        json_write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,reason=str(error),
            collection_download_attempted=False,direct_transfer_approved=False))
        raise
    finally:
        (output/'README.md').write_text('# Bounded public SEG reference probe\n\n'
            'Producer: `python -m scripts.v7_geometry.probe_public_seg`.\n'
            'protocol.json declares one size-capped validation-patient sample. listing.xml and public_seg.dcm are unchanged public source bytes.\n'
            'reference_inventory.json lists SEG source UIDs/labels; current_ax_t1_comparisons.csv checks actual local source headers.\n'
            'referenced_series_metadata.csv identifies the referenced MRI series from a cached IDC catalogue; no MRI download is performed.\n'
            'inputs.json hashes local evidence; complete.json records size, source URL and compatibility result. A failed probe records failure.json.\n'
            'Pixel data are not decoded, masks are not transferred and no anatomy ground truth is inferred from matching patient IDs.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);print(probe(p.parse_args().output))
