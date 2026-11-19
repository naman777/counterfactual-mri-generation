"""Fail-closed source-reference diagnostics for DICOM SEG annotations."""
import numpy as np


def seg_references(seg):
    sops=sorted({str(e.value) for e in seg.iterall() if e.keyword=='ReferencedSOPInstanceUID'})
    series=[]
    for item in getattr(seg,'ReferencedSeriesSequence',[]):
        if hasattr(item,'SeriesInstanceUID'):series.append(str(item.SeriesInstanceUID))
    labels=[str(getattr(s,'SegmentLabel','')) for s in getattr(seg,'SegmentSequence',[])]
    return dict(referenced_sop_uids=sops,referenced_series_uids=sorted(set(series)),segment_labels=labels)


def check_source(seg,source):
    references=seg_references(seg)
    sop_match=str(source.SOPInstanceUID) in references['referenced_sop_uids']
    series_match=str(source.SeriesInstanceUID) in references['referenced_series_uids']
    dimensions_match=int(seg.Rows)==int(source.Rows) and int(seg.Columns)==int(source.Columns)
    if not series_match:status='different_or_unresolved_series'
    elif not sop_match:status='different_or_unresolved_source_instance'
    elif not dimensions_match:status='different_source_grid'
    else:status='source_reference_matches_requires_frame_geometry_and_label_review'
    return dict(status=status,source_sop_referenced=sop_match,source_series_referenced=series_match,
                native_dimensions_match=dimensions_match,
                direct_transfer_approved=False,reason='Source references are necessary, never sufficient; frame geometry and label provenance still require verification',
                **references)
