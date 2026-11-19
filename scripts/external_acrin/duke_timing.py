"""Compare Duke training-series start times to same-study dynamic series."""

import argparse
from pathlib import Path
import re

import pandas as pd
import pydicom

from audit import ROOT


def clock_key(dataset):
    date = str(dataset.get('SeriesDate', ''))
    time = str(dataset.get('SeriesTime', ''))
    if re.fullmatch(r'\d{8}', date) and re.fullmatch(r'\d{6}(?:\.\d{1,6})?', time):
        return date, float(time)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/processed/ax_t1_128/splits/train.csv', usecols=['patient_id'])
    source = pd.read_csv(ROOT / 'data/metadata/dicom_metadata.csv', low_memory=False)
    source = source[source.PatientID.isin(train.patient_id)].drop_duplicates('SeriesInstanceUID')
    audited = []
    for (patient, study), rows in source.groupby(['PatientID', 'StudyInstanceUID']):
        native = rows[rows.SeriesDescription.str.lower().eq('ax t1')]
        dynamic = rows[rows.SeriesDescription.str.contains(r'\bdyn(?:amic)?\b', case=False, na=False)]
        if native.empty:
            continue
        times = {}
        for row in pd.concat([native, dynamic]).itertuples(index=False):
            dataset = pydicom.dcmread(ROOT / row.slice_path, stop_before_pixels=True,
                                     specific_tags=['SeriesDate', 'SeriesTime', 'ContrastBolusStartTime'])
            times[row.SeriesInstanceUID] = clock_key(dataset), str(dataset.get('ContrastBolusStartTime', ''))
        starts = [times[uid][0] for uid in dynamic.SeriesInstanceUID if times[uid][0] is not None]
        first = min(starts, default=None)
        for row in native.itertuples(index=False):
            start = times[row.SeriesInstanceUID][0]
            before = start < first if start and first and start[0] == first[0] else None
            audited.append({'patient_id': patient, 'study_uid': study, 'native_series_uid': row.SeriesInstanceUID,
                            'native_series_start': start, 'earliest_dynamic_series_start': first,
                            'native_precedes_dynamic_same_day': before,
                            'bolus_start_tag_values': sum(bool(value) for _, value in times.values())})
    result = pd.DataFrame(audited)
    if set(result.patient_id) != set(train.patient_id) or not result.patient_id.is_unique:
        raise ValueError('Expected one timing comparison per Duke training patient')
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        result.to_csv(stream, index=False)
    print(result.native_precedes_dynamic_same_day.value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
