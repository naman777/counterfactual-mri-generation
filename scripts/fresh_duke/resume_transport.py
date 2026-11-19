"""Resume the frozen cohort after a slow IDC transfer, using the same IDC S3 objects."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, local
from time import sleep
from urllib.parse import quote
from uuid import UUID
from xml.etree import ElementTree

import requests
from idc_index import IDCClient
from unittest.mock import patch

from scripts.fresh_duke import acquire


class ByteBudget:
    def __init__(self, limit, used):
        self.limit, self.used = limit, used
        self.lock = Lock()

    def claim(self, size):
        with self.lock:
            if self.used + size > self.limit:
                raise RuntimeError('The next object would exceed the approved budget')
            self.used += size

    def release_unused(self, size):
        with self.lock:
            self.used -= size


class ResumeIDC:
    def __init__(self, order, previous, cache, budget, object_pool, log):
        self.index = IDCClient().index
        self.order = order.set_index('SeriesInstanceUID')
        self.catalogue = self.index.loc[self.index.SeriesInstanceUID.isin(self.order.index)].set_index(
            'SeriesInstanceUID', verify_integrity=True
        )
        self.previous, self.cache, self.budget, self.object_pool = previous, cache, budget, object_pool
        self.local = local()
        self.futures = {}
        self.log = log
        self.log_lock = Lock()

    def session(self):
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        return self.local.session

    def objects(self, uid):
        row = self.catalogue.loc[uid]
        expected = self.order.loc[uid]
        if (row.PatientID, row.StudyInstanceUID, row.Modality, row.SeriesDescription.strip().casefold()) != (
            expected.PatientID, expected.StudyInstanceUID, 'MR', 'ax t1'
        ) or row.collection_id != 'duke_breast_cancer_mri' or int(row.instanceCount) != int(expected.instanceCount):
            raise RuntimeError(f'Unexpected catalogue identity for {uid}')
        if row.aws_bucket not in {'idc-open-data', 'idc-open-data-cr'}:
            raise RuntimeError(f'Unexpected IDC bucket for {uid}')
        prefix = str(UUID(row.crdc_series_uuid)) + '/'
        endpoint = f'https://{row.aws_bucket}.s3.us-east-1.amazonaws.com'
        for attempt in range(3):
            try:
                response = self.session().get(endpoint, params={'list-type': 2, 'prefix': prefix}, timeout=(8, 20))
                response.raise_for_status()
                inventory = ElementTree.fromstring(response.content)
                break
            except (requests.RequestException, ElementTree.ParseError) as exc:
                if attempt == 2:
                    raise RuntimeError(f'IDC object inventory unavailable for {uid}') from exc
                sleep(attempt + 1)
        entries = []
        if inventory.findtext('{*}IsTruncated') != 'false':
            raise RuntimeError(f'Unexpected paginated object inventory for {uid}')
        for item in inventory.findall('{*}Contents'):
            key = item.findtext('{*}Key')
            size = int(item.findtext('{*}Size'))
            if not key.startswith(prefix) or size <= 0:
                raise RuntimeError(f'Unexpected object key or size for {uid}')
            name = key.removeprefix(prefix)
            if str(UUID(Path(name).stem)) + '.dcm' != name:
                raise RuntimeError(f'Unexpected object filename for {uid}')
            entries.append((name, f'{endpoint}/{quote(key)}', size, item.findtext('{*}ETag')))
        if len(entries) != int(expected.instanceCount):
            raise RuntimeError(f'Object inventory count differs from the frozen index for {uid}')
        return entries

    def get_object(self, older, target, url, size, etag):
        if target.is_file() and target.stat().st_size == size:
            return 0
        if older.is_file() and older.stat().st_size == size:
            os.link(older, target)
            return 0
        temporary = target.with_suffix('.incomplete')
        for attempt in range(3):
            self.budget.claim(size)
            received = 0
            try:
                with self.session().get(url, headers={'If-Match': etag}, stream=True, timeout=(8, 20)) as response:
                    response.raise_for_status()
                    if int(response.headers.get('Content-Length', 0)) != size:
                        raise RuntimeError(f'IDC content length changed for {target.name}')
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(64 * 1024):
                            received += len(chunk)
                            stream.write(chunk)
                if received != size:
                    raise RuntimeError(f'Incomplete IDC object {target.name}')
                temporary.replace(target)
                return size
            except (requests.RequestException, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError(f'IDC file retrieval failed for {target.name}') from exc
                sleep(attempt + 1)
            finally:
                self.budget.release_unused(max(0, size - received))
        raise RuntimeError(f'Unable to retrieve {target.name}')

    def hydrate(self, uid):
        patient = self.order.loc[uid].PatientID
        target = self.cache / patient
        target.mkdir(exist_ok=True)
        older = self.previous / patient
        entries = self.objects(uid)
        futures = [self.object_pool.submit(self.get_object, older / name, target / name, url, size, etag)
                   for name, url, size, etag in entries]
        new_bytes = sum(future.result() for future in futures)
        with self.log_lock:
            print(json.dumps({'patient_id': patient, 'series_uid': uid, 'objects': len(entries),
                              'new_object_bytes': new_bytes, 'budget_bytes_used_or_reserved': self.budget.used}),
                  file=self.log, flush=True)
        return [target / name for name, _, _, _ in entries]

    def download_dicom_series(self, uid, destination, **_):
        future = self.futures.pop(uid, None)
        files = future.result() if future else self.hydrate(uid)
        for path in files:
            target = Path(destination) / path.name
            if not target.exists():
                os.link(path, target)


def run(plan_run, previous_id, acquisition_id, max_gb):
    plan, order = acquire.load_plan(plan_run)
    root = acquire.ROOT
    previous = root / 'data/raw_fresh_duke' / acquire.valid_id(previous_id)
    acquisition_id = acquire.valid_id(acquisition_id)
    cache = root / 'data/raw_fresh_duke' / f'{acquisition_id}_object_cache'
    cache.mkdir(exist_ok=True)
    records = root / 'reports/v21_fresh_duke' / f'{acquisition_id}_transport'
    records.mkdir(exist_ok=False)
    reused_bytes = sum(p.stat().st_size for directory in (previous, cache) for p in directory.rglob('*') if p.is_file())
    budget = ByteBudget(int(max_gb * 1e9), reused_bytes)
    scope = order[order.role.ne('unreleased')]
    core = scope[scope.role.eq('initial')]
    if budget.used > budget.limit or len(core) != plan['initial_patients'] or not previous.is_dir():
        raise ValueError('Incomplete metadata or budget for the approved resume')
    receipt = {'previous_acquisition': previous_id, 'acquisition': acquisition_id, 'plan': plan_run,
               'approved_byte_cap': budget.limit, 'bytes_present_at_start': reused_bytes,
               'max_prefetch_series': 5, 'parallel_object_requests': 24, 'complete': False}
    (records / 'README.md').write_text(
        'Generated by `python -m scripts.fresh_duke.resume_transport`. transport.json records the '
        'resume budget and result. objects.jsonl logs the exact public IDC objects by screened series. '
        'Only initial series are prefetched. Reserves enter the transport on demand from the frozen '
        'acquisition verifier. Earlier raw files are reused by size and checked by the DICOM verifier.\n',
        encoding='utf-8',
    )
    try:
        with (records / 'objects.jsonl').open('x', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=24) as objects:
            client = ResumeIDC(scope, previous, cache, budget, objects, log)
            series_pool = ThreadPoolExecutor(max_workers=5)
            try:
                client.futures.update({uid: series_pool.submit(client.hydrate, uid) for uid in core.SeriesInstanceUID})
                with patch('idc_index.IDCClient', return_value=client):
                    acquire.acquire(plan_run, acquisition_id, max_gb)
            finally:
                series_pool.shutdown(wait=True, cancel_futures=True)
        receipt['complete'] = True
    finally:
        receipt['accounted_bytes'] = budget.used
        (records / 'transport.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-run', required=True)
    parser.add_argument('--previous-id', required=True)
    parser.add_argument('--acquisition-id', required=True)
    parser.add_argument('--max-gb', type=float, required=True)
    args = parser.parse_args()
    run(args.plan_run, args.previous_id, args.acquisition_id, args.max_gb)
