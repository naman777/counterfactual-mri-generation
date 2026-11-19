"""Canonical source loading and patient-aware sampling; never reads test data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from scripts.v7_geometry.orientation import (
    apply_orientation, invert_orientation, orientation_plan,
)
from scripts.v8_generator_dev.io import ROOT, digest, project_path

LABELS = ['manufacturer_label', 'field_strength_label']


class CanonicalDataset(Dataset):
    """Apply header transforms lazily; verify membership and source bytes.

    Metadata must agree with the historical split's patient/SOP/label/EchoTime
    and path. A forged split label cannot grant training access to another
    patient. Loading does not clamp, interpolate, or infer flips from labels.
    """

    def __init__(self, view, split, root=ROOT):
        if split not in {'train', 'val'}:
            raise ValueError('Generator development permits train/val only')
        self.root = Path(root).resolve()
        self.view = project_path(view, self.root)
        info = json.loads((self.view / 'view.json').read_text(encoding='utf-8'))
        if info['format'] != 'canonical_axial_view_v1':
            raise ValueError('Unsupported canonical view format')
        self.split = split
        frames = {}
        for name in ('train', 'val'):
            path = self.view / 'splits' / f'{name}.csv'
            frame = pd.read_csv(path, dtype={'patient_id': str, 'sop_uid': str})
            required = ['patient_id', 'sop_uid', 'split', *LABELS, 'echo_time_norm',
                        'source_image_path', 'source_array_sha256',
                        'orientation_plan', 'source_iop']
            if frame.empty or set(required) - set(frame):
                raise ValueError('Missing/empty canonical manifest')
            if frame[required].isna().any().any() or not frame.split.eq(name).all():
                raise ValueError('Missing values or manifest split mismatch')
            if frame.sop_uid.duplicated().any():
                raise ValueError('Duplicate SOP in canonical split')
            original_path = self.root / 'data/processed/ax_t1_128/splits' / f'{name}.csv'
            original = pd.read_csv(original_path, dtype={'patient_id': str, 'sop_uid': str})
            if original.sop_uid.duplicated().any():
                raise ValueError('Duplicate SOP in historical split')
            original = original.set_index('sop_uid')
            if not frame.sop_uid.isin(original.index).all():
                raise ValueError('SOP is not a member of the original declared split')
            ordered = original.loc[frame.sop_uid]
            for column in ['patient_id', *LABELS]:
                if not np.array_equal(frame[column].to_numpy(), ordered[column].to_numpy()):
                    raise ValueError(f'Historical split mismatch: {column}')
            if not np.allclose(frame.echo_time_norm, ordered.echo_time_norm, rtol=0, atol=1e-12):
                raise ValueError('EchoTime changed in canonical manifest')
            for source, historical in zip(frame.source_image_path, ordered.image_path):
                if project_path(source, self.root) != project_path(historical, self.root):
                    raise ValueError('Canonical source is not the historical split source')
            if not frame[LABELS].isin([0, 1]).all().all():
                raise ValueError('Acquisition labels must be binary')
            if not np.isfinite(frame.echo_time_norm).all() or not frame.echo_time_norm.between(0, 1).all():
                raise ValueError('EchoTime conditioning must be finite in [0,1]')
            if (frame.groupby('patient_id')[LABELS].nunique() > 1).any().any():
                raise ValueError('Patient has inconsistent acquisition-cell labels')
            frames[name] = frame.reset_index(drop=True)
        if set(frames['train'].patient_id) & set(frames['val'].patient_id):
            raise ValueError('Train/validation patient overlap')
        if set(frames['train'].sop_uid) & set(frames['val'].sop_uid):
            raise ValueError('Train/validation SOP overlap')
        self.frame = frames[split]
        self.plans = []
        for row in self.frame.itertuples():
            plan = json.loads(row.orientation_plan)
            expected = orientation_plan(json.loads(row.source_iop), (128, 128))
            if plan != expected:
                raise ValueError('Canonical transform does not match recorded DICOM IOP')
            self.plans.append(plan)

    def __len__(self):
        return len(self.frame)

    def image_01(self, index):
        row = self.frame.iloc[index]
        path = project_path(row.source_image_path, self.root)
        if digest(path) != row.source_array_sha256:
            raise ValueError('Source array hash mismatch')
        source = np.load(path, allow_pickle=False)
        if source.shape != (128, 128) or not np.isfinite(source).all():
            raise ValueError('Invalid source shape/values')
        if source.min() < 0 or source.max() > 1:
            raise ValueError('Source outside documented [0,1] range')
        canonical = apply_orientation(source, self.plans[index])
        if not np.array_equal(source, invert_orientation(canonical, self.plans[index])):
            raise AssertionError('Lossless source round trip failed')
        return canonical.astype(np.float32, copy=False)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            'image': torch.from_numpy(self.image_01(index).copy()).unsqueeze(0) * 2 - 1,
            'manufacturer_label': int(row.manufacturer_label),
            'field_strength_label': int(row.field_strength_label),
            'echo_time_norm': np.float32(row.echo_time_norm),
            'patient_id': row.patient_id, 'index': int(index),
        }


def sampling_probabilities(frame, min_cell_patients=5):
    """Balance supported cells and patients within cells; retain sparse cells.

    Sparse-cell patients retain their 1/N patient probability. The remaining
    mass is spread equally over supported cells, then patients, then slices.
    This never duplicates a patient to meet an artificial support threshold.
    """
    if frame.empty or min_cell_patients < 1:
        raise ValueError('Nonempty data and positive patient threshold required')
    if (frame.groupby('patient_id')[LABELS].nunique() > 1).any().any():
        raise ValueError('Patient labels change between slices')
    patients = frame.drop_duplicates('patient_id')
    cells = patients.groupby(LABELS).size()
    supported = cells[cells >= min_cell_patients]
    supported_patients = int(supported.sum())
    counts = frame.groupby('patient_id').size()
    weights = []
    for row in frame.itertuples():
        cell = (row.manufacturer_label, row.field_strength_label)
        if cell in supported.index:
            patient_mass = supported_patients / (len(patients) * len(supported) * cells.loc[cell])
        else:
            patient_mass = 1 / len(patients)
        weights.append(patient_mass / counts.loc[row.patient_id])
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights.sum(), 1, atol=1e-12) or not (weights > 0).all():
        raise AssertionError('Invalid sampling probabilities')
    return weights


class PatientCellSampler(Sampler):
    def __init__(self, dataset, seed, num_samples=None, min_cell_patients=5):
        if dataset.split != 'train':
            raise ValueError('Weighted sampling is restricted to training')
        self.weights = torch.from_numpy(sampling_probabilities(dataset.frame, min_cell_patients))
        self.seed, self.epoch = int(seed), 0
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples < 1:
            raise ValueError('num_samples must be positive')

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True,
                                      generator=generator).tolist())

    def __len__(self):
        return self.num_samples


def validation_indices(frame, seed):
    """One fixed slice per validation patient, no generated-output selection."""
    return frame.sample(frac=1, random_state=seed).drop_duplicates('patient_id').index.tolist()
