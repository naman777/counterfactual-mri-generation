#!/usr/bin/env python3
"""
P0.1 — Step 1: Create segmentation masks from bounding box annotations.

The Duke Breast MRI dataset provides tumor bounding box annotations
in the original NIFTI/DICOM space (512x512 resolution, 3D volume indices).

This script:
1. Reads segmentation_annotations_NIFTI.csv (Start/End Row/Col/Slice)
2. For each preprocessed 128x128 slice, determines if it falls within the
   annotated tumor slice range
3. Scales the bounding box from 512x512 → 128x128 (4x downscale)
4. Creates a binary foreground mask: 1 inside bbox, 0 outside
5. Saves masks to data/processed/ax_t1_128/seg_masks/ as .npy

Output mask values:
    0 = background
    1 = foreground (tumor region bounding box)

Note: These are bounding box masks (not pixel-accurate segmentations).
This is disclosed in the paper as "ROI-based segmentation using tumor
bounding box annotations provided with the Duke dataset."

Usage:
    python scripts/create_segmentation_masks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ORIG_DIM   = 512   # original DICOM spatial resolution
TARGET_DIM = 128   # our preprocessed resolution
SCALE      = TARGET_DIM / ORIG_DIM  # 0.25

BBOX_CSV    = PROJECT_ROOT / "data/full/archive/segmentation_annotations_NIFTI.csv"
META_CSV    = PROJECT_ROOT / "data/processed/ax_t1_128/processed_metadata.csv"
MASKS_DIR   = PROJECT_ROOT / "data/processed/ax_t1_128/seg_masks"


def scale_bbox(r0, r1, c0, c1, scale=SCALE, target=TARGET_DIM):
    """Scale bounding box from original space to target resolution."""
    r0_s = int(np.floor(r0 * scale))
    r1_s = int(np.ceil(r1 * scale))
    c0_s = int(np.floor(c0 * scale))
    c1_s = int(np.ceil(c1 * scale))
    # Clamp to [0, target)
    r0_s = max(0, min(r0_s, target - 1))
    r1_s = max(0, min(r1_s, target))
    c0_s = max(0, min(c0_s, target - 1))
    c1_s = max(0, min(c1_s, target))
    return r0_s, r1_s, c0_s, c1_s


def main():
    print("=" * 60)
    print("Creating segmentation masks from bounding box annotations")
    print("=" * 60)

    # Load annotations and metadata
    bbox_df = pd.read_csv(BBOX_CSV)
    meta_df = pd.read_csv(META_CSV)
    bbox_map = {row["Patient_ID"]: row for _, row in bbox_df.iterrows()}

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    in_bbox = 0
    out_bbox = 0
    no_annot = 0

    for _, row in meta_df.iterrows():
        pid = row["patient_id"]
        img_path = PROJECT_ROOT / row["image_path"]
        instance_num = int(row["instance_number"])  # 1-indexed slice number

        # Determine mask output path (same filename pattern as image)
        img_filename = Path(row["image_path"]).name
        mask_path = MASKS_DIR / img_filename

        if mask_path.exists():
            total += 1
            continue

        mask = np.zeros((TARGET_DIM, TARGET_DIM), dtype=np.uint8)

        if pid in bbox_map:
            ann = bbox_map[pid]
            start_slice = int(ann["Start Slice"])  # 1-indexed in NIFTI space
            end_slice   = int(ann["End Slice"])

            # Check if this slice falls within the annotated volume
            # instance_number is the DICOM slice index (1-indexed)
            if start_slice <= instance_num <= end_slice:
                r0, r1 = int(ann["Start Row"]), int(ann["End Row"])
                c0, c1 = int(ann["Start Column"]), int(ann["End Column"])
                r0_s, r1_s, c0_s, c1_s = scale_bbox(r0, r1, c0, c1)
                mask[r0_s:r1_s, c0_s:c1_s] = 1
                in_bbox += 1
            else:
                out_bbox += 1
        else:
            no_annot += 1

        np.save(mask_path, mask)
        total += 1

        if total % 500 == 0:
            print(f"  Processed {total}/{len(meta_df)} slices...")

    print(f"\nDone!")
    print(f"  Total slices processed:        {total}")
    print(f"  Slices WITH foreground mask:   {in_bbox}")
    print(f"  Slices OUTSIDE bbox range:     {out_bbox}")
    print(f"  Slices with no annotation:     {no_annot}")
    print(f"  Masks saved to: {MASKS_DIR}")

    # Quick sanity check
    mask_files = list(MASKS_DIR.glob("*.npy"))
    print(f"\n  Mask files on disk: {len(mask_files)}")
    if mask_files:
        sample = np.load(mask_files[0])
        print(f"  Sample mask shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"  Sample foreground fraction: {sample.mean():.4f}")


if __name__ == "__main__":
    main()
