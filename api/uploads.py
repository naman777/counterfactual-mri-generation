"""Bounded, memory-only uploads. Raw DICOM metadata is never returned or persisted."""
import io
import json
import uuid
from collections import OrderedDict

import numpy as np
import pydicom
from PIL import Image

from api.data import ROOT
from src.data.preprocess import (
    apply_rescale_if_present, normalize_manufacturer, normalize_field_strength,
    percentile_clip_and_normalize, resize_image,
)

UPLOADS = OrderedDict()


def preprocess_upload(content, filename, manufacturer=None, field_strength=None, echo_ms=None):
    warnings = ["Uploaded image is outside the verified evaluation cohort. Model validity is unknown."]
    if filename.lower().endswith(".png"):
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.width * image.height > 16_000_000:
                raise ValueError("Use a PNG smaller than 16 megapixels")
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
        if manufacturer not in (0, 1) or field_strength not in (0, 1) or echo_ms is None:
            raise ValueError("PNG needs source manufacturer, field strength and EchoTime in milliseconds")
        warnings.append("PNG has no acquisition metadata. Source conditions are user supplied; axial T1 and orientation cannot be verified.")
    elif filename.lower().endswith((".dcm", ".dicom")):
        ds = pydicom.dcmread(io.BytesIO(content))
        pixels = apply_rescale_if_present(ds, ds.pixel_array.astype(np.float32))
        mfg = normalize_manufacturer(getattr(ds, "Manufacturer", ""))
        fs = normalize_field_strength(getattr(ds, "MagneticFieldStrength", None))
        if mfg not in ("GE MEDICAL SYSTEMS", "SIEMENS") or fs not in ("1.5T", "3.0T"):
            raise ValueError("DICOM source must have supported GE/Siemens and 1.5T/3.0T metadata")
        manufacturer = 0 if mfg == "GE MEDICAL SYSTEMS" else 1
        field_strength = 0 if fs == "1.5T" else 1
        echo_ms = float(getattr(ds, "EchoTime", echo_ms if echo_ms is not None else float("nan")))
        description = str(getattr(ds, "SeriesDescription", "")).lower().strip()
        if description != "ax t1":
            warnings.append("DICOM series is not labelled 'ax t1', the model's training domain.")
        warnings.append("DICOM display orientation is preserved; vendor-linked orientation is a known shortcut risk.")
    else:
        raise ValueError("Only single-slice DICOM (.dcm/.dicom) and PNG files are supported")
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("Upload must be a finite, single two-dimensional image")
    if echo_ms is None or not np.isfinite(echo_ms) or echo_ms <= 0:
        raise ValueError("A finite positive source EchoTime is required")
    stats = json.loads((ROOT / "data/processed/ax_t1_128/label_maps/echo_time_stats.json").read_text())
    echo_norm = (echo_ms - stats["min"]) / (stats["max"] - stats["min"])
    if not 0 <= echo_norm <= 1:
        warnings.append("Source EchoTime is outside the training range. It remains fixed and is not clipped.")
    image = resize_image(percentile_clip_and_normalize(pixels), 128)
    source = dict(id="upload_" + uuid.uuid4().hex, patient="Local upload", manufacturer=manufacturer,
                  field_strength=field_strength, echo=float(echo_norm), echo_ms=float(echo_ms), instance=1,
                  source="Local upload / percentile clipping and normalisation / 128 x 128", warnings=warnings)
    UPLOADS[source["id"]] = (source, image)
    while len(UPLOADS) > 32:
        UPLOADS.popitem(last=False)
    return source
