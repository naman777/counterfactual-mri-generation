"""Run from the code root: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000."""
import asyncio
import hashlib
import io
import json
import logging
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api import data
from api import demo_cache
from api.publication import PAPER_PATH, PAPER_DOWNLOAD_NAME
from api.metrics import analyse, verdict
from api.models import INFERENCE_LOCK, MODEL_PATHS, classifiers, editor, probabilities, sample

app = FastAPI(title="MR-one research workbench", version="0.1.0")
LOG = logging.getLogger("mrone")
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mrone")
CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
BATCHES = {}


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slice_id: str
    manufacturer: Literal[0, 1]
    field_strength: Literal[0, 1]
    timestep: int = Field(100, ge=50, le=500)
    guidance: float = Field(5., ge=0, le=7, allow_inf_nan=False)
    seed: int = Field(42, ge=0, le=2147483647)
    model: Literal["converged", "development"] = "converged"
    null_edit: bool = False
    use_cache: bool = True


def require_slice(slice_id):
    try:
        return data.slice_info(slice_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/health")
def health():
    return dict(status="ok", device="cuda" if torch.cuda.is_available() else "cpu",
                slices=len(data.slices()), patients=len({s["patient"] for s in data.slices()}),
                models={k: p.exists() for k, p in MODEL_PATHS.items()}, evidence_sets=len(data.evidence_sets()),
                research_only=True, demo_only=demo_cache.demo_only(), demo_slices=len(demo_cache.manifest()["slices"]))


@app.get("/paper")
def paper():
    path = PAPER_PATH
    if not path.exists():
        raise HTTPException(404, "The current paper PDF is not available on this machine")
    return FileResponse(path, media_type="application/pdf", filename=PAPER_DOWNLOAD_NAME)


@app.get("/slices")
def slices(patient: str | None = None, manufacturer: int | None = None, field_strength: int | None = None):
    items = [s for s in data.slices() if (patient is None or s["patient"] == patient)
             and (manufacturer is None or s["manufacturer"] == manufacturer)
             and (field_strength is None or s["field_strength"] == field_strength)]
    recommended = str(data.pack("main_field_strength_t100_w5_s0")["indices"][0]) if data.evidence_sets() else (items[0]["id"] if items else None)
    demo_ids = [s["id"] for s in demo_cache.manifest()["slices"]]
    if demo_ids:
        recommended = demo_cache.recommended_id()
    return dict(items=items, total=len(items), patients=sorted({s["patient"] for s in data.slices()}), recommended_id=recommended,
                demo_ids=demo_ids, demo_only=demo_cache.demo_only())


@app.get("/slice/{slice_id}")
def slice_detail(slice_id: str):
    return require_slice(slice_id)


@app.get("/slice/{slice_id}/image")
def slice_image(slice_id: str):
    require_slice(slice_id)
    return Response(data.png_bytes(data.source_image(slice_id)), media_type="image/png")


def saved_edit(req, source):
    if req.model != "converged" or not req.slice_id.isdigit():
        return None
    changed = [k for k in ("manufacturer", "field_strength") if getattr(req, k) != source[k]]
    if len(changed) != 1:
        return None
    axis = changed[0]
    for name in data.evidence_sets():
        if not name.startswith(f"{'soundness' if req.null_edit else 'main'}_{axis}_t{req.timestep}_w{req.guidance:g}_s"):
            continue
        z = data.pack(name)
        match = np.where(z["indices"] == int(req.slice_id))[0]
        if len(match) and req.seed == 42 + int(match[0]) + int(name.rsplit("_s", 1)[1]):
            return z["null1" if req.null_edit else "cf"][int(match[0])], data.relative(data.B3 / name / "images.npz")
    return None


def run_edit(req, emit=lambda event: None):
    replay = demo_cache.lookup(req) if req.use_cache else None
    if replay:
        with CACHE_LOCK:
            CACHE[replay["id"]] = replay
            while len(CACHE) > 32:
                CACHE.popitem(last=False)
        return replay
    if demo_cache.demo_only():
        raise ValueError("Replay mode: choose a saved demo slice with an epoch-110 paper preset and its paper seed. Live inference is disabled.")
    source = data.slice_info(req.slice_id)
    before = data.source_image(req.slice_id)
    target_mfg = source["manufacturer"] if req.null_edit else req.manufacturer
    target_fs = source["field_strength"] if req.null_edit else req.field_strength
    changed = [k for k, value in [("mfg", target_mfg), ("fs", target_fs)]
               if value != source["manufacturer" if k == "mfg" else "field_strength"]]
    fingerprint = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in
                   [MODEL_PATHS[req.model], data.ROOT / "checkpoints/predictor/ax_t1_128/best.pt"]]
    key = hashlib.sha256(json.dumps([req.model_dump(exclude={"use_cache"}), fingerprint], sort_keys=True).encode()).hexdigest()
    with CACHE_LOCK:
        if req.use_cache and key in CACHE:
            CACHE.move_to_end(key)
            return {**CACHE[key], "cached": True, "cache_kind": "session result"}
    started = time.perf_counter()
    emit(dict(type="status", message="Waiting for the local inference worker"))
    with INFERENCE_LOCK:
        saved = saved_edit(req, source) if req.use_cache else None
        if saved:
            after, provenance = saved
            cache_kind = "saved B3 evaluation"
        else:
            emit(dict(type="status", message="Loading checkpoint and fixed source conditioning"))
            model = editor(req.model)
            emit(dict(type="status", message=f"Denoising on {model.device}; EchoTime fixed"))
            def progress(t, image):
                emit(dict(type="frame", timestep=t, progress=(req.timestep-t+1)/(req.timestep+1), image=data.image_url(image)))
            after = sample(model, before, target_mfg, target_fs, source["echo"], req.timestep, req.guidance, req.seed, progress)
            provenance = data.relative(MODEL_PATHS[req.model])
            cache_kind = None
        emit(dict(type="status", message="Computing independent judge probabilities and explanation maps"))
        judges = probabilities([before, after])
    metrics = analyse(before, after)
    result = dict(id=key, request=req.model_dump(), source=source, target=dict(manufacturer=target_mfg, field_strength=target_fs, echo=source["echo"]),
                  metrics=metrics, judges=judges, verdict=verdict(metrics["ssim"], judges, target_mfg, target_fs, changed), changed=changed,
                  cached=saved is not None, cache_kind=cache_kind, seconds=time.perf_counter()-started,
                  weights="EMA" if req.model == "converged" else "raw", provenance=provenance,
                  judge_provenance=[data.relative(data.B2 / f"training/orientation_robust_seed{i}/judge_heldout_resnet18_seed{i}.pt") for i in range(3)],
                  research_only=True)
    with CACHE_LOCK:
        CACHE[key] = result
        while len(CACHE) > 32:
            CACHE.popitem(last=False)
    return result


@app.post("/upload")
async def upload(file: UploadFile = File(...), manufacturer: int | None = Form(None),
                 field_strength: int | None = Form(None), echo_ms: float | None = Form(None)):
    content = await file.read(10 * 1024 * 1024 + 1)
    await file.close()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Upload limit is 10 MB")
    try:
        from api.uploads import preprocess_upload
        return preprocess_upload(content, file.filename or "", manufacturer, field_strength, echo_ms)
    except Exception as exc:
        LOG.info("Upload rejected: %s", exc)
        raise HTTPException(422, "Could not preprocess this source: " + str(exc)) from exc


@app.post("/edit")
async def edit(req: EditRequest):
    require_slice(req.slice_id)
    if demo_cache.demo_only() and (not req.use_cache or demo_cache.lookup(req) is None):
        raise HTTPException(409, "Replay mode: choose a saved demo slice, epoch 110, a paper preset and its paper seed. Live inference is disabled.")
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=32)
    def emit(event):
        # One worker, at most 15 frames; bounded queue also survives disconnects.
        def put():
            if not queue.full():
                queue.put_nowait(event)
        loop.call_soon_threadsafe(put)
    async def events():
        future = loop.run_in_executor(EXECUTOR, run_edit, req, emit)
        while not future.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=.5)
                yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
        try:
            result = await future
            yield "data: " + json.dumps(dict(type="result", result=result), allow_nan=False) + "\n\n"
        except Exception:
            LOG.exception("Edit failed")
            yield "data: " + json.dumps(dict(type="error", message="Local edit failed. Check the API log and checkpoint availability; no result was substituted.")) + "\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/null-edit")
async def null_edit(req: EditRequest):
    return await edit(req.model_copy(update={"null_edit": True}))


@app.post("/judge")
def judge(req: EditRequest):
    require_slice(req.slice_id)
    with INFERENCE_LOCK:
        return dict(judges=probabilities([data.source_image(req.slice_id)]))


class RobustnessRequest(BaseModel):
    slice_id: str
    sigma: float = Field(0, ge=0, le=.05, allow_inf_nan=False)
    vertical_flip: bool = False
    seed: int = Field(42, ge=0, le=2147483647)


@app.post("/robustness")
def robustness(req: RobustnessRequest):
    require_slice(req.slice_id)
    before = data.source_image(req.slice_id)
    after = np.clip(before + np.random.default_rng(req.seed).normal(0, req.sigma, before.shape), 0, 1).astype(np.float32)
    if req.vertical_flip:
        after = after[::-1].copy()
    with INFERENCE_LOCK:
        judges = probabilities([before, after], legacy=True)
    return dict(judges=judges, source=data.image_url(before), perturbed=data.image_url(after), request=req.model_dump(),
                note="Single-slice CPU probe with source labels fixed; this is not a cohort hit-rate estimate.")


@app.get("/evidence")
def evidence():
    return dict(sets=[dict(name=s, variants=[k for k, v in data.pack(s).items() if v.ndim == 3 and k != "source_images"], n=len(data.pack(s)["indices"])) for s in data.evidence_sets()])


@app.get("/evidence/{set_name}")
def evidence_set(set_name: str, variant: str = "cf"):
    try:
        return dict(items=data.evidence_rows(set_name, variant))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/evidence/{set_name}/{index}/image")
def evidence_image(set_name: str, index: int, variant: str = "cf"):
    try:
        z = data.pack(set_name)
        if variant not in z or z[variant].ndim != 3 or not 0 <= index < len(z[variant]):
            raise ValueError("Unknown evidence image")
        return Response(data.png_bytes(z[variant][index]), media_type="image/png")
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/results/{axis}")
def results(axis: Literal["field_strength", "manufacturer"]):
    return dict(items=data.headline_results(axis), interval="95% patient-clustered bootstrap CI",
                registry="research-paper-v5/numbers.md", caveat="Development manufacturer uses t=300 w=5; converged manufacturer uses t=300 w=3. These are different operating points.")


@app.get("/results-data/{chart}")
def result_data(chart: Literal["tradeoff", "variance", "high-frequency"]):
    path = {"tradeoff": data.ANALYSIS / "quality_scores.csv", "variance": data.ROOT / "reports/v4_stats_heldout_20261016_01/seed_variance_heldout.csv", "high-frequency": data.ANALYSIS / "hf_energy_ci.csv"}[chart]
    return dict(items=data.csv_rows(path), source=data.relative(path))


@app.get("/tradeoff/{axis}")
def tradeoff(axis: Literal["field_strength", "manufacturer"]):
    return dict(items=data.tradeoff(axis))


class BatchRequest(BaseModel):
    set_name: str = "main_field_strength_t100_w5_s0"
    variant: str = "cf"
    patient: str | None = None
    min_judges: int = Field(0, ge=0, le=3)


def make_batch(job_id, req):
    try:
        z = data.pack(req.set_name)
        rows = [r for r in data.evidence_rows(req.set_name, req.variant)
                if (req.patient is None or r["slice"]["patient"] == req.patient) and r["confirmed"] >= req.min_judges]
        if not rows:
            raise ValueError("No saved evaluation slices match this selection")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for r in rows:
                name = f"{r['slice']['patient']}/example_{r['index']:03d}"
                arr = z[req.variant][r["index"]]
                buffer = io.BytesIO()
                np.save(buffer, arr, allow_pickle=False)
                archive.writestr(name + ".npy", buffer.getvalue())
                archive.writestr(name + ".png", data.png_bytes(arr))
                archive.writestr(name + ".json", json.dumps(r, indent=2))
            job = json.loads((data.B3 / req.set_name / "job.json").read_text())
            archive.writestr("manifest.json", json.dumps(dict(research_only=True, mode="precomputed B3 evaluation export", request=req.model_dump(), generation=job, count=len(rows), slices=rows), indent=2))
        BATCHES[job_id].update(status="complete", count=len(rows), content=output.getvalue())
    except Exception as exc:
        LOG.exception("Batch export failed")
        BATCHES[job_id].update(status="failed", error=str(exc))


@app.post("/batch")
def batch(req: BatchRequest):
    if req.set_name not in data.evidence_sets():
        raise HTTPException(404, "Unknown evaluation set")
    # Bound memory use in this local prototype.
    if len(BATCHES) >= 8:
        oldest = next((k for k, v in BATCHES.items() if v["status"] != "queued"), None)
        if oldest is None:
            raise HTTPException(429, "Export queue is full")
        del BATCHES[oldest]
    job_id = uuid.uuid4().hex
    BATCHES[job_id] = dict(status="queued", id=job_id)
    EXECUTOR.submit(make_batch, job_id, req)
    return dict(id=job_id, status="queued")


@app.get("/batch/{job_id}")
def batch_status(job_id: str):
    if job_id not in BATCHES:
        raise HTTPException(404, "Export not found (jobs are session-local)")
    return {k: v for k, v in BATCHES[job_id].items() if k != "content"}


@app.get("/batch/{job_id}/download")
def batch_download(job_id: str):
    status = batch_status(job_id)
    if status["status"] != "complete":
        raise HTTPException(409, "Export is not ready")
    return Response(BATCHES[job_id]["content"], media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="mrone-evidence.zip"'})


@app.get("/report/{result_id}")
def report(result_id: str, format: Literal["json", "png", "pdf"] = "json"):
    with CACHE_LOCK:
        result = CACHE.get(result_id)
    if result is None:
        filename = demo_cache.manifest().get("result_ids", {}).get(result_id)
        if filename:
            result = json.loads((demo_cache.CACHE_DIR / filename).read_text(encoding="utf-8"))
        else:
            raise HTTPException(404, "Result expired; rerun the edit")
    if format == "json":
        content, mime = json.dumps(result, indent=2, allow_nan=False), "application/json"
    else:
        from api.report import report_card
        content, mime = report_card(result, format), "application/pdf" if format == "pdf" else "image/png"
    return Response(content, media_type=mime, headers={"Content-Disposition": f'attachment; filename="mrone-report.{format}"'})
