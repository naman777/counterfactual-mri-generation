# MR-one local API

From `counterfactual-mri-generation/`, using the system Python with the existing
scientific dependencies:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `/docs` on the API for its schema. The Next.js app proxies `/api/*` to this
loopback server. No cloud calls, downloads or modifications to research assets.

`python -m unittest api.test_contract -v` checks data/API contracts, source-metadata
validation, real report formats and exact-float patient ZIP contents.
`python -m api.smoke` performs a real epoch-110 GPU edit, compares it with saved
B3 example 000, loads the four classifiers and writes `api/runtime/smoke.json`.
Cross-device floating-point differences are expected; the compatibility gate is
SSIM > 0.999, not bitwise equality. Intermediate streamed frames are noisy states.
With both servers running, `python -m api.verify_runtime` exercises actual live
edits on both checkpoints, null editing, CPU probes, DICOM preprocessing, exports
and source-backed routes through the Next.js proxy. Receipts are in `api/runtime/`.

Both diffusion models are loaded lazily and retained. The converged loader
explicitly chooses `ema_*_state_dict`; development falls back to the raw keys.
Sampling uses B3's independent per-slice generator and full ancestral DDPM steps.
CPU classifiers and GPU edits share an inference lock. Run **one API worker**.

Saved B3 caching requires an exact match for slice, target, model, t, w and seed.
The response identifies saved evidence versus live inference. Session results
are bounded to 32 entries; batch archives to eight jobs. These caches are temporary.
Restarting the API expires report and batch links. A batch currently packages saved
evaluation images only; it does not launch a whole-patient diffusion job.

The persistent `api/demo_bundle/` is an exception to session expiry: 20 selected
slices have complete results for both paper presets and matching null edits.
`python -m api.build_demo` creates it from saved B3 arrays and CPU probabilities,
with denoising explicitly disabled. Exact-request replay returns before model
loading, checkpoint checks or inference. Its report cards also survive restarts.
It is a frozen snapshot, not a cache for newly generated edits. Changing the
checkpoint or the research protocol requires a separately versioned bundle.
`start_demo.bat` sets `MRONE_DEMO_ONLY=1`; uncached requests then return 409.
The slice catalogue exposes per-axis paper seeds and the selected demo IDs.
Histograms use a shared source foreground mask (intensity >0.01); other image
metrics retain their existing full-image definition. UI display labels and
rounding do not change stored CSV values.

`python -m api.verify_demo --output reports/<new_folder>` exercises all 80 saved
requests through the Next.js proxy. `python -m api.package_demo --output
demo/<new_archive>.zip` makes a transfer archive without diffusion checkpoints.

Uploads accept a single 2D DICOM or PNG (10 MB maximum), reuse the existing
preprocessing, and live in a bounded memory-only store (32 sources). DICOM source
conditions come from metadata; PNG requires user-supplied source conditions and
EchoTime. Neither route can certify in-distribution status: warnings explicitly
mark unknown provenance, sequence/orientation concerns and out-of-range EchoTime.
Raw DICOM headers and patient identifiers are not persisted or returned.

`/report/{id}?format=png|pdf|json` exports a deterministic evidence card or complete
metrics. `/paper` serves the existing local v5 IEEE PDF. Robustness probes use CPU
only and show source-label probabilities under seeded noise and/or vertical flip.

Data sources: the test split CSV; B3 `remote/full/test/*/{images.npz,metrics.csv}`;
B2 `regrade_b3/{summary,per_slice}.csv` and `regrade_old/summary.csv`; B3 analysis
CSV files; and held-out seed-variance results, as listed in the v5 numbers registry.
Missing inputs fail visibly. No synthetic fallback metrics are supplied.

This is a trusted, local research prototype. Keep it on loopback; it has no remote
authentication or persistent job service. Do not expose the API on a public host.
