# MR-one local API

From `counterfactual-mri-generation/`, using the system Python with the existing
scientific dependencies:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `/docs` on the API for its schema. The Next.js app proxies `/api/*` to this
loopback server. No cloud calls, downloads or modifications to research assets.

`python -m unittest api.test_contract -v` checks read-only data/API contracts.
`python -m api.smoke` performs a real epoch-110 GPU edit, compares it with saved
B3 example 000, loads the four classifiers and writes `api/runtime/smoke.json`.
Cross-device floating-point differences are expected; the compatibility gate is
SSIM > 0.999, not bitwise equality. Intermediate streamed frames are noisy states.

Both diffusion models are loaded lazily and retained. The converged loader
explicitly chooses `ema_*_state_dict`; development falls back to the raw keys.
Sampling uses B3's independent per-slice generator and full ancestral DDPM steps.
CPU classifiers and GPU edits share an inference lock. Run **one API worker**.

Saved B3 caching requires an exact match for slice, target, model, t, w and seed.
The response identifies saved evidence versus live inference. Session results
are bounded to 32 entries; batch archives to eight jobs. These caches are temporary.
Restarting the API expires report and batch links. A batch currently packages saved
evaluation images only; it does not launch a whole-patient diffusion job.

Data sources: the test split CSV; B3 `remote/full/test/*/{images.npz,metrics.csv}`;
B2 `regrade_b3/{summary,per_slice}.csv` and `regrade_old/summary.csv`; B3 analysis
CSV files; and held-out seed-variance results, as listed in the v5 numbers registry.
Missing inputs fail visibly. No synthetic fallback metrics are supplied.

This is a trusted, local research prototype. Keep it on loopback; it has no remote
authentication or persistent job service. Do not expose the API on a public host.
