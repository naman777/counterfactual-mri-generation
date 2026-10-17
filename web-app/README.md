# MR-one Counterfactual Workbench

Local Next.js App Router + MUI frontend for breast-MRI counterfactual research.
The FastAPI adapter in `../api/` reads existing data/models/results without changing
them. Research use only; not for diagnosis.

## Run on the laptop

For the panel demo, run `start_demo.bat -Live` from the code root. This opens the
Overview and supports uncached requests alongside saved results. Warm the model
with one live edit before presenting. The recommended Studio opener is Patient
880, slice 25 (3/3 held-out judges at the field-strength paper preset). Plain
`start_demo.bat` remains the saved-replay fallback if the GPU fails. Stop the
running API before switching modes. See `DEMO.md` for the timed script and checklist.

Terminal 1, from `counterfactual-mri-generation/`, using the system Python:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, from `counterfactual-mri-generation/web-app/`:

```powershell
pnpm install --frozen-lockfile
pnpm dev --hostname 127.0.0.1
```

Open http://localhost:3000. The app proxies `/api/*` to the local Python service;
set `MRONE_API_URL` before starting Next.js to change that destination. There are
no external services or credentials to configure. Keep the API on loopback.

## Pages

- `/`: Overview with problem, research users, pipeline and source-backed rates/CIs.
- `/studio`: Studio with manifest filters, PNG/DICOM input, fixed source EchoTime,
  epoch-110 EMA / epoch-10 selector, live denoising frames, maps, histogram,
  wipe comparison, four classifier panels and same-noise null comparison.
- `/evidence`: all saved B3 evaluation sets/variants, patient/direction/verdict
  filters and source-file provenance. No diffusion inference required.
- `/results`: saved hit rates with patient-clustered CIs, epoch comparison,
  per-slice SSIM vs gain, generation/judge-seed variation, HF energy and raw records.
- `/robustness`: deterministic Gaussian-noise and vertical-flip probes, CPU only.
- `/batch`: ZIP of saved evaluation images, full-precision arrays and manifests,
  filtered by patient and judge agreement.
- `/about`: intended use, limitations, team, checkpoints and the local IEEE paper.

In live mode, turn off **Use saved results when available** to force diffusion. Otherwise
an exact matching B3 edit or session cache may be used, with a visible chip. The
cache never labels playback as live denoising. A null run uses the completed edit's
source, t, w, seed and model, even if the controls have since changed.

## Inputs needed

The existing test split and `.npy` images; both diffusion checkpoints; original
predictor; frozen B2 held-out checkpoints (plus legacy seed 0 for the robustness
control); B3 image packs/metrics, B2 re-grades and analysis CSVs. See `api/data.py`
for exact locations and `../api/README.md` for API behaviour. Research assets stay
outside this dashboard's commits. A fresh clone needs those inputs provisioned.

## Checks

```powershell
# From web-app/
pnpm exec tsc --noEmit
pnpm build
# From the code root
python -m unittest api.test_contract api.test_dashboard_fixes -v
python -m api.smoke
```

MUI uses the Next.js 16 App Router cache provider for streaming-safe Emotion SSR.
The interface uses responsive CSS and real MRI bitmaps, with no generated or mock
results. `DEMO.md` provides a short walkthrough. `../DASHBOARD_PROGRESS.md` tracks
implementation, verification, commits and remaining work.

## Current limits

Batch exports contain the selected patient's **evaluation slices**, not the whole
patient's test series. Whole-patient live generation and persistent queues/cache
remain planned. Upload warnings are metadata/domain checks, not a validated OOD
detector. Uploads, report links and batch jobs expire on API restart. There is no
authentication, deployment, Groq integration or database history in this version.
Demo recording and a second-machine rehearsal remain to be done.
