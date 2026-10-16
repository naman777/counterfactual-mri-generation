# MR-one dashboard progress

Started 2026-10-16. This is the implementation log for the six-page workbench plan.
Research outputs and checkpoints are read-only inputs. EchoTime stays fixed.

## Decisions

- Next.js App Router + MUI, FastAPI, Recharts; local laptop deployment first.
- Converged epoch 110 / EMA is the default; development epoch 10 remains selectable.
- Reproduce B3's per-slice RNG sampler, not the legacy editor's global RNG.
- Cached evidence is labelled explicitly. Missing data produces an error, never mock values.
- Keep local commits scoped to dashboard changes; preserve existing work in the repository.
- No cloud spending or publication in this implementation session.

## Checklist

- [x] Phase 0: EMA-aware backend, edit/judge endpoints and B3 reproduction smoke check. [`api/`]
- [x] Phase 1: Studio, navigation, maps, streamed progress and verification. [`web-app/src/`]
- [x] Phase 2: precomputed evidence and source-backed headline charts. [`api/data.py`, `web-app/src/components/research-charts.tsx`]
- [x] Phase 3 implementation: CPU robustness probes, same-noise null edit and PNG/PDF/JSON report exports. [`api/`, `web-app/src/components/`]
- [x] Phase 4 first increment: DICOM/PNG upload with explicit domain warnings, saved-patient batch ZIP and model card.
- [ ] Phase 4 remaining: queue **new live** edits for all slices of a patient (current exports use saved evaluation slices).
- [ ] Phase 5: curated cache, demo script, backup recording and second-machine rehearsal.

## Verification / commits

- Initial inspection: the code folder already has Git and an origin remote. Existing
  unrelated modified/untracked files will be left out of dashboard commits.
- Confirmed local models, precomputed B3 images, re-grade CSVs and Python API dependencies.
- Backend milestone: four contract tests passed. Real CUDA epoch-110 example 000
  reproduces B3 with SSIM 0.999999814 (maximum pixel difference 0.0004722).
  EMA selection and fixed EchoTime verified; original predictor + three held-out
  classifiers load successfully. Receipt: `api/runtime/smoke.json` (local/ignored).
- Eight backend contract tests now pass, including PNG source metadata, exact-float
  patient archive contents, joint-target verdicts and PNG/PDF/JSON report formats.
- Next.js production build and TypeScript check pass for all six routes.
- Browser QA caught and fixed MUI/Emotion SSR hydration and loading-select warnings.
  The Studio displays actual per-judge probabilities and labels saved results.
- HTTP integration through the Next.js proxy passed with caching disabled: epoch
  110 EMA, same-noise null and epoch 10 each streamed 13 real intermediate frames.
  Fixed EchoTime and four-classifier verification passed in every run. CPU noise
  and flip probes loaded five classifiers (including the legacy control); a real
  local DICOM passed preprocessing without returning its patient identifier.
  Receipt: `api/runtime/integration.json` (local/ignored).
- Milestones committed: `580a573` progress log; `fc2420b` numerical backend;
  `aedfaf6` six-page MUI frontend. Further integration/export fixes follow.

## Continue here

Finish browser/API integration QA and commit this first complete local prototype.
Next priorities: persistent jobs/cache, whole-patient live generation queue, demo
recording and second-machine rehearsal. Cloud/public deployment and Groq/history
remain optional and have not been started.
