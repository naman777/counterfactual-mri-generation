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
- [ ] Phase 1: Studio, navigation, maps, streamed progress and verification.
- [ ] Phase 2: precomputed evidence and source-backed headline charts.
- [ ] Phase 3: CPU robustness probes, same-noise null edit and report exports.
- [ ] Phase 4: upload, batch zip and model card.
- [ ] Phase 5: curated cache, demo script, backup recording and second-machine rehearsal.

## Verification / commits

- Initial inspection: the code folder already has Git and an origin remote. Existing
  unrelated modified/untracked files will be left out of dashboard commits.
- Confirmed local models, precomputed B3 images, re-grade CSVs and Python API dependencies.
- Backend milestone: four contract tests passed. Real CUDA epoch-110 example 000
  reproduces B3 with SSIM 0.999999814 (maximum pixel difference 0.0004722).
  EMA selection and fixed EchoTime verified; original predictor + three held-out
  classifiers load successfully. Receipt: `api/runtime/smoke.json` (local/ignored).

## Continue here

Build the API adapter and validate one epoch-110 edit against saved B3 data before
claiming model compatibility. Record checks, limitations and commit milestones below.
