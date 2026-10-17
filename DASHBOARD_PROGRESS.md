# MR-one dashboard progress

## Small polish and live-mode readiness (2026-10-17)

- Histogram ticks show 0.25/0.75 correctly. Table integers retain their whole
  form (including seed 0), while non-integers keep two decimals.
- `demo/**/*.zip` is ignored; the existing archive is untracked and excluded.
- Studio opens on Patient 880, slice 25, seed 64, a saved 3/3 held-out case.
  Patient 061, slice 47 remains available for the disagreement demonstration.
- The demo script now recommends `start_demo.bat -Live`, an uncached warm-up
  before presenting, and plain replay as the fallback. Real screen recording was
  cancelled by the user. Timed team rehearsal and Dr. Nijhawan's review remain open.
- Targeted lint and production build pass. Formatting assertions and the saved
  opener contract pass. Live-mode HTTP verification loaded the saved opener,
  then completed one uncached t=200 edit with 13 real intermediate frames.
  Receipt: `api/runtime/polish_live_check_20261017.json` (ignored local QA).
  The live API remains running with the model loaded for this session.
  Research results and checkpoints are unchanged; no commit was made.

## Dashboard fixes and demo preparation (2026-10-17)

- [x] Per-axis paper seeds come from the saved evaluation index order. Every
  evaluation slice can match its saved preset; changing slices preserves the
  active paper axis and clears old status. Presets also select verification axis.
- [x] Human-readable method/set/metric labels, two-decimal research table values,
  one “original predictor” name, and source-foreground histograms (same mask for
  both images). Research arrays, CSVs and full-image metrics are unchanged.
- [x] Persistent 20-slice/80-request replay bundle, covering all 16 test patients
  and different judge outcomes. Both paper axes and their same-noise nulls replay
  without any model loading. Exact-request matching rejects other settings in
  replay mode. Saved report exports survive API restarts.
- [x] `start_demo.bat` starts both servers, waits for readiness, opens Overview
  and defaults to replay. It prefers a production build and detects API mode
  mismatch. `-Live` explicitly allows uncached inference.
- [x] Overview at `/`, Studio at `/studio`; problem, research users, pipeline and
  source-backed epoch-110 rates with patient-clustered CIs.
- [x] Actual screenshots, a captioned silent saved-screen backup walkthrough,
  a transfer bundle and an updated deck copy. Assets: `demo/20261017/`; deck at
  `../presentation/dashboard-update-20261017/` (original kept intact).
- [ ] Second-laptop rehearsal. User confirmed no second laptop is available.

Validation: ESLint and production build pass; 11 backend checks pass, including
all evaluation-slice seed matches, the shared histogram mask and every persistent
replay with editor/classifier calls patched to fail. All 80 HTTP replays match
their saved metrics/probabilities exactly through the Next.js proxy; uncached
requests are blocked and JSON/PNG/PDF report exports pass. Receipt:
`reports/v4_dashboard_demo_20261017_01/replay_http.json`. Browser checks cover
both presets, a non-default paper seed, readable evidence/results, null control
and a completed patient ZIP. No browser warnings/errors in the final session.
No GPU sampling, cloud spending or research headline changes in this task.
Chart entry animations are disabled so quick inspection/recording shows final
bar and curve values. The supplied presentation's existing proposal/work-plan
slides remain in the new copy; the four dashboard additions do not rewrite them.

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
- [x] Phase 5 first increment: presets, loading/error states, exact B3/session caching and three-minute demo script. [`web-app/DEMO.md`]
- [x] Phase 5 demo additions: persistent curated cache and captioned saved-screen backup (2026-10-17).
- [ ] Phase 5 remaining: dark mode and second-machine rehearsal (no second laptop available).

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
  `aedfaf6` six-page MUI frontend; `3f83061` uploads, exports and proxy integration.
- Final browser QA: live Studio and same-noise null metrics rendered; original-only
  Evidence filter showed actual matching records; Results displayed source-backed
  patient CIs; CPU noise probe rendered five classifiers; a seven-slice patient
  archive completed with a download link. No browser errors/warnings remained.
- Responsive checks: Studio at a 390 px viewport had 390 px document width (no
  horizontal overflow); laptop layout checked at 1440 px. Preview screenshot:
  `api/runtime/studio.jpg` (ignored). Export PDF rendered as one unclipped page.
- Final polish preserves the target image while a null run streams, identifies
  changed controls before regeneration, and uses array-row markers rather than
  asserting an unverified anatomical orientation.

## Continue here

The replay demo, launcher, Overview and presentation additions are complete.
Next priorities: a second-machine rehearsal when a laptop is available, then the
lower-priority whole-patient live queue and dark mode. Cloud/public deployment
and Groq/history remain optional. Nothing has been pushed or published.
