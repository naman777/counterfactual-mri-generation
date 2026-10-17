# Three-minute local demonstration

On demo day, use `start_demo.bat -Live` from the code root. It starts both servers,
waits for them and opens http://localhost:3000. Exact saved cases are checked
first, so the 20 demo slices still replay at their paper presets. Other settings,
such as t=200, run live. Leave saved results enabled, and describe each result
using its saved/live chip. Use the production frontend build when available.

Before the panel arrives, run one uncached t=200 edit to load the model, check
the GPU and observe actual denoising frames. Return to the field-strength paper
preset (t=100, w=5) for the opener: Patient 880, slice 25, seed 64 (3/3 held-out
judges). Keep Patient 061, slice 47, seed 119 as the disagreement example.
These are illustrative cases, not cohort estimates.

If GPU inference fails, stop the live API and double-click `start_demo.bat` for
**saved replay**. Both paper presets and their null edits remain available for
all 20 selected slices (80 exact requests), without model loading. Other settings
are blocked explicitly. Use one API worker in either mode.
Logs: `api/runtime/`. Re-running the launcher reuses servers already running in
the requested mode and rejects a mode mismatch.

1. **Overview (30 s).** Explain the problem, follow the pipeline and name the
   intended research users. Read headline rates with patient-clustered intervals.
2. **Studio (40 s).** Open with the recommended 3/3 case. Leave saved results
   enabled. Pick an epoch-110 paper preset and Generate. The paper seed follows the slice/preset.
   Point to “Saved demo evidence”. EchoTime stays fixed. Manufacturer also
   switches verification to Manufacturer.
3. **Explanations (25 s).** Show difference/local-SSIM maps and the histogram.
   It excludes source intensities ≤0.01 with one shared foreground mask.
   SSIM, MAD and high-frequency metrics still cover the full image.
4. **Verification and null (30 s).** Read the original predictor and held-out
   judges separately. Run the saved same-noise null edit, compare SSIM and texture,
   then export PNG/PDF/JSON evidence cards.
5. **Evidence and Results (35 s).** Inspect agreement and disagreement, patient
   CIs, epoch comparison and the rounded high-frequency table. Cohort rates use
   all evaluation slices, not the illustrative demo selection.
6. **Export (20 s).** Pick a patient and build the saved-evidence ZIP. Its manifest
   preserves settings, source paths and verdicts. Whole-patient live generation
   remains pending.

## Readiness

- [x] Six UI fixes, Overview and persistent replay bundle.
- [x] One-click launcher; local build, browser and replay verification.
- [x] Captioned backup: `demo/20261017/mrone-backup-walkthrough.mp4`.
  Silent sequence of recorded UI views; it does not show live inference.
- [ ] Real screen capture showing live generation: deferred at the user's request.
  The existing backup remains a labelled slideshow, not a live screen recording.
- [ ] Timed team rehearsal. Each member presents the page they built. Agree the
  handoffs first, rehearse the three-minute route with a timer and verify that
  the presentation's role-division slide matches the actual assignments.
- [ ] Show the running dashboard and the updated deck to Dr. Nijhawan before
  the panel and apply his feedback.
- [x] Screenshots and new deck copy at the capstone root:
  `presentation/dashboard-update-20261017/`.
- [ ] Second-laptop rehearsal: no laptop available (user confirmed 17 October).

## Transfer and offline rehearsal

`demo/20261017/mrone-demo.zip` includes code, saved results, test source arrays,
the evidence used by the pages, CPU classifier weights and the paper. It omits
raw DICOMs, diffusion checkpoints and installed dependencies.

On a future second Windows laptop, extract the archive. Before going offline,
install Python/Node and the project's scientific/API dependencies, run
`pnpm install --frozen-lockfile` inside `web-app`, then `pnpm build` once. CPU-only
PyTorch suffices for replay and optional Robustness Lab probes. Disconnect the
network, double-click `start_demo.bat` and follow the script above.

Run `python -m api.verify_demo --output reports/<new_rehearsal_folder>` from the
code root. Verify all saved requests, blocked uncached settings and downloaded
report formats. Open the MP4 in a video player. Tick the second-machine box only
after this actually succeeds on another laptop.
