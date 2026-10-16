# Three-minute local demonstration

Before the panel: start the API and frontend, open each page once and check the
local API badge. Keep this document and the paper available offline. Do not say
that the model reliably converts scanner acquisitions.

1. **Studio (45 seconds).** The default source is the first saved B3 evaluation
   example, with the corresponding field-strength preset. Generate and identify
   the saved-result chip. Turn caching off and generate again to demonstrate real
   inference and noisy intermediate denoising states. EchoTime remains locked.
2. **Verification (25 seconds).** Read the original predictor and each independent
   judge separately. If all agree on this slice, say so; explain that agreement
   across the entire cohort is shown on Results. Show the difference/local-SSIM
   maps and histogram. Use Wipe to compare the images.
3. **Null control (20 seconds).** Run the same-noise null edit. Compare its SSIM
   and target/null high-frequency energy. Explain why re-noising alone can change
   an image. Download a report card and its complete JSON.
4. **Robustness Lab (30 seconds).** Select a slice, use noise or a vertical flip,
   then run the CPU probe. Read P(source class); switch between vendor and field
   strength. The legacy control is intentionally orientation-sensitive. A single
   slice is a demonstration, not a cohort estimate.
5. **Evidence and Results (40 seconds).** Filter evidence to all-three-confirmed
   and original-only cases. Open Results, inspect patient-clustered intervals and
   hover a bar for its exact source path. Show epoch comparison, SSIM vs gain and
   high-frequency charts. Manufacturer operating points differ; the caveat stays.
6. **Export (20 seconds).** Pick a patient in Batch Export, build the saved-evidence
   ZIP and open `manifest.json`. It contains source paths, generation settings,
   checkpoint hash, per-image metrics and judge verdicts.

Fallback: Evidence, Results, Model Card and saved-evidence exports need no GPU.
Studio can use exact saved B3 matches plus CPU classifier inference. If live GPU
inference fails, explicitly use the saved-result mode; do not present it as live.
Backup video and a second-machine dry run are not supplied by this first build.
