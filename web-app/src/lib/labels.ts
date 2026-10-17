const labels: Record<string, string> = {
  cf: "Diffusion edit", null1: "Null edit", null2: "Null edit · second pass",
  null3: "Null edit · third pass", uncond: "Unconditional reconstruction",
  cycle: "Return to source", identity: "Unedited source", gamma: "Gamma adjustment",
  contrast_shift: "Contrast shift", gaussian_noise: "Gaussian noise",
  histogram_matching: "Histogram matching", source_images: "Source image",
  hf_cf_over_null: "High-frequency energy · edit / null (×)",
  hf_x1000: "High-frequency energy (×10⁻³)",
  real_test_fs_0: "Real test slices · 1.5 T", real_test_fs_1: "Real test slices · 3.0 T",
  "FS source 0": "Edit towards 3.0 T", "FS source 1": "Edit towards 1.5 T",
  set: "Evaluation group", metric: "Measure", mean: "Mean",
  slice_lo: "95% slice CI · lower", slice_hi: "95% slice CI · upper",
  patient_lo: "95% patient CI · lower", patient_hi: "95% patient CI · upper",
  fs_hit: "Field-strength hit rate", mfg_hit: "Manufacturer hit rate",
  original_predictor: "Original predictor", orientation_robust: "Orientation-robust judge",
};
export function readable(value: string): string {
  if (labels[value]) return labels[value];
  const match = value.match(/^(main|soundness|baselines|fixed|stronger|joint)_(field_strength|manufacturer|both)_t(\d+)_w([\d.]+)_s(\d+)$/);
  if (match) {
    const [, kind, axis, t, w, seed] = match;
    const title = axis === "field_strength" ? "Field strength" : axis === "manufacturer" ? "Manufacturer" : "Joint edit";
    const paper = kind === "main" && seed === "0";
    return `${title} · ${paper ? "paper setting" : `${({ soundness: "null controls", baselines: "baselines", fixed: "fixed setting", stronger: "strength sweep", joint: "both targets", main: "seed comparison" })[kind as "main"]} · t=${t}, w=${w} · seed offset ${seed}`}`;
  }
  return value.replaceAll("_", " ");
}
export function tableValue(value: string): string {
  if (value.trim() && Number.isFinite(Number(value))) {
    const number = Number(value);
    return Number.isInteger(number) ? String(number) : number.toFixed(2);
  }
  return readable(value);
}
