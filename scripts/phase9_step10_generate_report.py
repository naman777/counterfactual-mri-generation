import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, digits=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def write_report(report_path: Path, content: str):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def main():
    print("=" * 80)
    print("PHASE 9 STEP 10: GENERATE FINAL PHASE 9 REPORT")
    print("=" * 80)

    base = ROOT / "reports" / "counterfactuals"

    manufacturer_summary_path = base / "phase9_step6_manufacturer_tuning" / "manufacturer_tuning_summary.json"
    field_summary_path = base / "phase9_step7_field_strength_edit" / "field_strength_edit_summary.json"
    echo_summary_path = base / "phase9_step8_echo_time_edit" / "echo_time_edit_summary.json"
    explainability_summary_path = base / "phase9_step9_final_explainability" / "phase9_final_explainability_summary.json"

    manufacturer = load_json(manufacturer_summary_path)
    field = load_json(field_summary_path)
    echo = load_json(echo_summary_path)
    explainability = load_json(explainability_summary_path)

    output_dir = base / "phase9_step10_final_report"
    report_path = output_dir / "phase9_completion_report.md"

    lines = []

    lines.append("# Phase 9 Completion Report")
    lines.append("")
    lines.append("## Counterfactual MRI Generation and Acquisition Parameter Editing")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Phase 9 Objective")
    lines.append("")
    lines.append(
        "The objective of Phase 9 was to move from unconditional/conditional synthetic sampling "
        "to actual counterfactual editing. Instead of generating an MRI from random noise, the "
        "system now takes a real MRI slice, partially noises it, and denoises it under a new "
        "target acquisition condition."
    )
    lines.append("")
    lines.append("The counterfactual editing pipeline follows this process:")
    lines.append("")
    lines.append("```text")
    lines.append("Original MRI slice")
    lines.append("→ partial noising at selected edit timestep")
    lines.append("→ target-conditioned reverse diffusion")
    lines.append("→ counterfactual MRI image")
    lines.append("→ explainability and predictor validation")
    lines.append("```")
    lines.append("")

    lines.append("## 2. What Was Implemented")
    lines.append("")
    lines.append("The following Phase 9 components were implemented:")
    lines.append("")
    lines.append("- `src/inference/counterfactual_editor.py`")
    lines.append("- `src/explainability/validation.py`")
    lines.append("- Manufacturer counterfactual editing grid")
    lines.append("- Manufacturer tuning grid")
    lines.append("- Field strength counterfactual editing grid")
    lines.append("- EchoTime counterfactual editing grid")
    lines.append("- Final explainability panel generator")
    lines.append("")
    lines.append("Each generated counterfactual was evaluated using:")
    lines.append("")
    lines.append("- Global SSIM")
    lines.append("- Mean absolute difference")
    lines.append("- Local SSIM map")
    lines.append("- Absolute difference map")
    lines.append("- Histogram comparison")
    lines.append("- Predictor before/after validation")
    lines.append("")

    lines.append("## 3. Manufacturer Editing Results")
    lines.append("")

    if manufacturer:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Number of outputs | {manufacturer.get('num_outputs')} |")
        lines.append(f"| Positive target gain rate | {fmt(manufacturer.get('positive_gain_rate'))} |")
        lines.append(f"| Target hit rate after editing | {fmt(manufacturer.get('hit_target_rate_after'))} |")
        lines.append(f"| Mean target gain | {fmt(manufacturer.get('mean_target_gain'))} |")
        lines.append(f"| Median target gain | {fmt(manufacturer.get('median_target_gain'))} |")
        lines.append(f"| Max target gain | {fmt(manufacturer.get('max_target_gain'))} |")
        lines.append(f"| Mean global SSIM | {fmt(manufacturer.get('mean_global_ssim'))} |")
        lines.append(f"| Mean absolute difference | {fmt(manufacturer.get('mean_abs_diff'))} |")
        lines.append("")
        lines.append("### Interpretation")
        lines.append("")
        lines.append(
            "Manufacturer editing showed a clear tradeoff. Low edit strength preserved anatomy but "
            "caused only a weak shift in the target manufacturer probability. High edit strength could "
            "successfully flip the predictor to the target manufacturer, but it damaged anatomical "
            "similarity and reduced SSIM strongly."
        )
        lines.append("")

        best_balanced = manufacturer.get("best_balanced", [])
        best_high_gain = manufacturer.get("best_high_gain", [])

        if best_balanced:
            b = best_balanced[0]
            lines.append("### Best anatomy-preserving manufacturer candidate")
            lines.append("")
            lines.append(f"- Source → Target: `{b.get('source_manufacturer')}` → `{b.get('target_manufacturer')}`")
            lines.append(f"- Edit timestep: `{b.get('edit_timestep')}`")
            lines.append(f"- Guidance scale: `{b.get('guidance_scale')}`")
            lines.append(f"- SSIM: `{fmt(b.get('global_ssim'))}`")
            lines.append(f"- Mean absolute difference: `{fmt(b.get('mean_abs_diff'))}`")
            lines.append(f"- Target probability gain: `{fmt(b.get('target_manufacturer_prob_gain'), 6)}`")
            lines.append(f"- Hit target after editing: `{b.get('hit_target_after')}`")
            lines.append("")

        if best_high_gain:
            h = best_high_gain[0]
            lines.append("### Best target-achieving manufacturer candidate")
            lines.append("")
            lines.append(f"- Source → Target: `{h.get('source_manufacturer')}` → `{h.get('target_manufacturer')}`")
            lines.append(f"- Edit timestep: `{h.get('edit_timestep')}`")
            lines.append(f"- Guidance scale: `{h.get('guidance_scale')}`")
            lines.append(f"- SSIM: `{fmt(h.get('global_ssim'))}`")
            lines.append(f"- Mean absolute difference: `{fmt(h.get('mean_abs_diff'))}`")
            lines.append(f"- Target probability gain: `{fmt(h.get('target_manufacturer_prob_gain'), 6)}`")
            lines.append(f"- Hit target after editing: `{h.get('hit_target_after')}`")
            lines.append("")
    else:
        lines.append("Manufacturer summary was not found.")
        lines.append("")

    lines.append("## 4. Field Strength Editing Results")
    lines.append("")

    if field:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Number of outputs | {field.get('num_outputs')} |")
        lines.append(f"| Positive field gain rate | {fmt(field.get('positive_field_gain_rate'))} |")
        lines.append(f"| Field target hit rate after editing | {fmt(field.get('field_hit_target_rate_after'))} |")
        lines.append(f"| Mean target field gain | {fmt(field.get('mean_target_field_gain'))} |")
        lines.append(f"| Median target field gain | {fmt(field.get('median_target_field_gain'))} |")
        lines.append(f"| Max target field gain | {fmt(field.get('max_target_field_gain'))} |")
        lines.append(f"| Mean global SSIM | {fmt(field.get('mean_global_ssim'))} |")
        lines.append(f"| Mean absolute difference | {fmt(field.get('mean_abs_diff'))} |")
        lines.append("")
        lines.append("### Interpretation")
        lines.append("")
        lines.append(
            "Field strength editing was the strongest and cleanest result in Phase 9. The predictor "
            "moved to the target field strength in all tested cases while preserving anatomy reasonably well."
        )
        lines.append("")

        best_balanced = field.get("best_balanced", [])
        if best_balanced:
            b = best_balanced[0]
            lines.append("### Best field strength candidate")
            lines.append("")
            lines.append(f"- Manufacturer: `{b.get('manufacturer')}`")
            lines.append(f"- Source → Target: `{b.get('source_field')}` → `{b.get('target_field')}`")
            lines.append(f"- Edit timestep: `{b.get('edit_timestep')}`")
            lines.append(f"- Guidance scale: `{b.get('guidance_scale')}`")
            lines.append(f"- SSIM: `{fmt(b.get('global_ssim'))}`")
            lines.append(f"- Mean absolute difference: `{fmt(b.get('mean_abs_diff'))}`")
            lines.append(f"- Target field probability gain: `{fmt(b.get('target_field_prob_gain'), 6)}`")
            lines.append(f"- Before predictor: `{b.get('before_field_pred')}`")
            lines.append(f"- After predictor: `{b.get('after_field_pred')}`")
            lines.append(f"- Manufacturer stayed stable: `{b.get('before_mfg_pred')}` → `{b.get('after_mfg_pred')}`")
            lines.append("")
    else:
        lines.append("Field strength summary was not found.")
        lines.append("")

    lines.append("## 5. EchoTime Editing Results")
    lines.append("")

    if echo:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Number of outputs | {echo.get('num_outputs')} |")
        lines.append(f"| Moved closer to target rate | {fmt(echo.get('moved_closer_rate'))} |")
        lines.append(f"| Mean EchoTime error improvement | {fmt(echo.get('mean_echo_error_improvement'))} |")
        lines.append(f"| Median EchoTime error improvement | {fmt(echo.get('median_echo_error_improvement'))} |")
        lines.append(f"| Max EchoTime error improvement | {fmt(echo.get('max_echo_error_improvement'))} |")
        lines.append(f"| Mean global SSIM | {fmt(echo.get('mean_global_ssim'))} |")
        lines.append(f"| Mean absolute difference | {fmt(echo.get('mean_abs_diff'))} |")
        lines.append(f"| Manufacturer stability rate | {fmt(echo.get('manufacturer_stability_rate'))} |")
        lines.append(f"| Field strength stability rate | {fmt(echo.get('field_strength_stability_rate'))} |")
        lines.append("")
        lines.append("### Interpretation")
        lines.append("")
        lines.append(
            "EchoTime editing was successful as directional continuous control. In all tested cases, "
            "the predictor moved closer to the requested EchoTime target. The result was not always exact, "
            "especially for low-to-high EchoTime movement, but the direction was correct and the other controls "
            "remained stable."
        )
        lines.append("")

        best_balanced = echo.get("best_balanced", [])
        if best_balanced:
            b = best_balanced[0]
            lines.append("### Best EchoTime candidate")
            lines.append("")
            lines.append(f"- Manufacturer: `{b.get('manufacturer')}`")
            lines.append(f"- Field strength: `{b.get('field_strength')}`")
            lines.append(f"- Source EchoTime norm: `{fmt(b.get('source_echo_norm'))}`")
            lines.append(f"- Target EchoTime norm: `{fmt(b.get('target_echo_norm'))}`")
            lines.append(f"- Edit timestep: `{b.get('edit_timestep')}`")
            lines.append(f"- Guidance scale: `{b.get('guidance_scale')}`")
            lines.append(f"- SSIM: `{fmt(b.get('global_ssim'))}`")
            lines.append(f"- Mean absolute difference: `{fmt(b.get('mean_abs_diff'))}`")
            lines.append(f"- Predictor before: `{fmt(b.get('before_echo_pred'))}`")
            lines.append(f"- Predictor after: `{fmt(b.get('after_echo_pred'))}`")
            lines.append(f"- Error improvement: `{fmt(b.get('echo_error_improvement'))}`")
            lines.append(f"- Manufacturer stable: `{b.get('manufacturer_stable')}`")
            lines.append(f"- Field strength stable: `{b.get('field_strength_stable')}`")
            lines.append("")
    else:
        lines.append("EchoTime summary was not found.")
        lines.append("")

    lines.append("## 6. Final Explainability Panel Summary")
    lines.append("")

    if explainability:
        lines.append(f"Number of selected final cases: `{explainability.get('num_cases')}`")
        lines.append("")
        lines.append("| Case | SSIM | MAD | Local SSIM mean | Histogram L1 | Interpretation |")
        lines.append("|---|---:|---:|---:|---:|---|")

        for case in explainability.get("cases", []):
            lines.append(
                f"| {case.get('case_name')} "
                f"| {fmt(case.get('global_ssim'))} "
                f"| {fmt(case.get('mean_abs_diff'))} "
                f"| {fmt(case.get('mean_local_ssim'))} "
                f"| {fmt(case.get('hist_l1_distance'))} "
                f"| {case.get('why_selected')} |"
            )

        lines.append("")
        lines.append(f"Combined explainability grid:")
        lines.append("")
        lines.append(f"`{explainability.get('combined_grid_path')}`")
        lines.append("")
    else:
        lines.append("Final explainability summary was not found.")
        lines.append("")

    lines.append("## 7. Final Control Ranking")
    lines.append("")
    lines.append("Based on Phase 9 results, the control strength ranking is:")
    lines.append("")
    lines.append("1. **Field strength** — strongest and cleanest control.")
    lines.append("2. **EchoTime** — successful directional continuous control.")
    lines.append("3. **Manufacturer** — controllable, but strong target movement requires high edit strength and causes anatomy tradeoff.")
    lines.append("")

    lines.append("## 8. What We Can Claim")
    lines.append("")
    lines.append("We can claim:")
    lines.append("")
    lines.append("- A working counterfactual editing prototype has been implemented.")
    lines.append("- The system can edit real MRI slices using target acquisition parameters.")
    lines.append("- Field strength control is strongly validated by the predictor.")
    lines.append("- EchoTime control moves predictions in the desired direction.")
    lines.append("- Manufacturer control shows a measurable but tradeoff-heavy effect.")
    lines.append("- The outputs are explainable through difference maps, local SSIM maps, histograms, and predictor validation.")
    lines.append("")

    lines.append("## 9. What We Should Not Claim Yet")
    lines.append("")
    lines.append("We should not claim:")
    lines.append("")
    lines.append("- Clinical realism.")
    lines.append("- Perfect anatomy preservation.")
    lines.append("- Final medical usability.")
    lines.append("- Perfect manufacturer transfer.")
    lines.append("- Fully validated deployment-ready model.")
    lines.append("")

    lines.append("## 10. Recommended Next Phase")
    lines.append("")
    lines.append("The recommended next phase is dashboard integration and expanded evaluation.")
    lines.append("")
    lines.append("Next implementation tasks:")
    lines.append("")
    lines.append("- Add a Streamlit interface for selecting image and target condition.")
    lines.append("- Use `edit_timestep=100`, `guidance_scale=3.0` for field strength demo.")
    lines.append("- Use `edit_timestep=100`, `guidance_scale=1.0` for EchoTime demo.")
    lines.append("- Show manufacturer as a tradeoff example, not as the strongest result.")
    lines.append("- Run expanded evaluation on more examples for final report.")
    lines.append("")

    report = "\n".join(lines)
    write_report(report_path, report)

    print(f"Saved report: {report_path}")
    print("\nSTEP 10 COMPLETE")
    print("Send me the generated phase9_completion_report.md contents or screenshot if needed.")


if __name__ == "__main__":
    main()