import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def find_metadata_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "metadata_with_predictor_validation.json",
        run_dir / "predictor_validation.json",
        run_dir / "metadata.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"No metadata JSON found in {run_dir}")


def compute_histogram_metrics(original: np.ndarray, generated: np.ndarray, bins: int = 64):
    hist_o, bin_edges = np.histogram(original.flatten(), bins=bins, range=(0, 1), density=False)
    hist_g, _ = np.histogram(generated.flatten(), bins=bins, range=(0, 1), density=False)

    hist_o = hist_o.astype(np.float64)
    hist_g = hist_g.astype(np.float64)

    hist_o = hist_o / max(hist_o.sum(), 1.0)
    hist_g = hist_g / max(hist_g.sum(), 1.0)

    hist_l1 = float(np.sum(np.abs(hist_o - hist_g)))
    hist_intersection = float(np.sum(np.minimum(hist_o, hist_g)))

    return {
        "hist_original": hist_o,
        "hist_generated": hist_g,
        "bin_edges": bin_edges,
        "hist_l1_distance": hist_l1,
        "hist_intersection": hist_intersection,
    }


def compute_local_ssim(original: np.ndarray, generated: np.ndarray):
    score, ssim_map = ssim(
        original,
        generated,
        data_range=1.0,
        full=True,
        win_size=7,
    )

    ssim_map = np.asarray(ssim_map, dtype=np.float32)
    ssim_map_clipped = np.clip(ssim_map, 0.0, 1.0)

    return float(score), ssim_map_clipped


def extract_predictor_text(metadata: dict, case_type: str):
    """
    Handles metadata formats from manufacturer, field, and EchoTime scripts.
    """

    shift = metadata.get("predictor_shift")

    # Step 5 had nested format in predictor_validation.json.
    if shift is None and "predictor_shift" in metadata.get("metadata", {}):
        shift = metadata["metadata"]["predictor_shift"]

    if shift is None:
        return "Predictor metadata not found."

    before = shift.get("before", {})
    after = shift.get("after", {})

    lines = []

    if case_type.startswith("manufacturer"):
        target_label = metadata.get("target_manufacturer_label")
        if target_label is None:
            target_label = shift.get("target_manufacturer_label")

        if int(target_label) == 0:
            before_prob = before.get("manufacturer_prob_0_GE")
            after_prob = after.get("manufacturer_prob_0_GE")
            target_name = "GE"
        else:
            before_prob = before.get("manufacturer_prob_1_SIEMENS")
            after_prob = after.get("manufacturer_prob_1_SIEMENS")
            target_name = "SIEMENS"

        lines.extend([
            "Control: Manufacturer",
            f"Target: {target_name}",
            f"Before pred: {label_manufacturer(before.get('manufacturer_pred'))}",
            f"After pred: {label_manufacturer(after.get('manufacturer_pred'))}",
            f"Target prob before: {format_float(before_prob)}",
            f"Target prob after:  {format_float(after_prob)}",
            f"Gain: {format_float(shift.get('target_manufacturer_prob_gain'))}",
        ])

    elif case_type.startswith("field"):
        target_label = metadata.get("target_field_strength_label")
        if target_label is None:
            target_label = shift.get("target_field_strength_label")

        if int(target_label) == 0:
            before_prob = before.get("field_strength_prob_0_1p5T")
            after_prob = after.get("field_strength_prob_0_1p5T")
            target_name = "1.5T"
        else:
            before_prob = before.get("field_strength_prob_1_3p0T")
            after_prob = after.get("field_strength_prob_1_3p0T")
            target_name = "3.0T"

        lines.extend([
            "Control: Field strength",
            f"Target: {target_name}",
            f"Before pred: {label_field(before.get('field_strength_pred'))}",
            f"After pred: {label_field(after.get('field_strength_pred'))}",
            f"Target prob before: {format_float(before_prob)}",
            f"Target prob after:  {format_float(after_prob)}",
            f"Gain: {format_float(shift.get('target_field_strength_prob_gain'))}",
        ])

    elif case_type.startswith("echo"):
        lines.extend([
            "Control: EchoTime",
            f"Target norm: {format_float(shift.get('target_echo_time_norm'))}",
            f"Before pred norm: {format_float(shift.get('before_echo_time_norm_pred'))}",
            f"After pred norm:  {format_float(shift.get('after_echo_time_norm_pred'))}",
            f"Before abs err: {format_float(shift.get('before_echo_abs_error_to_target'))}",
            f"After abs err:  {format_float(shift.get('after_echo_abs_error_to_target'))}",
            f"Improvement: {format_float(shift.get('echo_abs_error_improvement'))}",
            f"Mfg stable: {label_manufacturer(before.get('manufacturer_pred'))} → {label_manufacturer(after.get('manufacturer_pred'))}",
            f"Field stable: {label_field(before.get('field_strength_pred'))} → {label_field(after.get('field_strength_pred'))}",
        ])

    return "\n".join(lines)


def label_manufacturer(value):
    if value is None:
        return "NA"
    value = int(value)
    return "GE" if value == 0 else "SIEMENS"


def label_field(value):
    if value is None:
        return "NA"
    value = int(value)
    return "1.5T" if value == 0 else "3.0T"


def format_float(value):
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def make_case_panel(
    original: np.ndarray,
    generated: np.ndarray,
    local_ssim_map: np.ndarray,
    hist_info: dict,
    predictor_text: str,
    case_title: str,
    output_path: Path,
):
    import matplotlib.pyplot as plt

    diff = np.abs(original - generated)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(generated, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Counterfactual")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")

    axes[3].imshow(local_ssim_map, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Local SSIM Map")
    axes[3].axis("off")

    bins = hist_info["bin_edges"]
    centers = (bins[:-1] + bins[1:]) / 2

    axes[4].plot(centers, hist_info["hist_original"], label="Original")
    axes[4].plot(centers, hist_info["hist_generated"], label="Counterfactual")
    axes[4].set_title("Histogram")
    axes[4].set_xlabel("Intensity")
    axes[4].set_ylabel("Normalized count")
    axes[4].legend(fontsize=8)

    axes[5].text(0.02, 0.5, predictor_text, fontsize=10, va="center")
    axes[5].set_title("Predictor Validation")
    axes[5].axis("off")

    fig.suptitle(case_title, fontsize=16)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_combined_grid(case_records, output_path: Path):
    import matplotlib.pyplot as plt

    n = len(case_records)
    fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, rec in enumerate(case_records):
        original = rec["original"]
        generated = rec["generated"]
        diff = np.abs(original - generated)

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{rec['case_name']}\nOriginal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Counterfactual")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="hot")
        axes[i, 2].set_title(f"Difference\nMAD={rec['mean_abs_diff']:.4f}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(rec["local_ssim_map"], cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title(f"Local SSIM\nGlobal={rec['global_ssim']:.4f}")
        axes[i, 3].axis("off")

        axes[i, 4].text(0.02, 0.5, rec["short_text"], fontsize=9, va="center")
        axes[i, 4].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_selected_cases():
    """
    Selects final cases from existing Step 6/7/8 summaries.
    """

    cases = []

    manufacturer_summary_path = (
        ROOT
        / "reports"
        / "counterfactuals"
        / "phase9_step6_manufacturer_tuning"
        / "manufacturer_tuning_summary.json"
    )

    field_summary_path = (
        ROOT
        / "reports"
        / "counterfactuals"
        / "phase9_step7_field_strength_edit"
        / "field_strength_edit_summary.json"
    )

    echo_summary_path = (
        ROOT
        / "reports"
        / "counterfactuals"
        / "phase9_step8_echo_time_edit"
        / "echo_time_edit_summary.json"
    )

    if manufacturer_summary_path.exists():
        manufacturer_summary = load_json(manufacturer_summary_path)

        if manufacturer_summary.get("best_balanced"):
            cases.append({
                "case_name": "Manufacturer balanced",
                "case_type": "manufacturer_balanced",
                "run_dir": manufacturer_summary["best_balanced"][0]["run_dir"],
                "why_selected": "Best anatomy-preserving manufacturer edit candidate.",
            })

        if manufacturer_summary.get("best_high_gain"):
            cases.append({
                "case_name": "Manufacturer high-gain",
                "case_type": "manufacturer_high_gain",
                "run_dir": manufacturer_summary["best_high_gain"][0]["run_dir"],
                "why_selected": "Shows target manufacturer can be achieved, but with anatomy tradeoff.",
            })

    if field_summary_path.exists():
        field_summary = load_json(field_summary_path)

        if field_summary.get("best_balanced"):
            cases.append({
                "case_name": "Field strength best",
                "case_type": "field_strength_best",
                "run_dir": field_summary["best_balanced"][0]["run_dir"],
                "why_selected": "Best field-strength edit with strong predictor alignment and good SSIM.",
            })

    if echo_summary_path.exists():
        echo_summary = load_json(echo_summary_path)

        if echo_summary.get("best_balanced"):
            cases.append({
                "case_name": "EchoTime best",
                "case_type": "echo_time_best",
                "run_dir": echo_summary["best_balanced"][0]["run_dir"],
                "why_selected": "Best EchoTime edit with clear movement toward target and stable other controls.",
            })

    return cases


def process_case(case, output_dir: Path):
    run_dir = resolve_path(case["run_dir"])

    original_path = run_dir / "original.npy"
    generated_path = run_dir / "counterfactual.npy"

    if not original_path.exists():
        raise FileNotFoundError(f"Missing original.npy: {original_path}")
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing counterfactual.npy: {generated_path}")

    original = np.load(original_path).astype(np.float32)
    generated = np.load(generated_path).astype(np.float32)

    original = np.clip(original, 0, 1)
    generated = np.clip(generated, 0, 1)

    global_ssim, local_ssim_map = compute_local_ssim(original, generated)
    mean_abs_diff = float(np.mean(np.abs(original - generated)))
    max_abs_diff = float(np.max(np.abs(original - generated)))

    hist_info = compute_histogram_metrics(original, generated)

    metadata_path = find_metadata_file(run_dir)
    metadata = load_json(metadata_path)

    predictor_text = extract_predictor_text(metadata, case["case_type"])

    case_safe_name = case["case_name"].lower().replace(" ", "_").replace("-", "_")
    case_dir = output_dir / case_safe_name
    case_dir.mkdir(parents=True, exist_ok=True)

    panel_path = case_dir / "explainability_panel.png"

    case_title = (
        f"{case['case_name']} | "
        f"SSIM={global_ssim:.4f}, MAD={mean_abs_diff:.4f}, "
        f"Hist L1={hist_info['hist_l1_distance']:.4f}"
    )

    make_case_panel(
        original=original,
        generated=generated,
        local_ssim_map=local_ssim_map,
        hist_info=hist_info,
        predictor_text=predictor_text,
        case_title=case_title,
        output_path=panel_path,
    )

    metrics = {
        "case_name": case["case_name"],
        "case_type": case["case_type"],
        "why_selected": case["why_selected"],
        "run_dir": str(run_dir),
        "metadata_path": str(metadata_path),
        "panel_path": str(panel_path),
        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "mean_local_ssim": float(np.mean(local_ssim_map)),
        "min_local_ssim": float(np.min(local_ssim_map)),
        "hist_l1_distance": hist_info["hist_l1_distance"],
        "hist_intersection": hist_info["hist_intersection"],
        "predictor_text": predictor_text,
    }

    save_json(metrics, case_dir / "metrics.json")

    return {
        **metrics,
        "original": original,
        "generated": generated,
        "local_ssim_map": local_ssim_map,
        "short_text": predictor_text,
    }


def main():
    print("=" * 80)
    print("PHASE 9 STEP 9: FINAL EXPLAINABILITY PANELS")
    print("=" * 80)

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step9_final_explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = get_selected_cases()

    if not selected_cases:
        raise RuntimeError(
            "No selected cases found. Make sure Steps 6, 7, and 8 have been run."
        )

    print("Selected cases:")
    for case in selected_cases:
        print(f"  - {case['case_name']}: {case['run_dir']}")

    processed = []

    for case in selected_cases:
        print("\n" + "-" * 80)
        print(f"Processing: {case['case_name']}")
        print("-" * 80)

        rec = process_case(case, output_dir)
        processed.append(rec)

        print(f"Panel: {rec['panel_path']}")
        print(f"SSIM={rec['global_ssim']:.4f}")
        print(f"MAD={rec['mean_abs_diff']:.4f}")
        print(f"Histogram L1={rec['hist_l1_distance']:.4f}")

    combined_grid_path = output_dir / "combined_final_explainability_grid.png"
    make_combined_grid(processed, combined_grid_path)

    summary_rows = []
    for rec in processed:
        summary_rows.append({
            "case_name": rec["case_name"],
            "case_type": rec["case_type"],
            "why_selected": rec["why_selected"],
            "global_ssim": rec["global_ssim"],
            "mean_abs_diff": rec["mean_abs_diff"],
            "mean_local_ssim": rec["mean_local_ssim"],
            "hist_l1_distance": rec["hist_l1_distance"],
            "hist_intersection": rec["hist_intersection"],
            "run_dir": rec["run_dir"],
            "panel_path": rec["panel_path"],
            "predictor_text": rec["predictor_text"],
        })

    summary = {
        "num_cases": len(summary_rows),
        "cases": summary_rows,
        "combined_grid_path": str(combined_grid_path),
        "interpretation": {
            "manufacturer": (
                "Manufacturer control shows a tradeoff: low edit strength preserves anatomy "
                "but gives weak target shift; high edit strength can hit target but damages SSIM."
            ),
            "field_strength": (
                "Field strength control is the strongest result: target predictor alignment is high "
                "while anatomy preservation remains acceptable."
            ),
            "echo_time": (
                "EchoTime control moves predictions toward the requested target while keeping "
                "manufacturer and field strength stable, but the movement is directional rather than exact."
            ),
        },
    }

    summary_path = output_dir / "phase9_final_explainability_summary.json"
    save_json(summary, summary_path)

    csv_path = output_dir / "phase9_final_explainability_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("STEP 9 COMPLETE")
    print("=" * 80)
    print(f"Combined grid: {combined_grid_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV: {csv_path}")

    print("\nSend me:")
    print("1. Full terminal output")
    print("2. combined_final_explainability_grid.png")
    print("3. phase9_final_explainability_summary.json contents")


if __name__ == "__main__":
    main()