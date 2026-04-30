import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explainability.validation import PredictorValidator, compute_predictor_shift


MANUFACTURER_ID_TO_NAME = {
    0: "GE MEDICAL SYSTEMS",
    1: "SIEMENS",
}

FIELD_ID_TO_NAME = {
    0: "1.5T",
    1: "3.0T",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_step3_runs(runs_dir: Path):
    if not runs_dir.exists():
        raise FileNotFoundError(f"Step 3 runs directory not found: {runs_dir}")

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    valid_runs = []
    for run_dir in run_dirs:
        required = [
            run_dir / "original.npy",
            run_dir / "counterfactual.npy",
            run_dir / "metadata.json",
        ]

        if all(p.exists() for p in required):
            valid_runs.append(run_dir)

    return valid_runs


def flatten_validation_record(run_dir: Path, metadata: dict, shift: dict):
    before = shift["before"]
    after = shift["after"]

    target_mfg = int(metadata["target_manufacturer_label"])
    source_mfg = int(metadata["source_manufacturer_label"])

    target_field = int(metadata["field_strength_label"])
    target_echo = float(metadata["echo_time_norm"])

    return {
        "run_name": run_dir.name,
        "patient_id": metadata.get("patient_id"),
        "source_manufacturer_label": source_mfg,
        "source_manufacturer": metadata.get("source_manufacturer"),
        "target_manufacturer_label": target_mfg,
        "target_manufacturer": metadata.get("target_manufacturer"),
        "field_strength_label": target_field,
        "field_strength": metadata.get("field_strength"),
        "echo_time": metadata.get("echo_time"),
        "echo_time_norm": target_echo,
        "edit_timestep": metadata.get("edit_timestep"),
        "guidance_scale": metadata.get("guidance_scale"),
        "global_ssim": metadata.get("global_ssim"),
        "mean_abs_diff": metadata.get("mean_abs_diff"),
        "max_abs_diff": metadata.get("max_abs_diff"),

        "before_manufacturer_pred": before["manufacturer_pred"],
        "after_manufacturer_pred": after["manufacturer_pred"],
        "before_manufacturer_pred_name": MANUFACTURER_ID_TO_NAME[int(before["manufacturer_pred"])],
        "after_manufacturer_pred_name": MANUFACTURER_ID_TO_NAME[int(after["manufacturer_pred"])],

        "before_prob_GE": before["manufacturer_prob_0_GE"],
        "before_prob_SIEMENS": before["manufacturer_prob_1_SIEMENS"],
        "after_prob_GE": after["manufacturer_prob_0_GE"],
        "after_prob_SIEMENS": after["manufacturer_prob_1_SIEMENS"],

        "before_target_manufacturer_prob": shift["before_target_manufacturer_prob"],
        "after_target_manufacturer_prob": shift["after_target_manufacturer_prob"],
        "target_manufacturer_prob_gain": shift["target_manufacturer_prob_gain"],
        "manufacturer_prediction_hit_target_after": int(after["manufacturer_pred"]) == target_mfg,

        "before_field_strength_pred": before["field_strength_pred"],
        "after_field_strength_pred": after["field_strength_pred"],
        "before_field_strength_pred_name": FIELD_ID_TO_NAME[int(before["field_strength_pred"])],
        "after_field_strength_pred_name": FIELD_ID_TO_NAME[int(after["field_strength_pred"])],
        "field_strength_stayed_same_as_target": int(after["field_strength_pred"]) == target_field,

        "before_prob_1p5T": before["field_strength_prob_0_1p5T"],
        "before_prob_3p0T": before["field_strength_prob_1_3p0T"],
        "after_prob_1p5T": after["field_strength_prob_0_1p5T"],
        "after_prob_3p0T": after["field_strength_prob_1_3p0T"],

        "before_target_field_strength_prob": shift["before_target_field_strength_prob"],
        "after_target_field_strength_prob": shift["after_target_field_strength_prob"],
        "target_field_strength_prob_gain": shift["target_field_strength_prob_gain"],

        "before_echo_time_norm_pred": shift["before_echo_time_norm_pred"],
        "after_echo_time_norm_pred": shift["after_echo_time_norm_pred"],
        "before_echo_abs_error_to_target": shift["before_echo_abs_error_to_target"],
        "after_echo_abs_error_to_target": shift["after_echo_abs_error_to_target"],
        "echo_abs_error_improvement": shift["echo_abs_error_improvement"],

        "run_dir": str(run_dir),
    }


def make_validation_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    labels = []
    gains = []

    for _, row in df.iterrows():
        short_name = (
            f"{row['source_manufacturer']}→{row['target_manufacturer']}\n"
            f"t={int(row['edit_timestep'])}, gs={float(row['guidance_scale']):.1f}"
        )
        short_name = short_name.replace("GE MEDICAL SYSTEMS", "GE")
        labels.append(short_name)
        gains.append(float(row["target_manufacturer_prob_gain"]))

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(gains)), gains)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(gains)), labels, rotation=45, ha="right")
    plt.ylabel("Target manufacturer probability gain")
    plt.title("Predictor Validation: Did Counterfactual Move Toward Target Manufacturer?")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 9 STEP 5: PREDICTOR VALIDATION FOR MANUFACTURER EDITS")
    print("=" * 80)

    step3_dir = ROOT / "reports" / "counterfactuals" / "phase9_step3_manufacturer_edit"
    runs_dir = step3_dir / "runs"

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step5_predictor_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_step3_runs(runs_dir)

    print(f"Found valid Step 3 runs: {len(run_dirs)}")
    for p in run_dirs:
        print(f"  - {p.name}")

    validator = PredictorValidator(
        checkpoint_path="checkpoints/predictor/ax_t1_128/best.pt",
        echo_stats_path="data/processed/ax_t1_128/label_maps/echo_time_stats.json",
    )

    records = []

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        original_path = run_dir / "original.npy"
        generated_path = run_dir / "counterfactual.npy"

        metadata = load_json(metadata_path)

        original = np.load(original_path).astype(np.float32)
        generated = np.load(generated_path).astype(np.float32)

        target_manufacturer_label = int(metadata["target_manufacturer_label"])
        target_field_strength_label = int(metadata["field_strength_label"])
        target_echo_time_norm = float(metadata["echo_time_norm"])

        shift = compute_predictor_shift(
            original_01=original,
            generated_01=generated,
            target_manufacturer_label=target_manufacturer_label,
            target_field_strength_label=target_field_strength_label,
            target_echo_time_norm=target_echo_time_norm,
            validator=validator,
        )

        validation_json = {
            "run_name": run_dir.name,
            "metadata": metadata,
            "predictor_shift": shift,
        }

        save_json(validation_json, run_dir / "predictor_validation.json")

        record = flatten_validation_record(run_dir, metadata, shift)
        records.append(record)

        print("\n" + "-" * 80)
        print(run_dir.name)
        print("-" * 80)
        print(f"Source manufacturer: {record['source_manufacturer']}")
        print(f"Target manufacturer: {record['target_manufacturer']}")
        print(f"Before predictor: {record['before_manufacturer_pred_name']}")
        print(f"After predictor : {record['after_manufacturer_pred_name']}")
        print(f"Target prob before: {record['before_target_manufacturer_prob']:.4f}")
        print(f"Target prob after : {record['after_target_manufacturer_prob']:.4f}")
        print(f"Target prob gain  : {record['target_manufacturer_prob_gain']:.4f}")
        print(f"SSIM={record['global_ssim']:.4f}, MAD={record['mean_abs_diff']:.4f}")

    df = pd.DataFrame(records)

    metrics_csv = output_dir / "manufacturer_predictor_validation.csv"
    df.to_csv(metrics_csv, index=False)

    positive_gain_rate = float((df["target_manufacturer_prob_gain"] > 0).mean())
    hit_target_rate = float(df["manufacturer_prediction_hit_target_after"].mean())

    summary = {
        "num_runs": int(len(df)),
        "positive_target_manufacturer_gain_rate": positive_gain_rate,
        "manufacturer_prediction_hit_target_rate_after": hit_target_rate,
        "mean_target_manufacturer_prob_gain": float(df["target_manufacturer_prob_gain"].mean()),
        "median_target_manufacturer_prob_gain": float(df["target_manufacturer_prob_gain"].median()),
        "mean_global_ssim": float(df["global_ssim"].mean()),
        "mean_abs_diff": float(df["mean_abs_diff"].mean()),
        "best_by_target_gain": df.sort_values(
            "target_manufacturer_prob_gain",
            ascending=False,
        ).head(3)[
            [
                "run_name",
                "source_manufacturer",
                "target_manufacturer",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_manufacturer_prob_gain",
                "before_target_manufacturer_prob",
                "after_target_manufacturer_prob",
            ]
        ].to_dict(orient="records"),
        "best_balanced_candidates": df[
            (df["target_manufacturer_prob_gain"] > 0)
            & (df["global_ssim"] >= 0.75)
        ].sort_values(
            ["target_manufacturer_prob_gain", "global_ssim"],
            ascending=False,
        )[
            [
                "run_name",
                "source_manufacturer",
                "target_manufacturer",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_manufacturer_prob_gain",
                "before_target_manufacturer_prob",
                "after_target_manufacturer_prob",
            ]
        ].head(5).to_dict(orient="records"),
        "metrics_csv": str(metrics_csv),
    }

    save_json(summary, output_dir / "manufacturer_predictor_validation_summary.json")

    plot_path = output_dir / "target_manufacturer_probability_gain.png"
    make_validation_plot(df, plot_path)

    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE")
    print("=" * 80)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(metrics_csv)
    print(output_dir / "manufacturer_predictor_validation_summary.json")
    print(plot_path)

    print("\nSend me:")
    print("1. Full terminal output")
    print("2. target_manufacturer_probability_gain.png")
    print("3. manufacturer_predictor_validation_summary.json contents")


if __name__ == "__main__":
    main()