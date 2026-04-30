import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.counterfactual_editor import CounterfactualEditor
from src.explainability.validation import PredictorValidator, compute_predictor_shift


MANUFACTURER_ID_TO_NAME = {
    0: "GE MEDICAL SYSTEMS",
    1: "SIEMENS",
}

FIELD_ID_TO_NAME = {
    0: "1.5T",
    1: "3.0T",
}


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def choose_examples(df):
    ge_rows = df[df["manufacturer_label"] == 0]
    siemens_rows = df[df["manufacturer_label"] == 1]

    if len(ge_rows) == 0:
        raise ValueError("No GE examples found.")
    if len(siemens_rows) == 0:
        raise ValueError("No SIEMENS examples found.")

    return [
        ge_rows.iloc[0],
        siemens_rows.iloc[0],
    ]


def save_tuning_grid(records, output_path):
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, rec in enumerate(records):
        original = rec["original"]
        generated = rec["generated"]
        diff = np.abs(original - generated)

        before_target_prob = rec["before_target_manufacturer_prob"]
        after_target_prob = rec["after_target_manufacturer_prob"]
        gain = rec["target_manufacturer_prob_gain"]

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"Original\n{rec['source_manufacturer_short']} | {rec['field_strength']} | TE={rec['echo_time_norm']:.3f}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Target: {rec['target_manufacturer_short']}\n"
            f"t={rec['edit_timestep']}, gs={rec['guidance_scale']}"
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="hot")
        axes[i, 2].set_title(
            f"Diff\nSSIM={rec['global_ssim']:.3f}, MAD={rec['mean_abs_diff']:.3f}"
        )
        axes[i, 2].axis("off")

        text = (
            f"Predictor\n"
            f"Before pred: {rec['before_pred_short']}\n"
            f"After pred: {rec['after_pred_short']}\n"
            f"Target prob before: {before_target_prob:.6f}\n"
            f"Target prob after:  {after_target_prob:.6f}\n"
            f"Gain: {gain:.6f}\n"
            f"Hit target: {rec['hit_target_after']}"
        )
        axes[i, 3].text(0.02, 0.5, text, fontsize=10, va="center")
        axes[i, 3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_one_setting(
    editor,
    validator,
    row,
    edit_timestep,
    guidance_scale,
    seed,
    run_dir,
):
    image_path = ROOT / str(row["image_path"])
    original = CounterfactualEditor.load_image_npy(str(image_path))

    source_mfg = int(row["manufacturer_label"])
    source_field = int(row["field_strength_label"])
    source_echo = float(row["echo_time_norm"])

    target_mfg = 1 - source_mfg
    target_field = source_field
    target_echo = source_echo

    result = editor.generate_counterfactual(
        image_01=original,
        target_manufacturer_label=target_mfg,
        target_field_strength_label=target_field,
        target_echo_time_norm=target_echo,
        edit_timestep=edit_timestep,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    generated = result["generated_01"]

    global_ssim = float(ssim(original, generated, data_range=1.0))
    mean_abs_diff = float(np.mean(np.abs(original - generated)))
    max_abs_diff = float(np.max(np.abs(original - generated)))

    shift = compute_predictor_shift(
        original_01=original,
        generated_01=generated,
        target_manufacturer_label=target_mfg,
        target_field_strength_label=target_field,
        target_echo_time_norm=target_echo,
        validator=validator,
    )

    before_pred = int(shift["before"]["manufacturer_pred"])
    after_pred = int(shift["after"]["manufacturer_pred"])
    hit_target_after = after_pred == target_mfg

    source_mfg_name = MANUFACTURER_ID_TO_NAME[source_mfg]
    target_mfg_name = MANUFACTURER_ID_TO_NAME[target_mfg]
    field_name = FIELD_ID_TO_NAME[source_field]

    source_short = source_mfg_name.replace("GE MEDICAL SYSTEMS", "GE")
    target_short = target_mfg_name.replace("GE MEDICAL SYSTEMS", "GE")
    before_short = MANUFACTURER_ID_TO_NAME[before_pred].replace("GE MEDICAL SYSTEMS", "GE")
    after_short = MANUFACTURER_ID_TO_NAME[after_pred].replace("GE MEDICAL SYSTEMS", "GE")

    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "original.npy", original)
    np.save(run_dir / "counterfactual.npy", generated)

    editor.save_png(original, run_dir / "original.png")
    editor.save_png(generated, run_dir / "counterfactual.png")
    editor.save_png(np.abs(original - generated), run_dir / "absolute_difference.png")

    metadata = {
        "patient_id": row.get("patient_id", None),
        "image_path": str(image_path),
        "source_manufacturer_label": source_mfg,
        "source_manufacturer": source_mfg_name,
        "target_manufacturer_label": target_mfg,
        "target_manufacturer": target_mfg_name,
        "field_strength_label": source_field,
        "field_strength": field_name,
        "echo_time": float(row["echo_time"]),
        "echo_time_norm": source_echo,
        "edit_timestep": int(edit_timestep),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "predictor_shift": shift,
    }

    save_json(metadata, run_dir / "metadata_with_predictor_validation.json")

    record = {
        "original": original,
        "generated": generated,
        "patient_id": metadata["patient_id"],
        "source_manufacturer_label": source_mfg,
        "source_manufacturer": source_mfg_name,
        "source_manufacturer_short": source_short,
        "target_manufacturer_label": target_mfg,
        "target_manufacturer": target_mfg_name,
        "target_manufacturer_short": target_short,
        "field_strength_label": source_field,
        "field_strength": field_name,
        "echo_time": metadata["echo_time"],
        "echo_time_norm": source_echo,
        "edit_timestep": int(edit_timestep),
        "guidance_scale": float(guidance_scale),
        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "before_manufacturer_pred": before_pred,
        "after_manufacturer_pred": after_pred,
        "before_pred_short": before_short,
        "after_pred_short": after_short,
        "hit_target_after": bool(hit_target_after),
        "before_target_manufacturer_prob": float(shift["before_target_manufacturer_prob"]),
        "after_target_manufacturer_prob": float(shift["after_target_manufacturer_prob"]),
        "target_manufacturer_prob_gain": float(shift["target_manufacturer_prob_gain"]),
        "before_prob_GE": float(shift["before"]["manufacturer_prob_0_GE"]),
        "before_prob_SIEMENS": float(shift["before"]["manufacturer_prob_1_SIEMENS"]),
        "after_prob_GE": float(shift["after"]["manufacturer_prob_0_GE"]),
        "after_prob_SIEMENS": float(shift["after"]["manufacturer_prob_1_SIEMENS"]),
        "run_dir": str(run_dir),
    }

    return record


def make_gain_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    labels = []
    gains = []

    for _, row in df.iterrows():
        label = (
            f"{row['source_manufacturer_short']}→{row['target_manufacturer_short']}\n"
            f"t={int(row['edit_timestep'])}, gs={float(row['guidance_scale']):.1f}"
        )
        labels.append(label)
        gains.append(float(row["target_manufacturer_prob_gain"]))

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(gains)), gains)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(gains)), labels, rotation=60, ha="right")
    plt.ylabel("Target manufacturer probability gain")
    plt.title("Manufacturer Tuning Grid: Target Probability Gain")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_tradeoff_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(df["global_ssim"], df["target_manufacturer_prob_gain"])

    for _, row in df.iterrows():
        label = f"t{int(row['edit_timestep'])}/g{float(row['guidance_scale']):.0f}"
        plt.text(
            float(row["global_ssim"]),
            float(row["target_manufacturer_prob_gain"]),
            label,
            fontsize=8,
        )

    plt.xlabel("Global SSIM")
    plt.ylabel("Target manufacturer probability gain")
    plt.title("Tradeoff: Anatomy Preservation vs Target Manufacturer Shift")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 9 STEP 6: MANUFACTURER TUNING GRID")
    print("=" * 80)

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step6_manufacturer_tuning"
    runs_dir = output_dir / "runs"
    grids_dir = output_dir / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = ROOT / "data" / "processed" / "ax_t1_128" / "splits" / "test.csv"
    df = pd.read_csv(test_csv)

    examples = choose_examples(df)

    print(f"Loaded test set: {test_csv}")
    print(f"Test shape: {df.shape}")
    print("Selected examples:")
    for idx, row in enumerate(examples):
        print(
            f"  Example {idx}: patient={row['patient_id']} | "
            f"manufacturer={row['manufacturer']} | "
            f"field={row['field_strength']} | "
            f"TE_norm={float(row['echo_time_norm']):.6f}"
        )

    editor = CounterfactualEditor(
        config_path="configs/diffusion.yaml",
        checkpoint_path="checkpoints/diffusion/ax_t1_128/latest.pt",
    )

    validator = PredictorValidator(
        checkpoint_path="checkpoints/predictor/ax_t1_128/best.pt",
        echo_stats_path="data/processed/ax_t1_128/label_maps/echo_time_stats.json",
    )

    edit_timesteps = [100, 200, 300, 400]
    guidance_scales = [1.0, 3.0, 5.0]

    all_records = []

    for ex_idx, row in enumerate(examples):
        source_mfg = int(row["manufacturer_label"])
        target_mfg = 1 - source_mfg

        print("\n" + "-" * 80)
        print(
            f"Running example {ex_idx}: "
            f"{MANUFACTURER_ID_TO_NAME[source_mfg]} -> {MANUFACTURER_ID_TO_NAME[target_mfg]}"
        )
        print("-" * 80)

        example_records = []

        for edit_timestep in edit_timesteps:
            for guidance_scale in guidance_scales:
                source_name = MANUFACTURER_ID_TO_NAME[source_mfg].replace(" ", "_")
                target_name = MANUFACTURER_ID_TO_NAME[target_mfg].replace(" ", "_")

                run_name = (
                    f"example_{ex_idx:02d}_"
                    f"{source_name}_to_{target_name}_"
                    f"t{edit_timestep}_gs{str(guidance_scale).replace('.', 'p')}"
                )

                run_dir = runs_dir / run_name

                rec = run_one_setting(
                    editor=editor,
                    validator=validator,
                    row=row,
                    edit_timestep=edit_timestep,
                    guidance_scale=guidance_scale,
                    seed=42,
                    run_dir=run_dir,
                )

                print(
                    f"t={edit_timestep:3d}, gs={guidance_scale:.1f} | "
                    f"SSIM={rec['global_ssim']:.4f} | "
                    f"MAD={rec['mean_abs_diff']:.4f} | "
                    f"target_gain={rec['target_manufacturer_prob_gain']:.8f} | "
                    f"target_after={rec['after_target_manufacturer_prob']:.8f} | "
                    f"pred {rec['before_pred_short']} -> {rec['after_pred_short']} | "
                    f"hit={rec['hit_target_after']}"
                )

                example_records.append(rec)
                all_records.append(rec)

        grid_path = grids_dir / f"example_{ex_idx:02d}_manufacturer_tuning_grid.png"
        save_tuning_grid(example_records, grid_path)
        print(f"Saved grid: {grid_path}")

    df_out = pd.DataFrame([
        {
            "patient_id": r["patient_id"],
            "source_manufacturer": r["source_manufacturer"],
            "source_manufacturer_short": r["source_manufacturer_short"],
            "target_manufacturer": r["target_manufacturer"],
            "target_manufacturer_short": r["target_manufacturer_short"],
            "field_strength": r["field_strength"],
            "echo_time": r["echo_time"],
            "echo_time_norm": r["echo_time_norm"],
            "edit_timestep": r["edit_timestep"],
            "guidance_scale": r["guidance_scale"],
            "global_ssim": r["global_ssim"],
            "mean_abs_diff": r["mean_abs_diff"],
            "max_abs_diff": r["max_abs_diff"],
            "before_manufacturer_pred": r["before_manufacturer_pred"],
            "after_manufacturer_pred": r["after_manufacturer_pred"],
            "before_pred_short": r["before_pred_short"],
            "after_pred_short": r["after_pred_short"],
            "hit_target_after": r["hit_target_after"],
            "before_target_manufacturer_prob": r["before_target_manufacturer_prob"],
            "after_target_manufacturer_prob": r["after_target_manufacturer_prob"],
            "target_manufacturer_prob_gain": r["target_manufacturer_prob_gain"],
            "before_prob_GE": r["before_prob_GE"],
            "before_prob_SIEMENS": r["before_prob_SIEMENS"],
            "after_prob_GE": r["after_prob_GE"],
            "after_prob_SIEMENS": r["after_prob_SIEMENS"],
            "run_dir": r["run_dir"],
        }
        for r in all_records
    ])

    metrics_csv = output_dir / "manufacturer_tuning_metrics.csv"
    df_out.to_csv(metrics_csv, index=False)

    # Candidate definitions:
    # balanced = positive manufacturer shift with at least moderate anatomy preservation.
    balanced = df_out[
        (df_out["target_manufacturer_prob_gain"] > 0)
        & (df_out["global_ssim"] >= 0.70)
    ].sort_values(
        ["target_manufacturer_prob_gain", "global_ssim"],
        ascending=False,
    )

    high_gain = df_out.sort_values(
        "target_manufacturer_prob_gain",
        ascending=False,
    )

    summary = {
        "num_outputs": int(len(df_out)),
        "edit_timesteps": edit_timesteps,
        "guidance_scales": guidance_scales,
        "positive_gain_rate": float((df_out["target_manufacturer_prob_gain"] > 0).mean()),
        "hit_target_rate_after": float(df_out["hit_target_after"].mean()),
        "mean_target_gain": float(df_out["target_manufacturer_prob_gain"].mean()),
        "median_target_gain": float(df_out["target_manufacturer_prob_gain"].median()),
        "max_target_gain": float(df_out["target_manufacturer_prob_gain"].max()),
        "mean_global_ssim": float(df_out["global_ssim"].mean()),
        "mean_abs_diff": float(df_out["mean_abs_diff"].mean()),
        "best_high_gain": high_gain.head(5)[
            [
                "source_manufacturer",
                "target_manufacturer",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_manufacturer_prob_gain",
                "before_target_manufacturer_prob",
                "after_target_manufacturer_prob",
                "before_pred_short",
                "after_pred_short",
                "hit_target_after",
                "run_dir",
            ]
        ].to_dict(orient="records"),
        "best_balanced": balanced.head(5)[
            [
                "source_manufacturer",
                "target_manufacturer",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_manufacturer_prob_gain",
                "before_target_manufacturer_prob",
                "after_target_manufacturer_prob",
                "before_pred_short",
                "after_pred_short",
                "hit_target_after",
                "run_dir",
            ]
        ].to_dict(orient="records"),
        "metrics_csv": str(metrics_csv),
    }

    summary_path = output_dir / "manufacturer_tuning_summary.json"
    save_json(summary, summary_path)

    gain_plot = output_dir / "manufacturer_tuning_target_gain.png"
    tradeoff_plot = output_dir / "manufacturer_tuning_tradeoff.png"

    make_gain_plot(df_out, gain_plot)
    make_tradeoff_plot(df_out, tradeoff_plot)

    print("\n" + "=" * 80)
    print("STEP 6 COMPLETE")
    print("=" * 80)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(metrics_csv)
    print(summary_path)
    print(gain_plot)
    print(tradeoff_plot)
    print(grids_dir)

    print("\nSend me:")
    print("1. Full terminal output")
    print("2. manufacturer_tuning_summary.json contents")
    print("3. manufacturer_tuning_tradeoff.png")
    print("4. example_00_manufacturer_tuning_grid.png")
    print("5. example_01_manufacturer_tuning_grid.png")


if __name__ == "__main__":
    main()