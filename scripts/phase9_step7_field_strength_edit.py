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
    rows_15 = df[df["field_strength_label"] == 0]
    rows_30 = df[df["field_strength_label"] == 1]

    if len(rows_15) == 0:
        raise ValueError("No 1.5T examples found.")
    if len(rows_30) == 0:
        raise ValueError("No 3.0T examples found.")

    return [
        rows_15.iloc[0],
        rows_30.iloc[0],
    ]


def save_field_grid(records, output_path):
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, rec in enumerate(records):
        original = rec["original"]
        generated = rec["generated"]
        diff = np.abs(original - generated)

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"Original\n{rec['manufacturer_short']} | {rec['source_field']} | TE={rec['echo_time_norm']:.3f}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Target: {rec['target_field']}\n"
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
            f"Before field: {rec['before_field_pred']}\n"
            f"After field: {rec['after_field_pred']}\n"
            f"Target prob before: {rec['before_target_field_prob']:.6f}\n"
            f"Target prob after:  {rec['after_target_field_prob']:.6f}\n"
            f"Gain: {rec['target_field_prob_gain']:.6f}\n"
            f"Hit target: {rec['hit_target_after']}\n\n"
            f"Mfg before: {rec['before_mfg_pred']}\n"
            f"Mfg after: {rec['after_mfg_pred']}"
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

    # Flip field strength only.
    target_mfg = source_mfg
    target_field = 1 - source_field
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

    before_field_pred_id = int(shift["before"]["field_strength_pred"])
    after_field_pred_id = int(shift["after"]["field_strength_pred"])
    before_mfg_pred_id = int(shift["before"]["manufacturer_pred"])
    after_mfg_pred_id = int(shift["after"]["manufacturer_pred"])

    hit_target_after = after_field_pred_id == target_field

    source_mfg_name = MANUFACTURER_ID_TO_NAME[source_mfg]
    source_field_name = FIELD_ID_TO_NAME[source_field]
    target_field_name = FIELD_ID_TO_NAME[target_field]

    mfg_short = source_mfg_name.replace("GE MEDICAL SYSTEMS", "GE")

    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "original.npy", original)
    np.save(run_dir / "counterfactual.npy", generated)

    editor.save_png(original, run_dir / "original.png")
    editor.save_png(generated, run_dir / "counterfactual.png")
    editor.save_png(np.abs(original - generated), run_dir / "absolute_difference.png")

    metadata = {
        "patient_id": row.get("patient_id", None),
        "image_path": str(image_path),

        "manufacturer_label": source_mfg,
        "manufacturer": source_mfg_name,

        "source_field_strength_label": source_field,
        "source_field_strength": source_field_name,
        "target_field_strength_label": target_field,
        "target_field_strength": target_field_name,

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
        "manufacturer_label": source_mfg,
        "manufacturer": source_mfg_name,
        "manufacturer_short": mfg_short,

        "source_field_strength_label": source_field,
        "source_field": source_field_name,
        "target_field_strength_label": target_field,
        "target_field": target_field_name,

        "echo_time": metadata["echo_time"],
        "echo_time_norm": source_echo,

        "edit_timestep": int(edit_timestep),
        "guidance_scale": float(guidance_scale),

        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,

        "before_field_pred_id": before_field_pred_id,
        "after_field_pred_id": after_field_pred_id,
        "before_field_pred": FIELD_ID_TO_NAME[before_field_pred_id],
        "after_field_pred": FIELD_ID_TO_NAME[after_field_pred_id],

        "before_mfg_pred_id": before_mfg_pred_id,
        "after_mfg_pred_id": after_mfg_pred_id,
        "before_mfg_pred": MANUFACTURER_ID_TO_NAME[before_mfg_pred_id].replace("GE MEDICAL SYSTEMS", "GE"),
        "after_mfg_pred": MANUFACTURER_ID_TO_NAME[after_mfg_pred_id].replace("GE MEDICAL SYSTEMS", "GE"),

        "hit_target_after": bool(hit_target_after),

        "before_target_field_prob": float(shift["before_target_field_strength_prob"]),
        "after_target_field_prob": float(shift["after_target_field_strength_prob"]),
        "target_field_prob_gain": float(shift["target_field_strength_prob_gain"]),

        "before_prob_1p5T": float(shift["before"]["field_strength_prob_0_1p5T"]),
        "before_prob_3p0T": float(shift["before"]["field_strength_prob_1_3p0T"]),
        "after_prob_1p5T": float(shift["after"]["field_strength_prob_0_1p5T"]),
        "after_prob_3p0T": float(shift["after"]["field_strength_prob_1_3p0T"]),

        "before_target_manufacturer_prob": float(shift["before_target_manufacturer_prob"]),
        "after_target_manufacturer_prob": float(shift["after_target_manufacturer_prob"]),
        "target_manufacturer_prob_gain": float(shift["target_manufacturer_prob_gain"]),

        "before_echo_time_norm_pred": float(shift["before_echo_time_norm_pred"]),
        "after_echo_time_norm_pred": float(shift["after_echo_time_norm_pred"]),

        "run_dir": str(run_dir),
    }

    return record


def make_gain_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    labels = []
    gains = []

    for _, row in df.iterrows():
        label = (
            f"{row['source_field']}→{row['target_field']}\n"
            f"t={int(row['edit_timestep'])}, gs={float(row['guidance_scale']):.1f}"
        )
        labels.append(label)
        gains.append(float(row["target_field_prob_gain"]))

    plt.figure(figsize=(13, 5))
    plt.bar(range(len(gains)), gains)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(gains)), labels, rotation=60, ha="right")
    plt.ylabel("Target field-strength probability gain")
    plt.title("Field Strength Edit: Target Probability Gain")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_tradeoff_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(df["global_ssim"], df["target_field_prob_gain"])

    for _, row in df.iterrows():
        label = f"t{int(row['edit_timestep'])}/g{float(row['guidance_scale']):.0f}"
        plt.text(
            float(row["global_ssim"]),
            float(row["target_field_prob_gain"]),
            label,
            fontsize=8,
        )

    plt.xlabel("Global SSIM")
    plt.ylabel("Target field-strength probability gain")
    plt.title("Tradeoff: Anatomy Preservation vs Field-Strength Shift")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 9 STEP 7: FIELD STRENGTH COUNTERFACTUAL EDIT")
    print("=" * 80)

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step7_field_strength_edit"
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

    edit_timesteps = [100, 200, 300]
    guidance_scales = [1.0, 3.0, 5.0]

    all_records = []

    for ex_idx, row in enumerate(examples):
        source_field = int(row["field_strength_label"])
        target_field = 1 - source_field

        print("\n" + "-" * 80)
        print(
            f"Running example {ex_idx}: "
            f"{FIELD_ID_TO_NAME[source_field]} -> {FIELD_ID_TO_NAME[target_field]}"
        )
        print("-" * 80)

        example_records = []

        for edit_timestep in edit_timesteps:
            for guidance_scale in guidance_scales:
                source_field_name = FIELD_ID_TO_NAME[source_field].replace(".", "p")
                target_field_name = FIELD_ID_TO_NAME[target_field].replace(".", "p")

                run_name = (
                    f"example_{ex_idx:02d}_"
                    f"{source_field_name}_to_{target_field_name}_"
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
                    f"field_gain={rec['target_field_prob_gain']:.8f} | "
                    f"field_after={rec['after_target_field_prob']:.8f} | "
                    f"field pred {rec['before_field_pred']} -> {rec['after_field_pred']} | "
                    f"hit={rec['hit_target_after']} | "
                    f"mfg pred {rec['before_mfg_pred']} -> {rec['after_mfg_pred']}"
                )

                example_records.append(rec)
                all_records.append(rec)

        grid_path = grids_dir / f"example_{ex_idx:02d}_field_strength_edit_grid.png"
        save_field_grid(example_records, grid_path)
        print(f"Saved grid: {grid_path}")

    df_out = pd.DataFrame([
        {
            "patient_id": r["patient_id"],
            "manufacturer": r["manufacturer"],
            "manufacturer_short": r["manufacturer_short"],

            "source_field": r["source_field"],
            "target_field": r["target_field"],

            "echo_time": r["echo_time"],
            "echo_time_norm": r["echo_time_norm"],

            "edit_timestep": r["edit_timestep"],
            "guidance_scale": r["guidance_scale"],

            "global_ssim": r["global_ssim"],
            "mean_abs_diff": r["mean_abs_diff"],
            "max_abs_diff": r["max_abs_diff"],

            "before_field_pred": r["before_field_pred"],
            "after_field_pred": r["after_field_pred"],
            "hit_target_after": r["hit_target_after"],

            "before_target_field_prob": r["before_target_field_prob"],
            "after_target_field_prob": r["after_target_field_prob"],
            "target_field_prob_gain": r["target_field_prob_gain"],

            "before_mfg_pred": r["before_mfg_pred"],
            "after_mfg_pred": r["after_mfg_pred"],

            "before_target_manufacturer_prob": r["before_target_manufacturer_prob"],
            "after_target_manufacturer_prob": r["after_target_manufacturer_prob"],
            "target_manufacturer_prob_gain": r["target_manufacturer_prob_gain"],

            "before_echo_time_norm_pred": r["before_echo_time_norm_pred"],
            "after_echo_time_norm_pred": r["after_echo_time_norm_pred"],

            "run_dir": r["run_dir"],
        }
        for r in all_records
    ])

    metrics_csv = output_dir / "field_strength_edit_metrics.csv"
    df_out.to_csv(metrics_csv, index=False)

    balanced = df_out[
        (df_out["target_field_prob_gain"] > 0)
        & (df_out["global_ssim"] >= 0.70)
    ].sort_values(
        ["target_field_prob_gain", "global_ssim"],
        ascending=False,
    )

    high_gain = df_out.sort_values("target_field_prob_gain", ascending=False)

    summary = {
        "num_outputs": int(len(df_out)),
        "edit_timesteps": edit_timesteps,
        "guidance_scales": guidance_scales,

        "positive_field_gain_rate": float((df_out["target_field_prob_gain"] > 0).mean()),
        "field_hit_target_rate_after": float(df_out["hit_target_after"].mean()),

        "mean_target_field_gain": float(df_out["target_field_prob_gain"].mean()),
        "median_target_field_gain": float(df_out["target_field_prob_gain"].median()),
        "max_target_field_gain": float(df_out["target_field_prob_gain"].max()),

        "mean_global_ssim": float(df_out["global_ssim"].mean()),
        "mean_abs_diff": float(df_out["mean_abs_diff"].mean()),

        "best_high_gain": high_gain.head(5)[
            [
                "manufacturer",
                "source_field",
                "target_field",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_field_prob_gain",
                "before_target_field_prob",
                "after_target_field_prob",
                "before_field_pred",
                "after_field_pred",
                "hit_target_after",
                "before_mfg_pred",
                "after_mfg_pred",
                "run_dir",
            ]
        ].to_dict(orient="records"),

        "best_balanced": balanced.head(5)[
            [
                "manufacturer",
                "source_field",
                "target_field",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "target_field_prob_gain",
                "before_target_field_prob",
                "after_target_field_prob",
                "before_field_pred",
                "after_field_pred",
                "hit_target_after",
                "before_mfg_pred",
                "after_mfg_pred",
                "run_dir",
            ]
        ].to_dict(orient="records"),

        "metrics_csv": str(metrics_csv),
    }

    summary_path = output_dir / "field_strength_edit_summary.json"
    save_json(summary, summary_path)

    gain_plot = output_dir / "field_strength_target_gain.png"
    tradeoff_plot = output_dir / "field_strength_tradeoff.png"

    make_gain_plot(df_out, gain_plot)
    make_tradeoff_plot(df_out, tradeoff_plot)

    print("\n" + "=" * 80)
    print("STEP 7 COMPLETE")
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
    print("2. field_strength_edit_summary.json contents")
    print("3. field_strength_tradeoff.png")
    print("4. example_00_field_strength_edit_grid.png")
    print("5. example_01_field_strength_edit_grid.png")


if __name__ == "__main__":
    main()