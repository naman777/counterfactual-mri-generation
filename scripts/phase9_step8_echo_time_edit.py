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
    """
    Pick one high EchoTime example and one low EchoTime example from test set.
    This gives us both directions:
      high -> low
      low -> high
    """
    df_sorted = df.sort_values("echo_time_norm")

    low_row = df_sorted.iloc[0]
    high_row = df_sorted.iloc[-1]

    return [
        high_row,
        low_row,
    ]


def choose_target_echo(source_echo_norm: float) -> float:
    """
    Choose a target far from the source.
    """
    if source_echo_norm >= 0.50:
        return 0.15
    return 0.85


def echo_norm_to_ms(echo_norm: float, echo_stats: dict):
    echo_min = float(echo_stats["min"])
    echo_max = float(echo_stats["max"])
    return float(echo_norm * (echo_max - echo_min) + echo_min)


def save_echo_grid(records, output_path):
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
            f"Original\n{rec['manufacturer_short']} | {rec['field_strength']} | "
            f"TE={rec['source_echo_norm']:.3f}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Target TE={rec['target_echo_norm']:.2f}\n"
            f"t={rec['edit_timestep']}, gs={rec['guidance_scale']}"
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="hot")
        axes[i, 2].set_title(
            f"Diff\nSSIM={rec['global_ssim']:.3f}, MAD={rec['mean_abs_diff']:.3f}"
        )
        axes[i, 2].axis("off")

        text = (
            f"Predictor EchoTime\n"
            f"Before pred norm: {rec['before_echo_pred']:.4f}\n"
            f"After pred norm:  {rec['after_echo_pred']:.4f}\n"
            f"Target norm:      {rec['target_echo_norm']:.4f}\n"
            f"Before abs err:   {rec['before_echo_abs_error']:.4f}\n"
            f"After abs err:    {rec['after_echo_abs_error']:.4f}\n"
            f"Improvement:      {rec['echo_error_improvement']:.4f}\n"
            f"Moved closer:     {rec['moved_closer_to_target']}\n\n"
            f"Mfg pred: {rec['before_mfg_pred']} → {rec['after_mfg_pred']}\n"
            f"Field pred: {rec['before_field_pred']} → {rec['after_field_pred']}"
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
    target_echo_norm,
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

    # Change EchoTime only.
    target_mfg = source_mfg
    target_field = source_field
    target_echo = float(target_echo_norm)

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

    before_echo_pred = float(shift["before_echo_time_norm_pred"])
    after_echo_pred = float(shift["after_echo_time_norm_pred"])

    before_echo_abs_error = float(shift["before_echo_abs_error_to_target"])
    after_echo_abs_error = float(shift["after_echo_abs_error_to_target"])
    echo_error_improvement = float(shift["echo_abs_error_improvement"])

    moved_closer_to_target = after_echo_abs_error < before_echo_abs_error

    before_mfg_pred_id = int(shift["before"]["manufacturer_pred"])
    after_mfg_pred_id = int(shift["after"]["manufacturer_pred"])
    before_field_pred_id = int(shift["before"]["field_strength_pred"])
    after_field_pred_id = int(shift["after"]["field_strength_pred"])

    mfg_name = MANUFACTURER_ID_TO_NAME[source_mfg]
    field_name = FIELD_ID_TO_NAME[source_field]

    mfg_short = mfg_name.replace("GE MEDICAL SYSTEMS", "GE")

    source_echo_ms = float(row["echo_time"])
    target_echo_ms = echo_norm_to_ms(target_echo, validator.echo_stats)

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
        "manufacturer": mfg_name,

        "field_strength_label": source_field,
        "field_strength": field_name,

        "source_echo_time": source_echo_ms,
        "source_echo_norm": source_echo,

        "target_echo_norm": target_echo,
        "target_echo_time_estimated_ms": target_echo_ms,

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
        "manufacturer": mfg_name,
        "manufacturer_short": mfg_short,

        "field_strength_label": source_field,
        "field_strength": field_name,

        "source_echo_time": source_echo_ms,
        "source_echo_norm": source_echo,
        "target_echo_norm": target_echo,
        "target_echo_time_estimated_ms": target_echo_ms,

        "edit_timestep": int(edit_timestep),
        "guidance_scale": float(guidance_scale),

        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,

        "before_echo_pred": before_echo_pred,
        "after_echo_pred": after_echo_pred,
        "before_echo_abs_error": before_echo_abs_error,
        "after_echo_abs_error": after_echo_abs_error,
        "echo_error_improvement": echo_error_improvement,
        "moved_closer_to_target": bool(moved_closer_to_target),

        "before_mfg_pred": MANUFACTURER_ID_TO_NAME[before_mfg_pred_id].replace("GE MEDICAL SYSTEMS", "GE"),
        "after_mfg_pred": MANUFACTURER_ID_TO_NAME[after_mfg_pred_id].replace("GE MEDICAL SYSTEMS", "GE"),

        "before_field_pred": FIELD_ID_TO_NAME[before_field_pred_id],
        "after_field_pred": FIELD_ID_TO_NAME[after_field_pred_id],

        "manufacturer_stable": before_mfg_pred_id == after_mfg_pred_id == source_mfg,
        "field_strength_stable": before_field_pred_id == after_field_pred_id == source_field,

        "run_dir": str(run_dir),
    }

    return record


def make_echo_improvement_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    labels = []
    improvements = []

    for _, row in df.iterrows():
        label = (
            f"TE {row['source_echo_norm']:.2f}→{row['target_echo_norm']:.2f}\n"
            f"t={int(row['edit_timestep'])}, gs={float(row['guidance_scale']):.1f}"
        )
        labels.append(label)
        improvements.append(float(row["echo_error_improvement"]))

    plt.figure(figsize=(13, 5))
    plt.bar(range(len(improvements)), improvements)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(improvements)), labels, rotation=60, ha="right")
    plt.ylabel("EchoTime error improvement")
    plt.title("EchoTime Edit: Positive Means Prediction Moved Closer to Target")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_tradeoff_plot(df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(df["global_ssim"], df["echo_error_improvement"])

    for _, row in df.iterrows():
        label = f"t{int(row['edit_timestep'])}/g{float(row['guidance_scale']):.0f}"
        plt.text(
            float(row["global_ssim"]),
            float(row["echo_error_improvement"]),
            label,
            fontsize=8,
        )

    plt.xlabel("Global SSIM")
    plt.ylabel("EchoTime error improvement")
    plt.title("Tradeoff: Anatomy Preservation vs EchoTime Target Movement")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("=" * 80)
    print("PHASE 9 STEP 8: ECHOTIME COUNTERFACTUAL EDIT")
    print("=" * 80)

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step8_echo_time_edit"
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
        target_echo = choose_target_echo(float(row["echo_time_norm"]))
        print(
            f"  Example {idx}: patient={row['patient_id']} | "
            f"manufacturer={row['manufacturer']} | "
            f"field={row['field_strength']} | "
            f"source_TE_norm={float(row['echo_time_norm']):.6f} | "
            f"target_TE_norm={target_echo:.2f}"
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
        source_echo = float(row["echo_time_norm"])
        target_echo = choose_target_echo(source_echo)

        print("\n" + "-" * 80)
        print(f"Running example {ex_idx}: EchoTime {source_echo:.4f} -> {target_echo:.4f}")
        print("-" * 80)

        example_records = []

        for edit_timestep in edit_timesteps:
            for guidance_scale in guidance_scales:
                run_name = (
                    f"example_{ex_idx:02d}_"
                    f"echo_{source_echo:.3f}_to_{target_echo:.3f}_"
                    f"t{edit_timestep}_gs{str(guidance_scale).replace('.', 'p')}"
                )
                run_name = run_name.replace(".", "p")

                run_dir = runs_dir / run_name

                rec = run_one_setting(
                    editor=editor,
                    validator=validator,
                    row=row,
                    target_echo_norm=target_echo,
                    edit_timestep=edit_timestep,
                    guidance_scale=guidance_scale,
                    seed=42,
                    run_dir=run_dir,
                )

                print(
                    f"t={edit_timestep:3d}, gs={guidance_scale:.1f} | "
                    f"SSIM={rec['global_ssim']:.4f} | "
                    f"MAD={rec['mean_abs_diff']:.4f} | "
                    f"echo_pred {rec['before_echo_pred']:.4f} -> {rec['after_echo_pred']:.4f} | "
                    f"target={rec['target_echo_norm']:.4f} | "
                    f"err_improve={rec['echo_error_improvement']:.4f} | "
                    f"closer={rec['moved_closer_to_target']} | "
                    f"mfg_stable={rec['manufacturer_stable']} | "
                    f"field_stable={rec['field_strength_stable']}"
                )

                example_records.append(rec)
                all_records.append(rec)

        grid_path = grids_dir / f"example_{ex_idx:02d}_echo_time_edit_grid.png"
        save_echo_grid(example_records, grid_path)
        print(f"Saved grid: {grid_path}")

    df_out = pd.DataFrame([
        {
            "patient_id": r["patient_id"],
            "manufacturer": r["manufacturer"],
            "manufacturer_short": r["manufacturer_short"],
            "field_strength": r["field_strength"],

            "source_echo_time": r["source_echo_time"],
            "source_echo_norm": r["source_echo_norm"],
            "target_echo_norm": r["target_echo_norm"],
            "target_echo_time_estimated_ms": r["target_echo_time_estimated_ms"],

            "edit_timestep": r["edit_timestep"],
            "guidance_scale": r["guidance_scale"],

            "global_ssim": r["global_ssim"],
            "mean_abs_diff": r["mean_abs_diff"],
            "max_abs_diff": r["max_abs_diff"],

            "before_echo_pred": r["before_echo_pred"],
            "after_echo_pred": r["after_echo_pred"],
            "before_echo_abs_error": r["before_echo_abs_error"],
            "after_echo_abs_error": r["after_echo_abs_error"],
            "echo_error_improvement": r["echo_error_improvement"],
            "moved_closer_to_target": r["moved_closer_to_target"],

            "before_mfg_pred": r["before_mfg_pred"],
            "after_mfg_pred": r["after_mfg_pred"],
            "before_field_pred": r["before_field_pred"],
            "after_field_pred": r["after_field_pred"],

            "manufacturer_stable": r["manufacturer_stable"],
            "field_strength_stable": r["field_strength_stable"],

            "run_dir": r["run_dir"],
        }
        for r in all_records
    ])

    metrics_csv = output_dir / "echo_time_edit_metrics.csv"
    df_out.to_csv(metrics_csv, index=False)

    balanced = df_out[
        (df_out["echo_error_improvement"] > 0)
        & (df_out["global_ssim"] >= 0.70)
    ].sort_values(
        ["echo_error_improvement", "global_ssim"],
        ascending=False,
    )

    high_improvement = df_out.sort_values("echo_error_improvement", ascending=False)

    summary = {
        "num_outputs": int(len(df_out)),
        "edit_timesteps": edit_timesteps,
        "guidance_scales": guidance_scales,

        "moved_closer_rate": float(df_out["moved_closer_to_target"].mean()),
        "mean_echo_error_improvement": float(df_out["echo_error_improvement"].mean()),
        "median_echo_error_improvement": float(df_out["echo_error_improvement"].median()),
        "max_echo_error_improvement": float(df_out["echo_error_improvement"].max()),

        "mean_global_ssim": float(df_out["global_ssim"].mean()),
        "mean_abs_diff": float(df_out["mean_abs_diff"].mean()),

        "manufacturer_stability_rate": float(df_out["manufacturer_stable"].mean()),
        "field_strength_stability_rate": float(df_out["field_strength_stable"].mean()),

        "best_high_improvement": high_improvement.head(5)[
            [
                "manufacturer",
                "field_strength",
                "source_echo_norm",
                "target_echo_norm",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "before_echo_pred",
                "after_echo_pred",
                "before_echo_abs_error",
                "after_echo_abs_error",
                "echo_error_improvement",
                "moved_closer_to_target",
                "manufacturer_stable",
                "field_strength_stable",
                "run_dir",
            ]
        ].to_dict(orient="records"),

        "best_balanced": balanced.head(5)[
            [
                "manufacturer",
                "field_strength",
                "source_echo_norm",
                "target_echo_norm",
                "edit_timestep",
                "guidance_scale",
                "global_ssim",
                "mean_abs_diff",
                "before_echo_pred",
                "after_echo_pred",
                "before_echo_abs_error",
                "after_echo_abs_error",
                "echo_error_improvement",
                "moved_closer_to_target",
                "manufacturer_stable",
                "field_strength_stable",
                "run_dir",
            ]
        ].to_dict(orient="records"),

        "metrics_csv": str(metrics_csv),
    }

    summary_path = output_dir / "echo_time_edit_summary.json"
    save_json(summary, summary_path)

    improvement_plot = output_dir / "echo_time_error_improvement.png"
    tradeoff_plot = output_dir / "echo_time_tradeoff.png"

    make_echo_improvement_plot(df_out, improvement_plot)
    make_tradeoff_plot(df_out, tradeoff_plot)

    print("\n" + "=" * 80)
    print("STEP 8 COMPLETE")
    print("=" * 80)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(metrics_csv)
    print(summary_path)
    print(improvement_plot)
    print(tradeoff_plot)
    print(grids_dir)

    print("\nSend me:")
    print("1. Full terminal output")
    print("2. echo_time_edit_summary.json contents")
    print("3. echo_time_tradeoff.png")
    print("4. example_00_echo_time_edit_grid.png")
    print("5. example_01_echo_time_edit_grid.png")


if __name__ == "__main__":
    main()