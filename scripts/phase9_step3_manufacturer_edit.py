import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.counterfactual_editor import CounterfactualEditor


MANUFACTURER_ID_TO_NAME = {
    0: "GE MEDICAL SYSTEMS",
    1: "SIEMENS",
}

FIELD_ID_TO_NAME = {
    0: "1.5T",
    1: "3.0T",
}


def save_comparison_grid(records, output_path):
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, rec in enumerate(records):
        original = rec["original"]
        generated = rec["generated"]
        diff = np.abs(original - generated)

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"Original\n{rec['source_manufacturer']} | {rec['field_strength']} | TE_norm={rec['echo_time_norm']:.3f}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Counterfactual\nTarget: {rec['target_manufacturer']}\n"
            f"t={rec['edit_timestep']}, gs={rec['guidance_scale']}"
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="hot")
        axes[i, 2].set_title(
            f"Absolute Difference\nSSIM={rec['global_ssim']:.4f}, MAD={rec['mean_abs_diff']:.4f}"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def choose_examples(df):
    """
    Pick one GE example and one SIEMENS example from the held-out test set.
    """
    ge_rows = df[df["manufacturer_label"] == 0]
    siemens_rows = df[df["manufacturer_label"] == 1]

    if len(ge_rows) == 0:
        raise ValueError("No GE examples found in test set.")
    if len(siemens_rows) == 0:
        raise ValueError("No SIEMENS examples found in test set.")

    return [
        ge_rows.iloc[0],
        siemens_rows.iloc[0],
    ]


def run_one_edit(editor, row, edit_timestep, guidance_scale, seed, run_dir):
    image_path = ROOT / str(row["image_path"])
    original = CounterfactualEditor.load_image_npy(str(image_path))

    source_mfg = int(row["manufacturer_label"])
    source_field = int(row["field_strength_label"])
    source_echo = float(row["echo_time_norm"])

    # Flip manufacturer only.
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

    source_mfg_name = MANUFACTURER_ID_TO_NAME[source_mfg]
    target_mfg_name = MANUFACTURER_ID_TO_NAME[target_mfg]
    field_name = FIELD_ID_TO_NAME[source_field]

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
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "original": original,
        "generated": generated,
        "source_manufacturer": source_mfg_name,
        "target_manufacturer": target_mfg_name,
        "field_strength": field_name,
        "echo_time_norm": source_echo,
        "edit_timestep": int(edit_timestep),
        "guidance_scale": float(guidance_scale),
        "global_ssim": global_ssim,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "run_dir": str(run_dir),
        **metadata,
    }


def main():
    print("=" * 80)
    print("PHASE 9 STEP 3: FIRST MANUFACTURER COUNTERFACTUAL EDIT")
    print("=" * 80)

    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step3_manufacturer_edit"
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

    # First actual manufacturer edit settings.
    # t=100 checks light editing.
    # t=200 checks stronger but still reasonable editing.
    settings = [
        {"edit_timestep": 100, "guidance_scale": 1.0},
        {"edit_timestep": 100, "guidance_scale": 3.0},
        {"edit_timestep": 200, "guidance_scale": 1.0},
        {"edit_timestep": 200, "guidance_scale": 3.0},
    ]

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

        for setting in settings:
            edit_timestep = setting["edit_timestep"]
            guidance_scale = setting["guidance_scale"]

            run_name = (
                f"example_{ex_idx:02d}_"
                f"{MANUFACTURER_ID_TO_NAME[source_mfg].replace(' ', '_')}_to_"
                f"{MANUFACTURER_ID_TO_NAME[target_mfg].replace(' ', '_')}_"
                f"t{edit_timestep}_gs{str(guidance_scale).replace('.', 'p')}"
            )

            run_dir = runs_dir / run_name

            rec = run_one_edit(
                editor=editor,
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
                f"saved={run_dir}"
            )

            example_records.append(rec)
            all_records.append(rec)

        grid_path = grids_dir / f"example_{ex_idx:02d}_manufacturer_edit_grid.png"
        save_comparison_grid(example_records, grid_path)
        print(f"Saved grid: {grid_path}")

    metrics_df = pd.DataFrame([
        {
            "patient_id": r["patient_id"],
            "source_manufacturer": r["source_manufacturer"],
            "target_manufacturer": r["target_manufacturer"],
            "field_strength": r["field_strength"],
            "echo_time": r["echo_time"],
            "echo_time_norm": r["echo_time_norm"],
            "edit_timestep": r["edit_timestep"],
            "guidance_scale": r["guidance_scale"],
            "global_ssim": r["global_ssim"],
            "mean_abs_diff": r["mean_abs_diff"],
            "max_abs_diff": r["max_abs_diff"],
            "run_dir": r["run_dir"],
        }
        for r in all_records
    ])

    metrics_csv = output_dir / "manufacturer_edit_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "num_examples": len(examples),
        "num_outputs": len(all_records),
        "settings": settings,
        "mean_global_ssim": float(metrics_df["global_ssim"].mean()),
        "mean_abs_diff": float(metrics_df["mean_abs_diff"].mean()),
        "metrics_csv": str(metrics_csv),
    }

    with open(output_dir / "manufacturer_edit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 3 COMPLETE")
    print("=" * 80)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("\nSend me:")
    print("1. Full terminal output")
    print("2. The two grid images from:")
    print(grids_dir)


if __name__ == "__main__":
    main()