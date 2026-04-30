import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.counterfactual_editor import CounterfactualEditor


def save_side_by_side(original, generated, output_path):
    import matplotlib.pyplot as plt

    diff = np.abs(original - generated)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(generated, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction / Same Condition")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    output_dir = ROOT / "reports" / "counterfactuals" / "phase9_step2_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = ROOT / "data" / "processed" / "ax_t1_128" / "splits" / "test.csv"
    df = pd.read_csv(test_csv)

    row = df.iloc[0]

    image_path = ROOT / str(row["image_path"])
    image = CounterfactualEditor.load_image_npy(str(image_path))

    source_manufacturer_label = int(row["manufacturer_label"])
    source_field_strength_label = int(row["field_strength_label"])
    source_echo_time_norm = float(row["echo_time_norm"])

    print("=" * 80)
    print("PHASE 9 STEP 2: RECONSTRUCTION SANITY TEST")
    print("=" * 80)
    print(f"Image path: {image_path}")
    print(f"Original manufacturer: {row['manufacturer']} | label={source_manufacturer_label}")
    print(f"Original field strength: {row['field_strength']} | label={source_field_strength_label}")
    print(f"Original echo_time: {row['echo_time']}")
    print(f"Original echo_time_norm: {source_echo_time_norm:.6f}")

    editor = CounterfactualEditor(
        config_path="configs/diffusion.yaml",
        checkpoint_path="checkpoints/diffusion/ax_t1_128/latest.pt",
    )

    # Important: target condition is SAME as source condition.
    # This checks whether partial noising + denoising can preserve the image.
    result = editor.generate_counterfactual(
        image_01=image,
        target_manufacturer_label=source_manufacturer_label,
        target_field_strength_label=source_field_strength_label,
        target_echo_time_norm=source_echo_time_norm,
        edit_timestep=100,
        guidance_scale=1.0,
        seed=42,
    )

    generated = result["generated_01"]

    global_ssim = ssim(image, generated, data_range=1.0)
    mean_abs_diff = float(np.mean(np.abs(image - generated)))
    max_abs_diff = float(np.max(np.abs(image - generated)))

    metrics = {
        "test_type": "same_condition_reconstruction",
        "image_path": str(image_path),
        "patient_id": row.get("patient_id", None),
        "manufacturer": row.get("manufacturer", None),
        "manufacturer_label": source_manufacturer_label,
        "field_strength": row.get("field_strength", None),
        "field_strength_label": source_field_strength_label,
        "echo_time": float(row["echo_time"]),
        "echo_time_norm": source_echo_time_norm,
        "edit_timestep": 100,
        "guidance_scale": 1.0,
        "global_ssim": float(global_ssim),
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
    }

    np.save(output_dir / "original.npy", image)
    np.save(output_dir / "reconstruction_same_condition.npy", generated)

    editor.save_png(image, output_dir / "original.png")
    editor.save_png(generated, output_dir / "reconstruction_same_condition.png")

    save_side_by_side(
        image,
        generated,
        output_dir / "comparison_original_reconstruction_diff.png",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, indent=2)

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved outputs:")
    print(output_dir / "original.png")
    print(output_dir / "reconstruction_same_condition.png")
    print(output_dir / "comparison_original_reconstruction_diff.png")
    print(output_dir / "metrics.json")

    print("\nSTEP 2 COMPLETE")
    print("Send me the terminal output and the comparison image.")


if __name__ == "__main__":
    main()