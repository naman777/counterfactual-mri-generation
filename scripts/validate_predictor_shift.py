#!/usr/bin/env python3
"""
Phase 10 - Step 10.5: Validate Predictor Shift Logic.

Manual verification on 5 examples to confirm:
  1. Categorical gain = after_target_prob - before_target_prob is positive when editing worked
  2. EchoTime improvement = abs(before_error) - abs(after_error) is positive when editing worked
  3. Unchanged controls have near-zero gain (stability check)

Usage:
    python scripts/validate_predictor_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.engine import ExplainabilityEngine
from src.inference.counterfactual_editor import CounterfactualEditor


# Label decode maps
MFG = {0: "GE MEDICAL SYSTEMS", 1: "SIEMENS"}
FS = {0: "1.5T", 1: "3.0T"}


def validate_single_example(
    editor: CounterfactualEditor,
    engine: ExplainabilityEngine,
    image_path: str,
    source_cond: dict,
    target_cond: dict,
    control: str,
    edit_timestep: int,
    guidance_scale: float,
    seed: int,
    example_label: str,
):
    """Validate predictor shift for one example."""
    print(f"\n{'='*60}")
    print(f"  {example_label}")
    print(f"{'='*60}")

    # Load image
    original = np.load(image_path).astype(np.float32)
    original = np.clip(original, 0.0, 1.0)

    # Generate counterfactual
    cf_result = editor.generate_counterfactual(
        image_01=original,
        target_manufacturer_label=target_cond["manufacturer_label"],
        target_field_strength_label=target_cond["field_strength_label"],
        target_echo_time_norm=target_cond["echo_time_norm"],
        edit_timestep=edit_timestep,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    counterfactual = cf_result["generated_01"]

    # Compute explainability
    results = engine.compute_all(
        original=original,
        counterfactual=counterfactual,
        source_condition=source_cond,
        target_condition=target_cond,
    )

    pred_before = results["predictor_before"]
    pred_after = results["predictor_after"]
    pred_gain = results["predictor_gain"]
    stability = results["stability"]

    # Print detailed output
    print(f"\nControl being edited: {control}")
    print(f"Edit timestep: {edit_timestep}, Guidance scale: {guidance_scale}")
    print(f"Image shape: {original.shape}")

    print(f"\nSource condition:")
    print(f"  Manufacturer: {MFG[source_cond['manufacturer_label']]} (label={source_cond['manufacturer_label']})")
    print(f"  Field Strength: {FS[source_cond['field_strength_label']]} (label={source_cond['field_strength_label']})")
    print(f"  EchoTime norm: {source_cond['echo_time_norm']:.4f}")

    print(f"\nTarget condition:")
    print(f"  Manufacturer: {MFG[target_cond['manufacturer_label']]} (label={target_cond['manufacturer_label']})")
    print(f"  Field Strength: {FS[target_cond['field_strength_label']]} (label={target_cond['field_strength_label']})")
    print(f"  EchoTime norm: {target_cond['echo_time_norm']:.4f}")

    print(f"\nOriginal image predictor output:")
    print(f"  Manufacturer: {pred_before['mfg_pred_name']} (GE={pred_before['mfg_prob_GE']:.4f}, SIEMENS={pred_before['mfg_prob_SIEMENS']:.4f})")
    print(f"  Field Strength: {pred_before['fs_pred_name']} (1.5T={pred_before['fs_prob_1p5T']:.4f}, 3.0T={pred_before['fs_prob_3p0T']:.4f})")
    print(f"  EchoTime norm: {pred_before['echo_norm_pred']:.4f}")

    print(f"\nCounterfactual image predictor output:")
    print(f"  Manufacturer: {pred_after['mfg_pred_name']} (GE={pred_after['mfg_prob_GE']:.4f}, SIEMENS={pred_after['mfg_prob_SIEMENS']:.4f})")
    print(f"  Field Strength: {pred_after['fs_pred_name']} (1.5T={pred_after['fs_prob_1p5T']:.4f}, 3.0T={pred_after['fs_prob_3p0T']:.4f})")
    print(f"  EchoTime norm: {pred_after['echo_norm_pred']:.4f}")

    print(f"\nGain analysis:")
    print(f"  Manufacturer gain: {pred_gain['mfg_gain']:+.6f} (hit target: {pred_gain['mfg_hit']})")
    print(f"  Field Strength gain: {pred_gain['fs_gain']:+.6f} (hit target: {pred_gain['fs_hit']})")
    print(f"  EchoTime improvement: {pred_gain['echo_improvement']:+.6f} (moved closer: {pred_gain['echo_moved_closer']})")

    print(f"\nImage quality:")
    print(f"  Global SSIM: {results['global_ssim']:.4f}")
    print(f"  Mean Abs Diff: {results['mean_abs_diff']:.4f}")
    print(f"  Hist L1: {results['hist_l1']:.4f}")

    # Validation checks
    print(f"\n--- VALIDATION CHECKS ---")
    checks_passed = 0
    checks_total = 0

    if control == "field_strength":
        # Check 1: FS gain should be positive
        checks_total += 1
        if pred_gain["fs_gain"] > 0:
            print(f"  [PASS] Field strength gain is positive: {pred_gain['fs_gain']:+.6f}")
            checks_passed += 1
        else:
            print(f"  [FAIL] Field strength gain is NOT positive: {pred_gain['fs_gain']:+.6f}")

        # Check 2: Manufacturer should stay stable (not being edited)
        checks_total += 1
        if stability["mfg_stable"]:
            print(f"  [PASS] Manufacturer is stable (not being edited)")
            checks_passed += 1
        else:
            print(f"  [FAIL] Manufacturer changed (unexpected)")

    elif control == "echo_time":
        # Check 1: Echo improvement should be positive
        checks_total += 1
        if pred_gain["echo_improvement"] > 0:
            print(f"  [PASS] EchoTime improvement is positive: {pred_gain['echo_improvement']:+.6f}")
            checks_passed += 1
        else:
            print(f"  [FAIL] EchoTime improvement is NOT positive: {pred_gain['echo_improvement']:+.6f}")

        # Check 2: Manufacturer should stay stable
        checks_total += 1
        if stability["mfg_stable"]:
            print(f"  [PASS] Manufacturer is stable")
            checks_passed += 1
        else:
            print(f"  [FAIL] Manufacturer changed")

        # Check 3: Field strength should stay stable
        checks_total += 1
        if stability["fs_stable"]:
            print(f"  [PASS] Field strength is stable")
            checks_passed += 1
        else:
            print(f"  [FAIL] Field strength changed")

    elif control == "manufacturer":
        # Check 1: Manufacturer gain should be positive
        checks_total += 1
        if pred_gain["mfg_gain"] > 0:
            print(f"  [PASS] Manufacturer gain is positive: {pred_gain['mfg_gain']:+.6f}")
            checks_passed += 1
        else:
            print(f"  [FAIL] Manufacturer gain is NOT positive: {pred_gain['mfg_gain']:+.6f}")

        # Check 2: Field strength should stay stable
        checks_total += 1
        if stability["fs_stable"]:
            print(f"  [PASS] Field strength is stable")
            checks_passed += 1
        else:
            print(f"  [FAIL] Field strength changed")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")

    return {
        "example_label": example_label,
        "control": control,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "all_passed": checks_passed == checks_total,
        "global_ssim": results["global_ssim"],
        "mean_abs_diff": results["mean_abs_diff"],
        "mfg_gain": pred_gain["mfg_gain"],
        "fs_gain": pred_gain["fs_gain"],
        "echo_improvement": pred_gain["echo_improvement"],
    }


def main():
    print("=" * 70)
    print("Phase 10 - Step 10.5: Predictor Shift Validation")
    print("=" * 70)

    # Load test CSV
    test_csv = str(PROJECT_ROOT / "data/processed/ax_t1_128/splits/test.csv")
    df = pd.read_csv(test_csv)
    print(f"\nLoaded {len(df)} test samples")

    # Load models
    print("\nLoading diffusion model...")
    editor = CounterfactualEditor()

    print("\nLoading explainability engine...")
    engine = ExplainabilityEngine()

    # Select 5 diverse examples (try to get different patients, conditions)
    np.random.seed(42)

    # Get diverse examples
    examples = []

    # Example 1: GE 1.5T -> SIEMENS 1.5T (manufacturer edit)
    ge_15 = df[(df["manufacturer_label"] == 0) & (df["field_strength_label"] == 0)]
    if len(ge_15) > 0:
        row = ge_15.iloc[0]
        source = {"manufacturer_label": 0, "field_strength_label": 0, "echo_time_norm": float(row["echo_time_norm"])}
        target = {"manufacturer_label": 1, "field_strength_label": 0, "echo_time_norm": float(row["echo_time_norm"])}
        examples.append((row, source, target, "manufacturer", 300, 5.0, "Ex1: GE->SIEMENS (mfg edit)"))

    # Example 2: 1.5T -> 3.0T (field strength edit)
    fs_15 = df[df["field_strength_label"] == 0]
    if len(fs_15) > 1:
        row = fs_15.iloc[1]
        source = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": 0, "echo_time_norm": float(row["echo_time_norm"])}
        target = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": 1, "echo_time_norm": float(row["echo_time_norm"])}
        examples.append((row, source, target, "field_strength", 100, 3.0, "Ex2: 1.5T->3.0T (fs edit)"))

    # Example 3: 3.0T -> 1.5T (field strength edit, other direction)
    fs_30 = df[df["field_strength_label"] == 1]
    if len(fs_30) > 0:
        row = fs_30.iloc[0]
        source = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": 1, "echo_time_norm": float(row["echo_time_norm"])}
        target = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": 0, "echo_time_norm": float(row["echo_time_norm"])}
        examples.append((row, source, target, "field_strength", 100, 3.0, "Ex3: 3.0T->1.5T (fs edit)"))

    # Example 4: High echo -> Low echo (echo time edit)
    high_echo = df[df["echo_time_norm"] > 0.5]
    if len(high_echo) > 0:
        row = high_echo.iloc[0]
        source = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": float(row["echo_time_norm"])}
        target = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": 0.15}
        examples.append((row, source, target, "echo_time", 100, 1.0, "Ex4: High->Low echo time"))

    # Example 5: Low echo -> High echo (echo time edit, other direction)
    low_echo = df[df["echo_time_norm"] < 0.3]
    if len(low_echo) > 0:
        row = low_echo.iloc[0]
        source = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": float(row["echo_time_norm"])}
        target = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": 0.85}
        examples.append((row, source, target, "echo_time", 100, 1.0, "Ex5: Low->High echo time"))
    else:
        # Fallback: use medium echo
        mid_echo = df[df["echo_time_norm"] <= 0.5]
        if len(mid_echo) > 0:
            row = mid_echo.iloc[0]
            source = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": float(row["echo_time_norm"])}
            target = {"manufacturer_label": int(row["manufacturer_label"]), "field_strength_label": int(row["field_strength_label"]), "echo_time_norm": 0.85}
            examples.append((row, source, target, "echo_time", 100, 1.0, "Ex5: Mid->High echo time"))

    print(f"\nSelected {len(examples)} examples for validation\n")

    # Run validation
    all_results = []
    for row, source, target, control, edit_t, gs, label in examples:
        image_path = str(PROJECT_ROOT / row["image_path"])
        result = validate_single_example(
            editor=editor,
            engine=engine,
            image_path=image_path,
            source_cond=source,
            target_cond=target,
            control=control,
            edit_timestep=edit_t,
            guidance_scale=gs,
            seed=42,
            example_label=label,
        )
        all_results.append(result)

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_passed = sum(r["checks_passed"] for r in all_results)
    total_checks = sum(r["checks_total"] for r in all_results)
    all_examples_passed = all(r["all_passed"] for r in all_results)

    for r in all_results:
        status = "[PASS] PASS" if r["all_passed"] else "[FAIL] FAIL"
        print(f"  {status} | {r['example_label']} | "
              f"SSIM={r['global_ssim']:.4f} | "
              f"{r['checks_passed']}/{r['checks_total']} checks")

    print(f"\nOverall: {total_passed}/{total_checks} checks passed")
    if all_examples_passed:
        print("[PASS] ALL EXAMPLES PASSED - predictor shift logic is validated!")
    else:
        print("[FAIL] SOME CHECKS FAILED - review before running Phase 12")

    # Save results
    output_dir = PROJECT_ROOT / "reports" / "explainability" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictor_shift_validation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
