"""Consolidates the GFLOPs breakdown, detection throughput/decode-split
measurements, and existing per-condition latency into one end-to-end table
(results/efficiency_breakdown.csv) plus the E=1 "largest-face" cost
estimate. Pure aggregation + one clearly-labeled estimate -- no new model
training, no test data touched.
"""
import csv
import json
import os

REPO_ROOT = "/home/devops/ept"

with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency_combined.json")) as f:
    EFF = json.load(f)
with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_gflops_breakdown.json")) as f:
    GB = json.load(f)
with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_detection_throughput.json")) as f:
    DT = json.load(f)

SINGLE_STREAM_REF_MS = DT["single_stream_reference_ms"]  # 6276.9, scripts/meld_phase6_efficiency.py
THROUGHPUT_AMORTIZED_MS = DT["throughput_60way"]["amortized_ms_per_clip"]
DECODE_FRAC = DT["decode_vs_detect_split"]["decode_ms_mean"] / (
    DT["decode_vs_detect_split"]["decode_ms_mean"] + DT["decode_vs_detect_split"]["scrfd_detect_ms_mean"]
)
RESOLUTION_SCALING_ASSUMPTION = (320 / 640) ** 2  # standard quadratic CNN cost-vs-resolution rule of thumb


def main():
    gflops_by_cond = {r["condition"]: r for r in GB["condition_rows"]}
    eff_by_cond = {r["condition"]: r for r in EFF["condition_rows"]}

    rows = []
    for cond in ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]:
        g = gflops_by_cond[cond]
        e = eff_by_cond[cond]
        det_single = 0.0 if cond == "A0" else SINGLE_STREAM_REF_MS
        det_amortized = 0.0 if cond == "A0" else THROUGHPUT_AMORTIZED_MS
        model_ms = e["gpu_end_to_end_latency_ms"]
        rows.append({
            "condition": cond, "e": g["e"], "s": g["s"],
            "encoder_gflops": round(g["encoder_gflops"], 1),
            "fusion_attention_gflops": round(g["fusion_attention_gflops"], 4),
            "head_gflops": round(g["head_gflops"], 4),
            "fusion_attention_pct_of_total": round(g["fusion_attention_pct_of_grand_total"], 4),
            "fusion_attention_pct_of_fusion_stack": round(g["fusion_attention_pct_of_fusion_stack"], 2),
            "model_only_gpu_latency_ms": round(model_ms, 2),
            "detection_single_stream_ms": round(det_single, 1),
            "detection_60way_amortized_ms": round(det_amortized, 1),
            "e2e_single_stream_ms": round(model_ms + det_single, 2),
            "e2e_throughput_amortized_ms": round(model_ms + det_amortized, 2),
        })

    a1 = next(r for r in rows if r["condition"] == "A1")
    a0 = next(r for r in rows if r["condition"] == "A0")
    ratio_single_stream = a1["e2e_single_stream_ms"] / a0["e2e_single_stream_ms"]
    ratio_model_only = a1["model_only_gpu_latency_ms"] / a0["model_only_gpu_latency_ms"]
    ratio_throughput = a1["e2e_throughput_amortized_ms"] / a0["e2e_throughput_amortized_ms"]

    out_csv = os.path.join(REPO_ROOT, "results", "efficiency_breakdown.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved -> {out_csv}")
    for r in rows:
        print(f"  {r['condition']}: model_only={r['model_only_gpu_latency_ms']:.1f}ms "
              f"e2e_single_stream={r['e2e_single_stream_ms']:.1f}ms "
              f"e2e_throughput={r['e2e_throughput_amortized_ms']:.1f}ms "
              f"fusion_attn%={r['fusion_attention_pct_of_total']:.4f}%")

    print(f"\nA1-vs-A0 cost ratio (model-only, GPU, internally consistent): {ratio_model_only:.4f}x "
          f"(A1 is CHEAPER: {'yes' if ratio_model_only < 1 else 'no'})")
    print("NOTE: the e2e_single_stream_ms / e2e_throughput_amortized_ms columns above mix a GPU-"
          "measured model latency with a CPU-measured detection latency -- kept in the CSV for the "
          "raw numbers, but NOT a valid same-hardware cost ratio. See "
          "scripts/meld_phase6_regimes.py / outputs/meld_phase6_regimes.json for the three "
          "internally-consistent regimes (GPU-only, CPU-only at matched thread counts, and an "
          "explicitly-assumption-labeled realistic-deployment split) -- those are the ratios to cite.")

    # --- encoder-reduction finding: reported as its own positive result ---
    import csv as _csv
    with open(os.path.join(REPO_ROOT, "results", "summary.csv")) as f:
        macro_f1 = {r["condition"]: float(r["macro_f1_mean"]) for r in _csv.DictReader(f)}
    encoder_reduction = a0["encoder_gflops"] / a1["encoder_gflops"]
    macro_f1_gain = macro_f1["A1"] - macro_f1["A0"]
    print(f"\n=== encoder-reduction finding (positive result) ===")
    print(f"Entity cropping cuts per-crop-encoder cost {encoder_reduction:.2f}x "
          f"({a0['encoder_gflops']:.1f} -> {a1['encoder_gflops']:.1f} GFLOPs/clip) "
          f"AND improves test macro-F1 by {macro_f1_gain:+.4f} "
          f"({macro_f1['A0']:.4f} -> {macro_f1['A1']:.4f}), A0 -> A1.")

    # --- E=1 "largest face" estimate (clearly labeled, not measured) ---
    detect_only_ms = DT["decode_vs_detect_split"]["scrfd_detect_ms_mean"]
    decode_only_ms = DT["decode_vs_detect_split"]["decode_ms_mean"]
    ref_detect_ms = SINGLE_STREAM_REF_MS * (1 - DECODE_FRAC)  # apply measured decode fraction to the
    ref_decode_ms = SINGLE_STREAM_REF_MS * DECODE_FRAC          # tabulated 6276.9ms reference, for consistency
    estimated_scrfd_ms = ref_detect_ms * RESOLUTION_SCALING_ASSUMPTION
    estimated_total_ms = ref_decode_ms + estimated_scrfd_ms  # clustering dropped entirely (E=1 needs none)

    e1s2_row = next(r for r in EFF["budget_sweep_rows"] if r["e"] == 1 and r["s"] == 2)
    e1s2_model_ms = e1s2_row["gpu_end_to_end_latency_ms"]
    e1s2_e2e_current_detector = e1s2_model_ms + SINGLE_STREAM_REF_MS
    e1s2_e2e_estimated_detector = e1s2_model_ms + estimated_total_ms

    estimate = {
        "assumption": "resolution 640x640 -> 320x320 for single-largest-face detection, "
                       "cost scales (320/640)^2=0.25x (standard quadratic CNN cost-resolution "
                       "rule of thumb) -- NOT independently measured with an actually-built "
                       "lower-resolution detector variant. Clustering dropped entirely (E=1 has "
                       "no multi-identity resolution to do), not just made cheap -- clustering "
                       "was already measured negligible (2.5ms) so this saves little on its own.",
            "reference_decode_ms": round(ref_decode_ms, 1), "reference_detect_ms": round(ref_detect_ms, 1),
        "estimated_scrfd_ms_at_320": round(estimated_scrfd_ms, 1),
        "estimated_total_detection_ms": round(estimated_total_ms, 1),
        "estimated_reduction_factor": round(SINGLE_STREAM_REF_MS / estimated_total_ms, 2),
        "e1_s2_model_only_ms": round(e1s2_model_ms, 2),
        "e1_s2_e2e_with_current_full-res_detector_ms": round(e1s2_e2e_current_detector, 1),
        "e1_s2_e2e_with_estimated_largest-face_detector_ms": round(e1s2_e2e_estimated_detector, 1),
    }
    print("\n=== E=1 'largest face' detection cost ESTIMATE (not measured) ===")
    print("NOTE: e1_s2_e2e_*_ms figures below also mix GPU model latency with CPU-measured "
          "detection latency, same caveat as above -- treat as an order-of-magnitude estimate, "
          "not a regime-consistent number.")
    print(json.dumps(estimate, indent=2))

    out = {"rows": rows, "a1_vs_a0_ratio_model_only": ratio_model_only,
           "a1_vs_a0_ratio_e2e_single_stream": ratio_single_stream,
           "a1_vs_a0_ratio_e2e_throughput": ratio_throughput, "e1_largest_face_estimate": estimate}
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_e2e_breakdown.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> outputs/meld_phase6_e2e_breakdown.json")


if __name__ == "__main__":
    main()
