"""Consolidates the three efficiency-measurement runs (meld_phase6_efficiency.py
+ two ad-hoc backbone-latency scripts) into results/efficiency.csv and a
combined JSON. No new measurement here -- pure aggregation of files already
on disk.
"""
import csv
import json
import os

REPO_ROOT = "/home/devops/ept"

with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency.json")) as f:
    EFF = json.load(f)
with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_backbone_gpu_latency.json")) as f:
    BACKBONE_GPU = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_backbone_cpu_latency.json")) as f:
    BACKBONE_CPU = {int(k): v for k, v in json.load(f).items()}

# Map each condition's rounded avg-crop count to the nearest measured backbone batch point.
CONDITION_BACKBONE_BATCH = {"A0": 128, "A1": 28, "A2": 28, "A3": 28, "A4": 28, "A5": 28, "mask_only": 28}
BUDGET_BACKBONE_GPU_BATCH = {1: 14, 2: 22, 4: 26, 6: 28, 8: 29}  # E -> nearest measured GPU batch


def main():
    rows = []
    for r in EFF["condition_rows"]:
        cond = r["condition"]
        bb = CONDITION_BACKBONE_BATCH[cond]
        backbone_gpu_ms = BACKBONE_GPU[bb]["latency_ms"]
        end_to_end_gpu_ms = backbone_gpu_ms + r["gpu_batch1_fp16_latency_ms"]
        backbone_cpu = BACKBONE_CPU[bb]
        rows.append({
            "condition": cond, "e": r["e"], "s": r["s"], "tokens": r["tokens"],
            "avg_raw_crops_per_clip": round(r["avg_raw_crops_per_clip"], 1),
            "backbone_gflops_per_clip": round(r["backbone_gflops_per_clip"], 2),
            "attention_stack_gflops": round(r["attention_stack_gflops"], 4),
            "total_model_gflops_per_clip": round(r["total_model_gflops_per_clip"], 2),
            "attention_stack_params": r["attention_stack_params"],
            "gpu_attn_only_latency_ms": round(r["gpu_batch1_fp16_latency_ms"], 3),
            "gpu_backbone_latency_ms": round(backbone_gpu_ms, 2),
            "gpu_end_to_end_latency_ms": round(end_to_end_gpu_ms, 2),
            "gpu_peak_mem_mb": round(r["gpu_peak_mem_mb"], 1),
            "cpu_end_to_end_ms_threads1": round(backbone_cpu["1"] + r["cpu_latency_ms"]["1"], 1),
            "cpu_end_to_end_ms_threads8": round(backbone_cpu["8"] + r["cpu_latency_ms"]["8"], 1),
            "cpu_end_to_end_ms_threads60": round(backbone_cpu["60"] + r["cpu_latency_ms"]["60"], 1),
            "detection_clustering_ms_per_clip": (
                0.0 if cond == "A0" else round(EFF["detection_clustering_latency"]["total_ms_mean"], 1)
            ),
        })

    out_csv = os.path.join(REPO_ROOT, "results", "efficiency.csv")
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved -> {out_csv}")
    for r in rows:
        print(f"  {r['condition']}: tokens={r['tokens']} total_gflops={r['total_model_gflops_per_clip']:.1f} "
              f"gpu_e2e={r['gpu_end_to_end_latency_ms']:.1f}ms cpu60_e2e={r['cpu_end_to_end_ms_threads60']:.1f}ms "
              f"det+cluster={r['detection_clustering_ms_per_clip']:.1f}ms")

    budget_rows = []
    for r in EFF["budget_sweep_rows"]:
        bb = BUDGET_BACKBONE_GPU_BATCH[r["e"]]
        backbone_gpu_ms = BACKBONE_GPU[bb]["latency_ms"]
        budget_rows.append({
            "e": r["e"], "s": r["s"], "tokens": r["tokens"],
            "avg_raw_crops_per_clip": round(r["avg_raw_crops_per_clip"], 1),
            "total_model_gflops_per_clip": round(r["total_model_gflops_per_clip"], 2),
            "gpu_end_to_end_latency_ms": round(backbone_gpu_ms + r["gpu_batch1_fp16_latency_ms"], 2),
            "detection_clustering_ms_per_clip": round(EFF["detection_clustering_latency"]["total_ms_mean"], 1),
        })
    out_csv2 = os.path.join(REPO_ROOT, "results", "efficiency_budget_sweep.csv")
    with open(out_csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        w.writeheader()
        for r in budget_rows:
            w.writerow(r)
    print(f"saved -> {out_csv2}")

    combined = {
        "backbone_gflops_per_crop": EFF["backbone_gflops_per_crop"], "backbone_params": EFF["backbone_params"],
        "detection_clustering_latency": EFF["detection_clustering_latency"],
        "backbone_gpu_latency_by_batch": BACKBONE_GPU, "backbone_cpu_latency_by_batch": BACKBONE_CPU,
        "condition_rows": rows, "budget_sweep_rows": budget_rows,
    }
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency_combined.json"), "w") as f:
        json.dump(combined, f, indent=2)


if __name__ == "__main__":
    main()
