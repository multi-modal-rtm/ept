"""Step 1: attempt onnxruntime-gpu / CUDAExecutionProvider for SCRFD face
detection on the RTX 5090. If it works, measure per-frame and per-clip
detection latency on GPU. If it fails, capture and report the EXACT error
-- no GPU number is estimated or guessed if this doesn't work.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = "/home/devops/ept"
OUT_PATH = os.path.join(REPO_ROOT, "outputs", "meld_phase6_gpu_detection_attempt.json")
N_CLIPS = 30


def main():
    result = {"success": False}
    try:
        import onnxruntime as ort
        result["onnxruntime_version"] = ort.__version__
        result["available_providers"] = ort.get_available_providers()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            result["error"] = (
                f"CUDAExecutionProvider not in available providers: {ort.get_available_providers()}. "
                f"onnxruntime version {ort.__version__} does not expose a CUDA execution provider "
                f"in this environment (CUDA 12.8, driver 595.84, RTX 5090 / sm_120)."
            )
            with open(OUT_PATH, "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
            return

        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            allowed_modules=["detection"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Confirm the detection model actually landed on CUDA, not silently CPU fallback.
        det_session = app.models["detection"].session
        actual_providers = det_session.get_providers()
        result["detection_model_providers"] = actual_providers
        if "CUDAExecutionProvider" not in actual_providers:
            result["error"] = (
                f"FaceAnalysis initialized but the detection model's session providers are "
                f"{actual_providers} -- CUDAExecutionProvider was requested but not used "
                f"(silent CPU fallback)."
            )
            with open(OUT_PATH, "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
            return

        from ept.tokenization.detect_cluster_meld import LABEL_CSVS, MELD_ROOT, SPLIT_DIRS, T, extract_frames
        import csv

        manifest = []
        with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS["test"])) as f:
            for row in csv.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS["test"], fname)
                if os.path.exists(path):
                    manifest.append(path)
                if len(manifest) >= N_CLIPS:
                    break

        # warmup
        frames = extract_frames(manifest[0], T)
        for _, (_, frame) in enumerate(frames):
            app.get(frame)

        clip_times, frame_times = [], []
        for path in manifest:
            frames = extract_frames(path, T)
            t0 = time.time()
            for _, (_, frame) in enumerate(frames):
                tf0 = time.time()
                app.get(frame)
                frame_times.append(time.time() - tf0)
            clip_times.append(time.time() - t0)

        import numpy as np
        result.update({
            "success": True,
            "n_clips": len(manifest),
            "per_frame_ms_mean": float(np.mean(frame_times) * 1000),
            "per_frame_ms_std": float(np.std(frame_times) * 1000),
            "per_clip_latency_ms": float(np.mean(clip_times) * 1000),
            "per_clip_latency_ms_std": float(np.std(clip_times) * 1000),
        })
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
