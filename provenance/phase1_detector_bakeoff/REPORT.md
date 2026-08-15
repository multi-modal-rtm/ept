# Phase 1, Step 1 — Detector bake-off (SCRFD vs YOLO person) on 200-clip OUC-CGE subsample

**Sample:** 200 clips stratified by class from `train.csv` (proportional to population),
seed 42. Population (post-exclusion of the two documented bad files):
low=2395, mid=1653, high=2115. Quotas drawn: low=77, mid=54, high=69.
T=32 frames/clip, uniformly sampled by frame index.

**Wall clock:** 1350.0s (22.5 min) for 200 clips × 32 frames × 2 detectors, 20 worker
processes, CPU only. First run at 48 workers drove load average to 587 on this 60-core
box — `onnxruntime.InferenceSession` defaults to using all cores per session with no
`SessionOptions`, and `insightface` doesn't expose that control, so 48 processes each
tried to claim ~60 threads. Fixed by monkeypatching a 1-thread `SessionOptions` default
and `torch.set_num_threads(1)`; re-run at 20 workers stayed well within budget.

**New finding, not previously catalogued:** during stratified sampling, 3 additional
clips (beyond the two documented bad files) failed a basic decode-readability check —
`low/view61.mp4`, `low/view1579.mp4`, `high/view1325.mp4` — all report a `CAP_PROP_FRAME_COUNT`
far higher than what OpenCV can actually decode (e.g. `view1325.mp4` reports 9 frames but
even those don't fully grab). These were backfilled with valid replacements from the same
class and excluded from all statistics below. Recommend adding these three to the
exclusion list in `CLAUDE.md` before Phase 1 step 2's full run — see recommendation below.

## Per-class detection statistics

| Detector | Class | n clips | mean det/frame | mean within-clip variance | % zero-detection frames |
|---|---|---:|---:|---:|---:|
| SCRFD | low  | 77 | 3.663 | 0.390 | 1.2% |
| SCRFD | mid  | 54 | 3.543 | 0.338 | 5.8% |
| SCRFD | high | 69 | 4.405 | 0.516 | 0.5% |
| **SCRFD spread (max−min mean det/frame)** | | | **0.861** | | |
| YOLO (person) | low  | 77 | 6.916 | 0.883 | 0.0% |
| YOLO (person) | mid  | 54 | 6.954 | 0.682 | 0.0% |
| YOLO (person) | high | 69 | 7.276 | 0.835 | 0.0% |
| **YOLO spread (max−min mean det/frame)** | | | **0.360** | | |

**SCRFD's detection rate is not uniform across engagement classes.** Mid-engagement
clips show a zero-detection-frame rate (5.8%) roughly 5× that of high (0.5%) and nearly
5× that of low (1.2%), and mid also has the lowest mean detections/frame despite not
being the class with fewest people. YOLO's spread across classes is less than half
SCRFD's (0.36 vs 0.86 det/frame), and it never returns a fully empty frame in this
sample (0.0% at all three classes).

## Manual recall check (10 sampled frames)

Viewed the raw frame images directly (no detector overlay) and counted visible people,
compared against both detectors' counts for that exact frame.

| # | Class | Manual count | SCRFD | YOLO | Note |
|---|---|---:|---:|---:|---|
| 0 | low  | ~7 | 0 | 4 | All 7 heads down/turned from camera — genuinely no visible faces. YOLO recovers 4/7 bodies, still undercounts under heavy occlusion. |
| 1 | low  | ~7 | 0 | 6 | Same room/pose type — heads down at laptops/phones. YOLO 6/7. |
| 2 | mid  | ~7 | 0 | 7 | Heads down/away. YOLO matches exactly. |
| 3 | mid  | ~7 | 0 | 7 | Heads down/away. YOLO matches exactly. |
| 4 | high | ~7 | 0 | 7 | Heads down/away, one person looking up/back (not frontal). YOLO matches. |
| 5 | high | ~7 | 0 | 7 | Same scene, later frame. YOLO matches. |
| 6 | low  | ~7 | 5 | 7 | Mixed poses, several faces visible (~3/4 view). SCRFD recovers 5/7. |
| 7 | low  | ~8–9 | 8 | 11 | Classroom desk rows, mostly side-profile faces. SCRFD close to manual count; YOLO appears to **overcount** here (desks/equipment or partial occlusions triggering extra person boxes). |
| 8 | mid  | ~6 | 5 | 7 | Round table, most faces at least partially visible. SCRFD 5/6, YOLO 7/6 (slight overcount). |
| 9 | high | ~5–6 | 2 | 6 | One person fully back-turned (foreground), several faces down at laptop. SCRFD 2/5, YOLO matches. |

**Reading:** whenever every visible person in frame has their head down or turned away
from the camera (frames 0–5, all recognizable as the same handful of recurring
table/room setups in this corpus), SCRFD returns exactly zero, every time — it is not
noisy under-detection, it is complete failure conditioned on head pose. YOLO degrades
gracefully in the same frames (recovers 4–7 of ~7 true people) because it keys on
torso/back silhouette, not facial features. In frames with more frontal/side face
visibility (6, 8, 9), SCRFD's recall is workable (33–83%) but still below YOLO's. YOLO's
failure mode is the opposite one — mild overcounting in cluttered desk-row scenes (frame
7: 11 vs. ~8–9 true), plausibly from partial/occluded torsos or desk equipment.

## Recommendation

**Switch the primitive detector to YOLO (person-class) for the Phase 1 step 2 full run.**

This is exactly the risk `docs/PLAN.md` §8 anticipated ("Tracking fails on small/occluded
classroom faces → switch face → full-person detection") and what `.claude/skills/ept-data.md`
already lists as the fallback path. The bake-off shows the failure mode is not merely low
yield but **systematically pose-correlated**: SCRFD goes fully blind whenever a group's
heads are down (a posture plausibly associated with the engagement label — writing/typing
vs. looking away/disengaged), and this shows up as class-dependent zero-frame rates and a
2.4× larger cross-class spread than YOLO. Adopting YOLO does **not** eliminate this
concern outright (mask-only control from step 0c stays fully warranted, and YOLO's
mild overcounting on cluttered desk rows needs watching in the tracking QA), but it
removes the dominant, most severe source of pose-conditioned missingness before step 2's
full run.

**Also recommend:** add `low/view61.mp4`, `low/view1579.mp4`, `high/view1325.mp4` to
`CLAUDE.md`'s exclusion list alongside the two documented bad files, before running
step 2 over the full corpus.
