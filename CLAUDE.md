# EPT — Project Constitution

Entity-Persistent Tokenization for video understanding. Target: ICFNDS 2026, IoT track,
deadline **1 September 2026**. Full plan: `docs/PLAN.md`. Pre-registration: `docs/DECISION_RULES.md`.

## Non-negotiables

1. **Decision rules are frozen.** `docs/DECISION_RULES.md` is committed before any run touches a test
   split. It is never edited after that commit. If reality forces a change, add a dated amendment
   section — never rewrite history.
2. **Test split is touched once per locked config.** All recipe search, architecture search, and
   debugging happens on validation. A test evaluation is a phase-gate event, logged with the commit
   hash of the config that produced it.
3. **Matched features across all conditions.** Every condition (A0–A6) consumes the *same* cached
   frozen DINOv2 features. Any measured difference must be attributable to tokenization or attention
   topology, never to encoder capacity. If a condition needs different features, it is a different
   experiment and must be flagged as such.
4. **Phase-boundary discipline.** No phase begins until the prior phase's gate criterion is written
   to `logs/GATES.md` with a timestamp. Report the gate result and stop; do not proceed on your own
   judgement that it "basically passed."
5. **Write-on-produce.** Every run writes `metrics.json` at completion. Paper numbers are read from
   `metrics.json` files, never transcribed from console output or from memory.
6. **Three seeds, always.** `{42, 1337, 2024}`. Every reported number is mean ± std over three seeds.
   A single-seed number never enters a table.
7. **Stop and show.** At each checkpoint in `docs/PHASE_PROMPTS.md`, present the result and wait.
   Do not chain phases.

## Stack

Python 3.11, PyTorch 2.7.0+cu128 (substituted for 2.4 in Phase 0 — Blackwell/sm_120 requires it;
see `docs/PLAN.md` §9 amendment, commit `d729a56`), transformers 4.45+, Hydra, `uv` as package manager.
Backbone: `facebook/dinov2-with-registers-base`, frozen, fp16, cached at
`/home/devops/.cache/huggingface/hub/`.
Detector: InsightFace SCRFD (`buffalo_l`). Tracker: `supervision` ByteTrack.

## Feature cache E_max — READ BEFORE TRAINING ANYTHING

**The OUC-CGE feature cache (`cache/features/ouccge/`) holds E=16 entity slots.
The LOCKED PRIMARY CONDITION (A1, and the pre-registered H1/H2 comparisons) REMAINS
E=8.** Train A1 — and every condition in the original A0–A6 matrix — by slicing the
cache to `[:8]`, never by using all 16 slots. 16 exists only for the extended
token-budget sweep (`E ∈ {1,2,4,8,16}`, `docs/DECISION_RULES.md` amendment,
2026-08-14). DAiSEE's cache is unaffected and stays at E=8 throughout (DAiSEE is
E=1 by construction; the cap was never binding there).

This is exactly the kind of detail that silently becomes a wrong result: loading
the cache and using all 16 slots for what's reported as the "primary" condition
would quietly change what A1 measures relative to the frozen pre-registration.
`tests/test_e_max_slice_equivalence.py` asserts `cache[:8]` is bitwise identical to
a from-scratch E_max=8 extraction — run it if you're ever unsure the slice is safe.

## Layout

```
configs/            Hydra configs, one per condition (A0–A6)
src/ept/            tokenization/, model/, train/, eval/
cache/features/     frozen DINOv2 features (.npy), rclone-backed-up
outputs/<run_id>/   config snapshot + metrics.json + checkpoints
logs/GATES.md       phase gate log
docs/               PLAN.md, DECISION_RULES.md, PHASE_PROMPTS.md, LOCKED_RECIPE.md
paper/              acmart sigconf sources
```

## Hard-won facts (do not rediscover)

- OUC-CGE: exclude `low/view2572.mp4` (missing) and `low/view2531.mp4` (moov atom not found).
  Test protocol is **771 clips, video-only**. Videos do contain audio; ignore it here.
- OUC-CGE: also exclude `low/view61.mp4`, `low/view1579.mp4`, `high/view1325.mp4` — found
  undecodable (reported frame count far exceeds what OpenCV can actually read) during the
  Phase 1 detector bake-off. All three are **train-split only** (not in `val.csv`/`test.csv`);
  the published 771-clip test protocol is unaffected.
- `onnxruntime.InferenceSession` defaults to using **all cores per session** when no
  `SessionOptions` is passed, and `insightface` exposes no way to set one through its public
  API. Under multiprocessing this oversubscribes savagely (measured: load average 587 on this
  60-core box at 48 workers). Cap it explicitly — monkeypatch a 1-thread `SessionOptions`
  default before constructing any `FaceAnalysis`/onnxruntime session, and keep worker count
  well under core count regardless. Cost this project 20 minutes once; should cost the next
  reader none.
- `ultralytics` YOLO silently resets torch's **intra-op** thread count to
  `min(8, ncpu)` internally on first `.predict()` call, overriding whatever
  `torch.set_num_threads(1)` was set beforehand. Also: with the default `fork`
  multiprocessing start method, worker processes inherit whatever native thread
  pools (BLAS/OpenCV/torch) the **parent** already initialized at module-import
  time — setting `OMP_NUM_THREADS`-style env vars inside a `Pool` initializer,
  after those libraries are already imported at module scope, is too late. Set
  thread-limiting env vars as the very first lines of the file (before any other
  import), and re-call `torch.set_num_threads(1)` before every single `.predict()`
  call, not just once at worker init. Measured impact: 20 workers went from
  6.0s/clip amortized (still-oversubscribed) to 0.32s/clip (~19x) after both fixes.
- OUC-CGE class distribution: 2396 / 1653 / 2115, verified against `train.csv`.
- MELD raw video lives at `/home/devops/socialarcnet-v2/data/meld/raw/MELD.Raw/MELD.Raw/`.
  Clip counts exceed label counts (test: 2747 files vs 2610 labels; dev: 1112 vs 1109) — this is a
  known MELD quirk. **Align by `(dialogue_id, utterance_id)` from the label CSV, never by directory
  listing.** Exclusions in `data/meld/bad_clips.txt` (`dia110_utt7.mp4` dev, `dia125_utt3.mp4` train).
- Two configs sharing a `results_dir` silently overwrite each other's checkpoints. Every condition
  gets its own `outputs/<run_id>/`. This has already cost this lab a paper's worth of confusion.
- The lab server is unreliable. Back up the feature cache via rclone the moment Phase 2 completes.
