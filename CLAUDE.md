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

Python 3.11, PyTorch 2.4, transformers 4.45+, Hydra, `uv` as package manager.
Backbone: `facebook/dinov2-with-registers-base`, frozen, fp16, cached at
`/home/devops/.cache/huggingface/hub/`.
Detector: InsightFace SCRFD (`buffalo_l`). Tracker: `supervision` ByteTrack.

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
- OUC-CGE class distribution: 2396 / 1653 / 2115, verified against `train.csv`.
- MELD raw video lives at `/home/devops/socialarcnet-v2/data/meld/raw/MELD.Raw/MELD.Raw/`.
  Clip counts exceed label counts (test: 2747 files vs 2610 labels; dev: 1112 vs 1109) — this is a
  known MELD quirk. **Align by `(dialogue_id, utterance_id)` from the label CSV, never by directory
  listing.** Exclusions in `data/meld/bad_clips.txt` (`dia110_utt7.mp4` dev, `dia125_utt3.mp4` train).
- Two configs sharing a `results_dir` silently overwrite each other's checkpoints. Every condition
  gets its own `outputs/<run_id>/`. This has already cost this lab a paper's worth of confusion.
- The lab server is unreliable. Back up the feature cache via rclone the moment Phase 2 completes.
