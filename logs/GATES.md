# EPT — Phase Gate Log

## Phase 0 — Bootstrap

- **Date:** 2026-08-14T10:47:15Z
- **Gate:** Decision rules committed before any run touches code or data.
- **Evidence:** `docs/DECISION_RULES.md` frozen at commit `b65063f03f978dd87bd55f1864ba6fd53ff6a885`
  ("Freeze pre-registration: hypotheses and decision rule (EPT / ICFNDS 2026)"), committed alone,
  before `src/`, `configs/`, or any other repo content existed.
- **Result:** PASS — pre-registration frozen before any run.

## Phase 1 — Detection + tracking, tracking QA

- **Date:** 2026-08-14T13:10:16Z
- **Gate:** ≥80% of clips yield ≥2 tracks with ≥50% frame coverage (`docs/PLAN.md` §7;
  else → §8 fallback to region-persistent tokens).
- **Evidence:** Detector switched to YOLO person-class (from SCRFD/buffalo_l) after the
  step-1 bake-off found SCRFD's failure mode pose-correlated and label-plausible —
  see `outputs/phase1_detector_bakeoff/REPORT.md`. Full detection+tracking run over all
  7700 non-excluded OUC-CGE clips (YOLO person + supervision ByteTrack), 1918.9s wall
  clock, 20 workers. Tracking QA: **98.5% of clips pass the gate overall** (low 100.0%,
  mid 100.0%, high 95.7%) — full numbers in `outputs/phase1_step2_summary/REPORT.md`.
- **Result:** PASS — clearly above the 80% threshold in aggregate and per class.
- **Flagged alongside, not a gate criterion but material to interpreting any later
  result:** the pre-registered mask-only control (`docs/DECISION_RULES.md` amendment,
  2026-08-14) came back well above the majority-class baseline (macro-F1 0.373 vs.
  0.190, logistic regression, VAL split, 3 seeds) — entity presence pattern alone
  carries real signal about the engagement label. Reported for discussion, not
  blocking the gate.
