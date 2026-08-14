# Skill: evaluation-and-artifacts

- Primary metric: macro-F1. Report accuracy and per-class F1 as secondary.
- Test evaluation is a phase-gate event: one per locked config, logged to `logs/GATES.md`
  with the config's commit hash.
- `results/summary.csv`: one row per condition, columns for mean and std across seeds.
- Efficiency table reports, separately: attention-stack GFLOPs, detection+tracking cost per
  clip, GPU latency (batch 1, fp16), CPU latency at threads {1, 8, 60}, peak memory.
  Folding detection cost into the model cost is a misrepresentation — never do it.
- Figures as vector PDF at single-column width.
- Before any number enters the paper, run a number-audit pass: every inline statistic and every
  table cell verified against the `metrics.json` it came from. This lab has caught four
  mismatches this way on a previous paper; assume there are more.
