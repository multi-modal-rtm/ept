# Skill: project-conventions

Bind on every task in this repo.

- Read `CLAUDE.md` before acting. Its "Hard-won facts" section overrides your priors about
  these datasets.
- Never edit `docs/DECISION_RULES.md` after its freeze commit. Append a dated amendment instead.
- Every run writes `outputs/<run_id>/metrics.json` and a snapshot of the resolved Hydra config.
- Every run gets a unique `results_dir`. Two configs pointing at one directory is a known
  failure mode in this lab and silently destroys checkpoints.
- Paper numbers come from `metrics.json`. Never transcribe from console output.
- Report mean ± std over seeds {42, 1337, 2024}. A single-seed number is never a result.
- At a stop-and-show checkpoint: present the artifact, state the gate result, and stop.
  Do not begin the next phase.
- If a gate fails, say so plainly and name the fallback from `docs/PLAN.md` §8. Do not
  reinterpret the criterion so that it passes.
