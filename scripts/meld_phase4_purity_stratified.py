"""Pre-registered secondary analysis (docs/DECISION_RULES.md, 2026-08-15):
A1-A2 within each purity tercile of MELD's test split. Terciles were fixed
before this ran (outputs/meld_purity/test_purity.json: p33=0.8008,
p67=0.9067, 870/870/870 clips) -- this script only stratifies and reuses
scripts/meld_phase4_statistics.py's paired-bootstrap machinery per tercile.
Secondary: does not alter the branch decision.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meld_phase4_statistics import paired_bootstrap_margin

REPO_ROOT = "/home/devops/ept"
PURITY_PATH = os.path.join(REPO_ROOT, "outputs", "meld_purity", "test_purity.json")
N_BOOTSTRAP = 10000


def load_terciles():
    with open(PURITY_PATH) as f:
        purity = json.load(f)
    p33, p67 = purity["tercile_p33"], purity["tercile_p67"]
    low, mid, high = set(), set(), set()
    for r in purity["results"]:
        cid, p = r["clip_id"], r["purity"]
        if p < p33:
            low.add(cid)
        elif p < p67:
            mid.add(cid)
        else:
            high.add(cid)
    return {"low": low, "mid": mid, "high": high}, p33, p67


def main():
    terciles, p33, p67 = load_terciles()
    print(f"tercile cutpoints: p33={p33:.4f} p67={p67:.4f}")
    for name, ids in terciles.items():
        print(f"  {name}: {len(ids)} clips")

    results = {}
    for name in ("low", "mid", "high"):
        r = paired_bootstrap_margin("A1", "A2", n_bootstrap=N_BOOTSTRAP, clip_id_filter=terciles[name])
        results[name] = r
        print(f"\n=== {name} purity tercile (n={r['n_items']}) ===")
        print(f"mean_margin={r['mean_margin']:.4f} 95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
              f"excludes_zero={r['ci_excludes_zero']}")

    margins = [results["low"]["mean_margin"], results["mid"]["mean_margin"], results["high"]["mean_margin"]]
    monotone = margins[0] <= margins[1] <= margins[2]
    print(f"\nmargins by tercile (low, mid, high): {[f'{m:.4f}' for m in margins]}")
    print(f"monotone non-decreasing low->high: {monotone}")

    out = {"tercile_p33": p33, "tercile_p67": p67,
           "tercile_counts": {k: len(v) for k, v in terciles.items()},
           "results": results, "margins_low_mid_high": margins, "monotone": monotone}
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase4_purity_stratified.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
