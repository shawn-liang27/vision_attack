"""Merge m_attack_edge metrics.csv with vlm_eval.json -> per-arm success rate.

Hypothesis confirmed if edge_v1's concealment success climbs from pad0's rate
toward pad0.15's. For edge_v2 also reports mean grad_align (>0 cooperate, <0 the
added boundary term fights M-Attack).

    uv run python summarize_edge.py --dir results/edge
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--object", default="dog")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "metrics.csv")) as f:
        met = list(csv.DictReader(f))
    with open(os.path.join(args.dir, "vlm_eval.json")) as f:
        vj = json.load(f)
    model_key = next(iter(vj))
    detected = {n: v.get("detected") for n, v in vj[model_key].items()}

    by_arm = defaultdict(list)
    for r in met:
        by_arm[r["arm"]].append(r)

    order = ["pad0", "pad0.15", "edge_v1", "edge_v2"]
    arms = [a for a in order if a in by_arm] + [a for a in by_arm if a not in order]
    print(f"model={model_key}  object={args.object}\n")
    hdr = f"{'arm':<10}{'n':>3}{'conceal%':>10}{'cos->tgt':>11}{'gP(dog)':>10}{'roiP(dog)':>11}{'grad_align':>12}"
    print(hdr); print("-" * len(hdr))
    for arm in arms:
        rs = by_arm[arm]
        det = [detected.get(r["filename"]) for r in rs]
        det = [d for d in det if d is not None]
        conceal = 100 * (1 - np.mean(det)) if det else float("nan")
        cos_t = np.array([float(r["cos_to_target"]) for r in rs])
        gpd = np.array([float(r["global_p_dog"]) for r in rs])
        rpd = np.array([float(r["roi_p_dog"]) for r in rs])
        ga = np.array([float(r["grad_align"]) for r in rs])
        ga = ga[~np.isnan(ga)]
        ga_s = f"{ga.mean():>11.3f}" if len(ga) else f"{'—':>12}"
        print(f"{arm:<10}{len(rs):>3}{conceal:>9.0f}%{cos_t.mean():>8.3f}±{cos_t.std():.2f}"
              f"{gpd.mean():>7.3f}±{gpd.std():.2f}{rpd.mean():>8.3f}±{rpd.std():.2f}{ga_s}")

    print("\nreading: hypothesis CONFIRMED if edge_v1 conceal% >> pad0 and ~ pad0.15.")
    print("if edge_v2 grad_align < 0, the added boundary term is fighting M-Attack (as suppression did).")


if __name__ == "__main__":
    main()
