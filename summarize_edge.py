"""Generalization summary: per-object AND aggregate arm success rates.

Merges m_attack_edge metrics.csv with vlm_eval.json. The headline test:
does edge_v1 recover pad0.15 (>> pad0) ACROSS objects, not just the dog?

    uv run python summarize_edge.py --dir results/edge_gen
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

ARM_ORDER = ["pad0", "pad0.15", "edge_v1", "edge_v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "metrics.csv")) as f:
        met = list(csv.DictReader(f))
    with open(os.path.join(args.dir, "vlm_eval.json")) as f:
        vj = json.load(f)
    model_key = next(iter(vj))
    det = {n: v.get("detected") for n, v in vj[model_key].items()}
    has_id = "id" in met[0]

    def conceal(rows):
        d = [det.get(r["filename"]) for r in rows]
        d = [x for x in d if x is not None]
        return (100 * (1 - np.mean(d)) if d else float("nan")), len(d)

    def roi(rows):
        return np.mean([float(r.get("roi_p_obj", r.get("roi_p_dog", "nan"))) for r in rows])

    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in met)]

    if has_id:
        print(f"model={model_key}\n=== per-object concealment success rate (%) ===\n")
        objs = sorted({(r["id"], r.get("object", "")) for r in met})
        hdr = f"{'object':<22}" + "".join(f"{a:>10}" for a in arms)
        print(hdr); print("-" * len(hdr))
        for sid, obj in objs:
            cells = ""
            for a in arms:
                rs = [r for r in met if r["id"] == sid and r["arm"] == a]
                c, _ = conceal(rs)
                cells += f"{c:>9.0f}%" if not np.isnan(c) else f"{'n/a':>10}"
            print(f"{(obj or sid)[:21]:<22}{cells}")

        print("\n=== AGGREGATE across objects (the generalization headline) ===\n")
        hdr = f"{'arm':<10}{'conceal% (mean over objs)':>28}{'roiP(obj)':>12}"
        print(hdr); print("-" * len(hdr))
        for a in arms:
            per_obj = []
            for sid, _ in objs:
                rs = [r for r in met if r["id"] == sid and r["arm"] == a]
                c, _ = conceal(rs)
                if not np.isnan(c):
                    per_obj.append(c)
            m = np.mean(per_obj) if per_obj else float("nan")
            s = np.std(per_obj) if per_obj else float("nan")
            rp = roi([r for r in met if r["arm"] == a])
            print(f"{a:<10}{m:>18.0f}% ± {s:<4.0f}{rp:>12.3f}")
        print("\nGENERALIZES if edge_v1 aggregate >> pad0 and ~ pad0.15 across objects.")
    else:
        # single-sample fallback
        print(f"model={model_key}\n{'arm':<10}{'conceal%':>10}{'roiP':>10}")
        for a in arms:
            rs = [r for r in met if r["arm"] == a]
            c, n = conceal(rs)
            print(f"{a:<10}{c:>9.0f}%{roi(rs):>10.3f}  (n={n})")


if __name__ == "__main__":
    main()
