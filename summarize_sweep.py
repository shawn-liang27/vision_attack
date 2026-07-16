"""Merge m_attack_sweep metrics.csv with vlm_eval.json -> seed-averaged curves.

Per padding level: concealment success rate (fraction of seeds where the VLM
did NOT report the object) with binomial spread, plus mean +/- std of the
continuous signals (cos_to_target, global/ROI P(dog)) and perturbation density.
This is the real signal; a single run's non-monotonicity is mostly noise.

    uv run python summarize_sweep.py --dir results/sweep_h1
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    detected = {name: v.get("detected") for name, v in vj[model_key].items()}

    by_pad = defaultdict(list)
    for r in met:
        by_pad[float(r["pad"])].append(r)

    pads = sorted(by_pad)
    print(f"model={model_key}  object={args.object}\n")
    hdr = (f"{'pad':>6}{'cov%':>7}{'n':>4}{'conceal%':>10}{'cos->tgt':>12}"
           f"{'gP(dog)':>10}{'roiP(dog)':>11}{'|d|/eps':>9}{'cap%':>7}")
    print(hdr); print("-" * len(hdr))
    agg = []
    for pad in pads:
        rs = by_pad[pad]
        cov = float(rs[0]["coverage"]) * 100
        det = [detected.get(r["filename"]) for r in rs]
        det = [d for d in det if d is not None]
        conceal = 100 * (1 - np.mean(det)) if det else float("nan")  # success = NOT detected
        cos_t = np.array([float(r["cos_to_target"]) for r in rs])
        gpd = np.array([float(r["global_p_dog"]) for r in rs])
        rpd = np.array([float(r["roi_p_dog"]) for r in rs])
        dmean = np.array([float(r["delta_mean_frac"]) for r in rs])
        cap = np.array([float(r["frac_at_cap"]) for r in rs])
        agg.append(dict(pad=pad, cov=cov, n=len(rs), conceal=conceal,
                        cos_t=cos_t, gpd=gpd, rpd=rpd, dmean=dmean, cap=cap))
        print(f"{pad:>6g}{cov:>7.1f}{len(rs):>4}{conceal:>9.0f}%"
              f"{cos_t.mean():>8.3f}±{cos_t.std():.2f}{gpd.mean():>7.3f}±{gpd.std():.2f}"
              f"{rpd.mean():>8.3f}±{rpd.std():.2f}{dmean.mean():>9.2f}{cap.mean()*100:>6.0f}%")

    cov = [a["cov"] for a in agg]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(cov, [a["conceal"] for a in agg], "o-", color="tab:purple")
    ax[0].set_ylim(-5, 105); ax[0].set_xlabel("region coverage (%)")
    ax[0].set_ylabel("concealment success rate (%)")
    ax[0].set_title(f"seed-averaged success ({agg[0]['n']} seeds/level)")

    for a in agg:
        ax[1].errorbar(a["cov"], a["rpd"].mean(), yerr=a["rpd"].std(), fmt="o", color="tab:red")
        ax[1].errorbar(a["cov"], a["gpd"].mean(), yerr=a["gpd"].std(), fmt="s", color="tab:orange")
    ax[1].plot(cov, [a["rpd"].mean() for a in agg], "-", color="tab:red", label=f"ROI P({args.object})")
    ax[1].plot(cov, [a["gpd"].mean() for a in agg], "--", color="tab:orange", label=f"global P({args.object})")
    ax[1].set_xlabel("region coverage (%)"); ax[1].set_ylabel("P(dog) (mean±std)")
    ax[1].set_title("continuous signal (threshold-noise check)"); ax[1].legend()

    ax[2].plot(cov, [a["dmean"].mean() for a in agg], "o-", color="tab:green", label="mean |d|/eps")
    ax[2].plot(cov, [a["cap"].mean() * 100 for a in agg], "s--", color="tab:blue", label="% at L_inf cap")
    ax[2].set_xlabel("region coverage (%)"); ax[2].set_title("perturbation density (dilution check)")
    ax[2].legend()

    fig.suptitle(f"seeded padding sweep summary — {os.path.basename(args.dir)}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.dir, "sweep_summary.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
