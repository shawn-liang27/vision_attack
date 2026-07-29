"""Stage 31: is object-ness LOW-DIMENSIONAL? SVD the per-patch object-direction vectors we
already have -- no collection, no optimization, no LLaVA.

For each existing image pair, at the read layer (hidden_states[-2], raw), over the object-region
patches:
    d_i = h_clean[i] - h_inpaint[i]          # the direction that must be removed at patch i
Stack {d_i} -> (n_patches x 1024), SVD, look at the spectrum.

  * top few components carry most energy -> object-ness is low-rank; a rank-k subspace-projection
    removal loss ( min ||U_k^T (h_adv - h_inpaint)||^2 ) can be built from THESE images and tested
    today -- collection is unnecessary; the plot's k@90/95% is the k to use.
  * spectrum flat / high participation ratio -> object-ness is high-dimensional here; a shared
    low-rank target won't capture it and collection (more pairs) may be needed.

Also reports the cross-object subspace overlap (principal angles between each object's top-k
directions): high overlap => a SHARED object subspace across dog/cat/car (one U generalizes);
low => object-specific subspaces (per-object U, still testable today).

Reports THREE spectra: RAW {d_i} (magnitude-weighted -- hijacked by massive-activation "sink"
tokens, so misleadingly low-rank), UNIT {d_i/||d_i||} (direction-only), and UNIT SINK-EXCLUDED
(sinks removed by clean-norm ||h_clean_i|| > sink_mult*median). The last is the trustworthy
effective rank. A few CLIP tokens carry ~10x norm and dominate raw SVD; excluding them (and/or
unit-normalizing) is required for the object-subspace rank to mean anything.

    uv run python object_subspace.py --dataset dataset.jsonl --objects dog,cat,car
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformers import CLIPModel, CLIPProcessor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def spectrum_stats(D):
    """D: (n, d) matrix of row-vectors. Return singular values + explained-variance stats."""
    S = torch.linalg.svdvals(D.float())                      # descending
    ev = (S ** 2) / (S ** 2).sum().clamp_min(1e-12)          # explained-variance ratio per component
    cum = torch.cumsum(ev, 0)
    k = {p: int((cum < p).sum().item()) + 1 for p in (0.90, 0.95, 0.99)}
    pr = ((S ** 2).sum() ** 2 / (S ** 4).sum().clamp_min(1e-12)).item()   # participation ratio = effective rank
    return S.cpu().numpy(), ev.cpu().numpy(), cum.cpu().numpy(), k, pr


def top_subspace(D, k):
    """orthonormal basis (k x d) for the top-k right-singular subspace of D."""
    _, _, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    return Vh[:k]                                            # (k, d), rows orthonormal


def subspace_overlap(A, B):
    """mean cos of principal angles between row-spaces of A (kxd) and B (kxd), in [0,1]."""
    s = torch.linalg.svdvals(A @ B.T)                        # cos(principal angles)
    return s.clamp(0, 1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat,car")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--thresh", type=float, default=0.3, help="object-token coverage threshold")
    ap.add_argument("--sink-mult", type=float, default=4.0, help="||h_clean|| > sink_mult*median => massive-activation sink token (excluded)")
    ap.add_argument("--overlap-k", type=int, default=5, help="top-k subspace for cross-object principal angles")
    ap.add_argument("--outdir", default="results/object_subspace")
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()
    clip.requires_grad_(False)
    ipc = CLIPProcessor.from_pretrained(args.surrogate).image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"surrogate={args.surrogate} grid={GRID}; d_i at hidden_states[-2] (read layer), object region cov>{args.thresh}")

    def vis(x01):
        return clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                 output_hidden_states=True).hidden_states[-2][:, 1:].squeeze(0)  # (P, D) raw

    def load01(p):
        im = Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    results = {}   # obj -> dict(S, ev, cum, k, pr, Sn, evn, cumn, kn, prn, Vk, Vkn, n)
    for s in samples:
        obj = s["object"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{obj}] missing mask/inpaint; skip"); continue
        x0, xR = load01(s["image"]), load01(s["target"])
        with torch.no_grad():
            hc = vis(x0); d = (hc - vis(xR)).detach()         # (P, D)
            cn = hc.norm(dim=-1)                              # clean feature norm -> sinks are intrinsic outliers
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        oi = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > args.thresh).reshape(-1)).to(DEVICE)
        sink = cn > args.sink_mult * cn.median()             # massive-activation tokens (global, by clean norm)
        keep = oi & (~sink)                                  # object-region tokens minus sinks
        n_sink = int(sink.sum()); sink_in_region = int((sink & oi).sum())
        D = d[oi]                                            # (n, 1024) raw region
        Dn = F.normalize(D, dim=-1)                          # unit (direction-only)
        De = F.normalize(d[keep], dim=-1)                    # unit, sinks excluded
        S, ev, cum, k, pr = spectrum_stats(D)
        Sn, evn, cumn, kn, prn = spectrum_stats(Dn)
        Se, eve, cume, ke, pre = spectrum_stats(De)
        kk = min(args.overlap_k, int(keep.sum()))
        results[obj] = dict(S=S, cum=cum, k=k, pr=pr, cumn=cumn, kn=kn, prn=prn, cume=cume, ke=ke, pre=pre,
                            n=D.shape[0], n_sink=n_sink, sink_in_region=sink_in_region,
                            Vk=top_subspace(F.normalize(d[keep], dim=-1), kk))   # cross-object uses sink-excluded unit dirs
        print(f"\n=== {obj}: n={D.shape[0]} object patches; sinks: {n_sink} total, {sink_in_region} inside the region ===")
        print(f"  RAW  d_i          : k@90%={k[0.90]:>2}  k@95%={k[0.95]:>2}  k@99%={k[0.99]:>2}  PR={pr:.1f}   (magnitude-weighted; sink-skewed)")
        print(f"  UNIT d_i          : k@90%={kn[0.90]:>2}  k@95%={kn[0.95]:>2}  k@99%={kn[0.99]:>2}  PR={prn:.1f}   (direction-only)")
        print(f"  UNIT sink-excluded: k@90%={ke[0.90]:>2}  k@95%={ke[0.95]:>2}  k@99%={ke[0.99]:>2}  PR={pre:.1f}   (the trustworthy number)")

    # cross-object subspace overlap (top-k principal angles), on sink-excluded unit directions
    objs = list(results.keys())
    if len(objs) >= 2:
        print(f"\n=== cross-object top-{args.overlap_k} subspace overlap, sink-excluded unit dirs "
              f"(1=identical direction space, 0=orthogonal) ===")
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                o = subspace_overlap(results[objs[i]]["Vk"], results[objs[j]]["Vk"])
                print(f"    {objs[i]:<4}-{objs[j]:<4}: {o:.3f}")

    # plot: singular-value spectrum (log) + cumulative explained variance
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"dog": "tab:blue", "cat": "tab:orange", "car": "tab:green", "airplane": "tab:red", "sign": "tab:purple"}
    for obj, r in results.items():
        c = colors.get(obj, "gray")
        a1.semilogy(range(1, len(r["S"]) + 1), r["S"] / r["S"][0], "-o", ms=3, color=c, label=obj)
        a2.plot(range(1, len(r["cum"]) + 1), r["cum"], "-", color=c, label=f"{obj} raw")
        a2.plot(range(1, len(r["cumn"]) + 1), r["cumn"], "--", color=c, alpha=0.7, label=f"{obj} unit")
        a2.plot(range(1, len(r["cume"]) + 1), r["cume"], ":", color=c, lw=2, label=f"{obj} unit,sink-excl")
    a1.set_xlabel("component"); a1.set_ylabel("raw singular value / sigma_1 (log)"); a1.grid(alpha=0.3, which="both"); a1.legend()
    a1.set_title("RAW spectrum (magnitude-weighted, sink-skewed)")
    for p in (0.90, 0.95):
        a2.axhline(p, ls=":", color="gray", lw=1)
    a2.set_xlabel("components k"); a2.set_ylabel("cumulative explained variance"); a2.set_ylim(0, 1.02)
    a2.grid(alpha=0.3); a2.legend(fontsize=7); a2.set_title("Cumulative EV: dotted (unit,sink-excl) is the trustworthy curve")
    fig.suptitle("Is object-ness low-dimensional? SVD of d_i = h_clean - h_inpaint (read layer, object patches)")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/subspace_spectrum.png", dpi=130); plt.close(fig)

    with open(f"{args.outdir}/stats.txt", "w") as fp:
        for obj, r in results.items():
            fp.write(f"{obj}: n={r['n']} n_sink={r['n_sink']} sink_in_region={r['sink_in_region']} "
                     f"raw[k90={r['k'][0.90]},k95={r['k'][0.95]},PR={r['pr']:.1f}] "
                     f"unit[k90={r['kn'][0.90]},k95={r['kn'][0.95]},PR={r['prn']:.1f}] "
                     f"unit_sinkexcl[k90={r['ke'][0.90]},k95={r['ke'][0.95]},PR={r['pre']:.1f}]\n")
    print(f"\nsaved {args.outdir}/subspace_spectrum.png + stats.txt")
    print("READ: compare RAW vs UNIT vs UNIT-sink-excluded k. RAW << the others => the raw spectrum was hijacked by "
          "magnitude/sinks (prior raw-d SVD was misleading). Trust UNIT-sink-excluded: small k@95% (<=5-10) + high "
          "cross-object overlap => object-ness low-rank AND shared, build min||U_k^T (h_adv - h_inpaint)||^2, k=k@95%, "
          "no collection. If sink_in_region is 0, the region-restricted SVD was already sink-free (unit ~ sink-excluded).")


if __name__ == "__main__":
    main()
