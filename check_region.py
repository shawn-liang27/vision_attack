"""Diagnostic: is the substitution region actually ON the object?

The oracle substitutes tokens chosen by the MASK (coverage>0.3, then ring-dilated). But the ground
truth of "which tokens carry the object" is where the features actually CHANGE when the object is
removed: ||d_i|| = ||h_clean[i] - h_inpaint[i]|| at the read layer. If most d-energy falls OUTSIDE
the mask region, or the mask centroid and the d-energy centroid are far apart, the mask is misplaced/
undersized and the mask-centered 1.5x dilation misses the real object tokens -- which would explain
why 1.5x removes one object (good mask) but not another (bad mask), with no per-object code bug.

Per object prints: n0 (tight tokens), fraction of total d-energy INSIDE the mask region and inside
1.5x, IoU(mask region, top-n0 ||d|| tokens), and the token-grid distance between mask centroid and
d-energy centroid. Saves region_check_<obj>.png: [image+mask region] | [||d|| heatmap] | [image+1.5x].

CLIP only, no LLaVA, no optimization.

    uv run python check_region.py --dataset dataset.jsonl --objects dog,cat,car
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformers import CLIPModel, CLIPProcessor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dilate(grid, k=1):
    g = grid.copy()
    for _ in range(k):
        p = g.copy()
        p[1:, :] |= g[:-1, :]; p[:-1, :] |= g[1:, :]; p[:, 1:] |= g[:, :-1]; p[:, :-1] |= g[:, 1:]
        g = p
    return g


def ring_order(base):
    ring = np.full(base.shape, 1 << 20, np.int64); ring[base] = 0; cur = base.copy(); r = 0
    while not cur.all():
        r += 1; nxt = dilate(cur, 1); newly = nxt & (~cur)
        if not newly.any():
            break
        ring[newly] = r; cur = nxt
    return np.argsort(ring.reshape(-1), kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat,car")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--dilate", type=float, default=1.5)
    ap.add_argument("--outdir", default="results/region_check")
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()
    clip.requires_grad_(False)
    ipc = CLIPProcessor.from_pretrained(args.surrogate).image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH

    def vis(x01):
        return clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                 output_hidden_states=True).hidden_states[-2][:, 1:].squeeze(0)  # (P,D) raw

    def load01(p):
        im = Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    gy, gx = gy.reshape(-1), gx.reshape(-1)

    print(f"{'obj':<5}{'n0':>4}{'e_in_mask':>11}{'e_in_1.5x':>11}{'IoU(mask,topd)':>16}{'centroid_dist':>15}  verdict")
    for s in samples:
        obj = s["object"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{obj}] missing mask/inpaint; skip"); continue
        x0, xR = load01(s["image"]), load01(s["target"])
        with torch.no_grad():
            d = (vis(x0) - vis(xR)).detach()               # (P,D)
        dn2 = (d ** 2).sum(-1).cpu().numpy()               # (P,) per-token feature-change energy
        total = dn2.sum() + 1e-12

        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        cov = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))          # (GRID,GRID) coverage
        base = cov > args.thresh
        n0 = int(base.sum())
        if n0 == 0:
            print(f"[{obj}] mask selects 0 tokens at thresh {args.thresh}; likely bad mask"); continue
        order = ring_order(base)
        T = min(GRID * GRID, max(n0, round(args.dilate * n0)))
        r15 = np.zeros(GRID * GRID, bool); r15[order[:T]] = True
        m1d = base.reshape(-1)

        e_mask = dn2[m1d].sum() / total
        e_15 = dn2[r15].sum() / total
        topd = np.zeros(GRID * GRID, bool); topd[np.argsort(-dn2)[:n0]] = True
        inter = (m1d & topd).sum(); union = (m1d | topd).sum()
        iou = inter / max(union, 1)
        # centroids in token-grid coords
        mc = np.array([gy[m1d].mean(), gx[m1d].mean()])
        w = dn2 / total
        dc = np.array([(gy * w).sum(), (gx * w).sum()])
        cdist = float(np.hypot(*(mc - dc)))
        verdict = ("MASK OK" if (e_mask > 0.5 and cdist < 3) else
                   "MASK SUSPECT (d-energy off the mask)" if cdist >= 3 or e_mask < 0.35 else "borderline")
        print(f"{obj:<5}{n0:>4}{e_mask:>11.2f}{e_15:>11.2f}{iou:>16.2f}{cdist:>15.1f}  {verdict}")

        # visualization
        img = np.asarray(Image.open(s["image"]).convert("RGB").resize((RES, RES), Image.BICUBIC))
        up = lambda g: np.kron(g.reshape(GRID, GRID), np.ones((PATCH, PATCH)))   # 24x24 -> 336x336
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
        ax[0].imshow(img); ax[0].imshow(up(m1d), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        ax[0].set_title(f"{obj}: image + MASK region ({n0} tok)"); ax[0].axis("off")
        hm = ax[1].imshow(dn2.reshape(GRID, GRID), cmap="viridis"); ax[1].set_title("||d_i|| (feature change) -- true object")
        ax[1].axis("off"); fig.colorbar(hm, ax=ax[1], fraction=0.046)
        ax[2].imshow(img); ax[2].imshow(up(r15), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        ax[2].set_title(f"image + 1.5x region ({T} tok)"); ax[2].axis("off")
        fig.suptitle(f"{obj}: e_in_mask={e_mask:.2f} e_in_1.5x={e_15:.2f} IoU={iou:.2f} centroid_dist={cdist:.1f} -> {verdict}")
        fig.tight_layout(); fig.savefig(f"{args.outdir}/region_check_{obj}.png", dpi=120); plt.close(fig)

    print(f"\nsaved {args.outdir}/region_check_<obj>.png")
    print("READ: e_in_mask high (>0.5) and centroid_dist small (<~3 tokens) => mask is on the object; a 1.5x failure "
          "is a size/context effect (try 2-2.5x). e_in_mask low / centroid_dist large => the MASK is misplaced -- the "
          "1.5x region misses the real object tokens; fix by defining the region from ||d_i|| (feature change), not the mask.")


if __name__ == "__main__":
    main()
