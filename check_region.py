"""Diagnostic: is the substitution region on the object -- ROBUST to massive-activation tokens.

A handful of ViT/CLIP "attention sink" / massive-activation tokens (low-info patches with ~10x
normal norm) dominate raw ||d_i|| = ||h_clean - h_inpaint||, because they aggregate global context
and swing hugely when the image changes -- even though they sit OFF the object. Raw ||d||-energy
fractions therefore falsely flag good masks. This version identifies the sink tokens by the CLEAN
feature norm ||h_clean_i|| (intrinsic, not inpaint-driven), excludes them, and judges the mask on
the MEDIAN change inside vs outside (outlier-robust).

Per object prints: n_sinks (||h_clean|| outliers) and how many fall in the mask; raw vs sink-excluded
e_in_mask; and obj/bg MEDIAN ||d|| ratio (>~2 => mask is on real object-driven change). Saves
region_check_<obj>.png: [image+mask] | [||d|| clipped to 97th pct, sinks marked X] | [||h_clean|| norm].

CLIP only, no LLaVA.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat,car")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--sink-mult", type=float, default=4.0, help="||h_clean|| > sink_mult * median => sink token")
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
    gy, gx = np.mgrid[0:GRID, 0:GRID]; gy, gx = gy.reshape(-1), gx.reshape(-1)

    print(f"{'obj':<5}{'n0':>4}{'n_sink':>7}{'sink_in_mask':>13}{'e_mask_raw':>12}{'e_mask_robust':>15}"
          f"{'obj/bg_med_d':>14}  verdict")
    for s in samples:
        obj = s["object"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{obj}] missing mask/inpaint; skip"); continue
        x0, xR = load01(s["image"]), load01(s["target"])
        with torch.no_grad():
            hc = vis(x0); hi = vis(xR)
        dn2 = ((hc - hi) ** 2).sum(-1).cpu().numpy()          # per-token change energy
        cn = hc.norm(dim=-1).cpu().numpy()                    # per-token CLEAN norm (intrinsic; sinks are huge)

        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        base = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > args.thresh
        n0 = int(base.sum()); m1d = base.reshape(-1)
        if n0 == 0:
            print(f"[{obj}] mask selects 0 tokens; bad mask"); continue

        sink = cn > args.sink_mult * np.median(cn)            # massive-activation tokens (by clean norm)
        n_sink = int(sink.sum()); sink_in_mask = int((sink & m1d).sum())
        nonsink = ~sink
        e_raw = dn2[m1d].sum() / (dn2.sum() + 1e-9)
        e_rob = dn2[m1d & nonsink].sum() / (dn2[nonsink].sum() + 1e-9)   # exclude sinks from the total
        bg = nonsink & (~m1d)
        obj_med = np.median(dn2[m1d & nonsink]) if (m1d & nonsink).any() else 0.0
        bg_med = np.median(dn2[bg]) if bg.any() else 1e-9
        ratio = obj_med / max(bg_med, 1e-9)
        verdict = "MASK OK (object drives the change)" if ratio > 2 else \
                  "MASK SUSPECT (object region no more changed than background)"
        print(f"{obj:<5}{n0:>4}{n_sink:>7}{sink_in_mask:>13}{e_raw:>12.2f}{e_rob:>15.2f}{ratio:>14.1f}  {verdict}")

        # visualization
        img = np.asarray(Image.open(s["image"]).convert("RGB").resize((RES, RES), Image.BICUBIC))
        up = lambda g: np.kron(g.reshape(GRID, GRID), np.ones((PATCH, PATCH)))
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        ax[0].imshow(img); ax[0].imshow(up(m1d), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        ax[0].set_title(f"{obj}: image + MASK ({n0} tok)"); ax[0].axis("off")
        vmax = np.percentile(dn2, 97)
        hm = ax[1].imshow(dn2.reshape(GRID, GRID), cmap="viridis", vmax=vmax)
        if n_sink:
            ax[1].scatter(gx[sink], gy[sink], marker="x", c="red", s=60, label=f"{n_sink} sink tokens")
            ax[1].legend(loc="lower right", fontsize=8)
        ax[1].set_title(f"||d_i|| (clipped @97pct); sinks x'd"); ax[1].axis("off"); fig.colorbar(hm, ax=ax[1], fraction=0.046)
        hn = ax[2].imshow(cn.reshape(GRID, GRID), cmap="magma")
        ax[2].set_title("||h_clean|| norm -- sinks = intrinsic outliers"); ax[2].axis("off")
        fig.colorbar(hn, ax=ax[2], fraction=0.046)
        fig.suptitle(f"{obj}: n_sink={n_sink} (in_mask={sink_in_mask}) e_mask raw={e_raw:.2f}/robust={e_rob:.2f} "
                     f"obj:bg median-d={ratio:.1f} -> {verdict}")
        fig.tight_layout(); fig.savefig(f"{args.outdir}/region_check_{obj}.png", dpi=120); plt.close(fig)

    print(f"\nsaved {args.outdir}/region_check_<obj>.png")
    print("READ: e_mask_raw low but e_mask_robust high AND obj/bg median-d > ~2 => the mask IS on the object; the low "
          "raw fraction was massive-activation SINK tokens (see ||h_clean|| panel) hijacking the energy, not a bad mask. "
          "Implication: exclude sinks from any d_i-based subspace/region; and test whether substituting the object region "
          "PLUS the sink tokens removes the object where the region alone did not (object info aggregated into sinks).")


if __name__ == "__main__":
    main()
