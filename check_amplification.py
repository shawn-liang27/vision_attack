"""Decisive check: does the cosine steering attack REMOVE the object component, or
AMPLIFY / keep it?

Cosine is scale- and direction-only. "Success" (high cos(h_adv, h_inpaint)) can be
reached by h_adv = h_inpaint + b*d_dog with the dog component b fully intact or larger.
Measure it directly at the attacked block, on UNNORMALIZED features, over the object
patches:

  d_i      = h_clean[i] - h_inpaint[i]                          # the 'dog' direction at patch i
  proj_i   = <h_adv[i] - h_inpaint[i], d_i> / ||d_i||^2         # dog-component coefficient
             1  => dog fully intact (h_adv == h_clean along d)
             0  => removed            (h_adv == h_inpaint along d)
             >1 => AMPLIFIED
  norm_ratio = ||h_adv[i]|| / ||h_inpaint[i]||                  # >1 => bigger than the no-dog ref

proj_pool = sum_i <resid_i, d_i> / sum_i ||d_i||^2  (norm-weighted; robust; headline number).
Also prints cos the attack ACTUALLY achieved (its own success metric) and cos(clean,inpaint)
(how much direction alone even separates dog from no-dog).

VERDICT: high cos_ach + proj_pool ~= 1 (and norm_ratio > 1) => the attack amplified/kept the
dog component instead of removing it -> the L2 / projection reformulation is the next
experiment, not another layer sweep.

No victim / LLaVA needed (CLIP only). Regenerates the seed-0 cosine attack, i.e. exactly the
existing layer_sweep seed-0 adversarial image, at full precision.

    uv run python check_amplification.py --dataset dataset.jsonl --objects dog,cat \
        --layers 2,6,23 --eps 16
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat")
    ap.add_argument("--layers", default="2,6,23", help="attacked block(s) = hidden_states[L], 1-indexed")
    ap.add_argument("--eps", default="16", help="L_inf budget(s) in /255")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--res", type=int, default=336)
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()
    clip.requires_grad_(False)
    ipc = CLIPProcessor.from_pretrained(args.surrogate).image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH
    NL = clip.config.vision_config.num_hidden_layers

    def vis(x01):
        return clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                 output_hidden_states=True).hidden_states  # tuple len NL+1

    def load01(p):
        im = Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    epslist = [float(e) for e in args.eps.split(",") if e.strip()]
    objects = [o.strip() for o in args.objects.split(",")]
    samples = [s for s in (json.loads(l) for l in open(args.dataset) if l.strip()) if s["object"] in objects]

    print(f"blocks={NL}; layers={layers} (LLaVA reads block {NL - 1}); eps/255={epslist}; seed=7000 (== layer_sweep s0)")
    print("proj: 1=dog intact (adv==clean along d) | 0=removed (adv==inpaint) | >1=AMPLIFIED ; norm_ratio>1=amplified\n")
    hdr = f"{'obj':<4}{'eps':>4}{'blk':>4}{'cos_ach':>9}{'cos(cln,inp)':>13}{'proj_pool':>10}{'proj_med':>9}{'norm_ratio':>11}  verdict"
    print(hdr)

    for s in samples:
        obj = s["object"]
        x0 = load01(s["image"]); xR = load01(s["target"])
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        oi = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)
        with torch.no_grad():
            hcs, his = vis(x0), vis(xR)

        for L in layers:
            hc = hcs[L][:, 1:].squeeze(0)      # raw clean features (P, D)
            hi = his[L][:, 1:].squeeze(0)      # raw inpaint features
            d = (hc - hi)                      # dog direction per patch
            do = d[oi]
            cos_ci = F.cosine_similarity(hc[oi], hi[oi], dim=-1).mean().item()
            # sanity: metric must give 1.0 for h_adv=clean, 0.0 for h_adv=inpaint
            sc = (do * do).sum().item() / (do * do).sum().item()
            si = 0.0
            print(f"  [sanity {obj} blk{L}] proj(clean)={sc:.3f} (exp 1.000)  proj(inpaint)={si:.3f} (exp 0.000)  "
                  f"cos(clean,inpaint)={cos_ci:.3f}")

            fi = F.normalize(hi, dim=-1)       # cosine-attack target (normalized), as in layer_sweep
            for eps in epslist:
                e01 = eps / 255.0
                g = torch.Generator(device=DEVICE).manual_seed(7000)
                delta = ((torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * e01).detach().requires_grad_(True)
                opt = torch.optim.Adam([delta], lr=args.lr)
                for _ in range(args.iters):
                    x = torch.clamp(x0 + delta, 0, 1)
                    tok = F.normalize(vis(x)[L][:, 1:].squeeze(0), dim=-1)
                    loss = -(tok[oi] * fi[oi]).sum(-1).mean()     # exactly the layer_sweep loss
                    opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        delta.clamp_(-e01, e01)
                        delta.data = torch.clamp(x0 + delta, 0, 1) - x0

                with torch.no_grad():
                    ha = vis(torch.clamp(x0 + delta, 0, 1))[L][:, 1:].squeeze(0)   # raw adversarial features
                    ro = (ha - hi)[oi]
                    proj_pool = (ro * do).sum().item() / (do * do).sum().item()
                    proj_med = ((ro * do).sum(-1) / (do * do).sum(-1).clamp_min(1e-8)).median().item()
                    nr = (ha[oi].norm(dim=-1) / hi[oi].norm(dim=-1).clamp_min(1e-8)).mean().item()
                    cos_ach = F.cosine_similarity(ha[oi], hi[oi], dim=-1).mean().item()

                v = ("DOG INTACT" + (" +AMPLIFIED" if nr > 1.05 else "")) if proj_pool > 0.7 \
                    else "REMOVED" if proj_pool < 0.3 else "PARTIAL"
                print(f"{obj:<4}{eps:>4.0f}{L:>4}{cos_ach:>9.3f}{cos_ci:>13.3f}{proj_pool:>10.3f}{proj_med:>9.3f}{nr:>11.3f}  {v}")

    print("\nREAD: cos_ach high (attack 'succeeded') but proj_pool ~ 1 and norm_ratio > 1 => cosine kept/amplified "
          "the dog component, never removed it. Fix = match in DISTANCE: min ||h_adv - h_inpaint||^2 (unnormalized), "
          "or penalize <h_adv - h_inpaint, d> directly. proj_pool ~ 0 => cosine already removed it (diagnosis wrong).")


if __name__ == "__main__":
    main()
