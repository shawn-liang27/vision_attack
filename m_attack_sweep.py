"""Stage 19: seeded padding sweep with continuous signals + perturbation density.

Rigor upgrade over m_attack_roi/h1: run each padding level N times with different
random-crop seeds, and log CONTINUOUS signals (not just the binary LLaVA verdict)
so we can tell threshold-noise from a real effect, plus per-pixel perturbation
density to test the budget-dilution hypothesis.

Modes:
  h1    : whole-image crop-matching LOSS, gradient masked to ROI (global signal,
          local perturbation).
  local : crops confined to the ROI region (local signal, local perturbation).

Per (pad, seed) it records, from the FULL-image forward pass:
  cos_to_target   -- global CLS embedding cosine to the target
  global_p_dog    -- whole-image zero-shot P(dog)
  roi_p_dog       -- dense (MaskCLIP) P(dog) averaged over the ROI patches
  delta_mean_frac -- mean |delta|/eps over region pixels   (density)
  delta_linf_frac -- max  |delta|/eps over region pixels
  frac_at_cap     -- fraction of region pixels at the L_inf cap
and saves the image as pad{frac}_seed{s}.png for vlm_eval.

Then: vlm_eval over the dir, then summarize_sweep.py to get success-rate/level.

Example:
    uv run python m_attack_sweep.py --mode h1 --pads 0,0.05,0.15,0.35,1.0 \
        --seeds 5 --steps 300 --eps 16
    uv run python vlm_eval.py --images-dir results/sweep_h1 --object dog \
        --models llava-hf/llava-1.5-7b-hf
    uv run python summarize_sweep.py --dir results/sweep_h1
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--source", default="original.png")
    ap.add_argument("--target", default="removed.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--mode", default="h1", choices=["h1", "local"])
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--pads", default="0,0.05,0.15,0.35,1.0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--object", default="dog")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    RES = args.input_res
    eps, alpha = args.eps / 255.0, args.alpha / 255.0
    outdir = args.outdir or f"results/sweep_{args.mode}"

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = model.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"mode={args.mode} surrogate={args.model} res={RES} grid={GRID}  "
          f"{args.source}->{args.target} eps={args.eps} seeds={args.seeds}")

    def load(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def embed_global(x01):
        v = model.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True).pooler_output
        return F.normalize(model.visual_projection(v), dim=-1)

    def embed_dense(x01):
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        layer = vm.encoder.layers[-1]
        h = out.hidden_states[-2]
        a = layer.layer_norm1(h)
        h = h + layer.self_attn.out_proj(layer.self_attn.v_proj(a))
        h = h + layer.mlp(layer.layer_norm2(h))
        d = model.visual_projection(vm.post_layernorm(h))
        return F.normalize(d[:, 1:], dim=-1).squeeze(0)

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [args.object, "cat", "couch", "cushion", "wall", "window"]
    label_txt = texts([f"a photo of a {l}" for l in labels])

    x0, x_tgt = load(args.source), load(args.target)
    tgt_global = embed_global(x_tgt).detach()
    mask = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    t0, b0, l0, r0 = ys.min(), ys.max(), xs.min(), xs.max()
    mg = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
    obj_patch = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5).reshape(-1)).to(DEVICE)
    crop = T.RandomResizedCrop(RES, scale=tuple(args.crop_scale), antialias=True)

    os.makedirs(outdir, exist_ok=True)

    def save(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    save(x0, f"{outdir}/baseline.png")

    def roi_pdog(x01):
        d = embed_dense(x01)
        return ((d[obj_patch] @ label_txt.T) * 100).softmax(-1)[:, 0].mean().item()

    def global_pdog(x01):
        return float(((embed_global(x01) @ label_txt.T) * 100).softmax(-1).squeeze(0)[0])

    pads = [float(p) for p in args.pads.split(",")]
    rows = ["mode,pad,coverage,seed,filename,cos_to_target,global_p_dog,roi_p_dog,"
            "delta_mean_frac,delta_linf_frac,frac_at_cap"]
    for frac in pads:
        pad = int(frac * RES)
        T_, B_ = max(0, t0 - pad), min(RES, b0 + pad + 1)
        L_, R_ = max(0, l0 - pad), min(RES, r0 + pad + 1)
        region = torch.zeros((1, 1, RES, RES), device=DEVICE)
        region[:, :, T_:B_, L_:R_] = 1.0
        coverage = region.mean().item()
        rmask = region > 0

        for seed in range(args.seeds):
            seed_everything(1000 + seed)
            delta = torch.zeros_like(x0, requires_grad=True)
            momentum = torch.zeros_like(x0)
            for _ in range(args.steps):
                with torch.no_grad():
                    tgt_feat = embed_global(crop(x_tgt if args.mode == "h1"
                                                 else x_tgt[:, :, T_:B_, L_:R_]))
                adv = torch.clamp(x0 + delta * region, 0, 1)
                src_in = adv if args.mode == "h1" else adv[:, :, T_:B_, L_:R_]
                sim = (embed_global(crop(src_in)) * tgt_feat).sum()
                g, = torch.autograd.grad(sim, delta)
                with torch.no_grad():
                    g = g * region                      # mask the update
                    momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                    delta.add_(alpha * (momentum * region).sign() * region).clamp_(-eps, eps)
                    delta.data = torch.clamp(x0 + delta * region, 0, 1) - x0
                delta.requires_grad_(True)

            adv = torch.clamp(x0 + delta.detach() * region, 0, 1)
            fn = f"pad{frac:g}_seed{seed}.png"
            save(adv, f"{outdir}/{fn}")
            with torch.no_grad():
                cos_t = (embed_global(adv) @ tgt_global.T).item()
                gpd = global_pdog(adv)
                rpd = roi_pdog(adv)
                dr = delta.detach().abs()[rmask]
                dmean = (dr.mean() / eps).item()
                dlinf = (dr.max() / eps).item()
                cap = (dr > 0.99 * eps).float().mean().item()
            rows.append(f"{args.mode},{frac:g},{coverage:.4f},{seed},{fn},{cos_t:.4f},"
                        f"{gpd:.4f},{rpd:.4f},{dmean:.4f},{dlinf:.4f},{cap:.4f}")
            print(f"pad={frac:<5g} seed={seed} cov={coverage*100:4.1f}%  cos->tgt={cos_t:.3f} "
                  f"gP(dog)={gpd:.3f} roiP(dog)={rpd:.3f}  |d|/eps mean={dmean:.2f} cap%={cap*100:.0f}")

    with open(f"{outdir}/metrics.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nsaved {outdir}/metrics.csv + {len(pads)*args.seeds} images. "
          f"Next: vlm_eval on {outdir}, then summarize_sweep.py --dir {outdir}")


if __name__ == "__main__":
    main()
