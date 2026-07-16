"""Stage 17: region-level M-Attack with a PADDING sweep (dose-response).

M-Attack's crop-matching engine, but the perturbation and the random crops are
confined to the object ROI expanded by a padding margin. Sweeping the padding
from tight (0) to whole-image locates the transition where concealment starts to
work:
  * flips at small padding  -> salvageable localized camouflage ("minimal
    padding for concealment" is the finding);
  * needs large padding      -> concealment fundamentally requires attacking
    context/silhouette broadly; localized camouflage does not work (clean
    negative).

Per padding level: expand the mask bbox by frac*RES px, confine delta to that
box, sample RandomResizedCrops WITHIN the box for both adv and target
(removed.png), MI-FGSM to maximize crop-embedding similarity. Single surrogate =
LLaVA-1.5's encoder (ViT-L/14-336).

Example:
    uv run python m_attack_roi.py --pads 0,0.05,0.15,0.35,1.0 --steps 300 --eps 16
    uv run python vlm_eval.py --images-dir results/m_attack_roi \
        --object dog --models llava-hf/llava-1.5-7b-hf
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--source", default="original.png")
    ap.add_argument("--target", default="removed.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0, help="L_inf in /255")
    ap.add_argument("--alpha", type=float, default=1.0, help="step in /255")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--pads", default="0,0.05,0.15,0.35,1.0",
                    help="padding fractions of RES added around the bbox (1.0=whole image)")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--outdir", default="results/m_attack_roi")
    args = ap.parse_args()
    RES = args.input_res
    eps, alpha = args.eps / 255.0, args.alpha / 255.0

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    print(f"surrogate={args.model} res={RES}  {args.source}->{args.target}  eps={args.eps}/255")

    def load(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def embed(x01):
        v = model.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True).pooler_output
        return F.normalize(model.visual_projection(v), dim=-1)

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [args.object, "cat", "couch", "cushion", "wall", "window"]
    label_txt = texts([f"a photo of a {l}" for l in labels])

    def zero_shot(x01):
        return ((embed(x01) @ label_txt.T) * 100).softmax(-1).squeeze(0)

    x0, x_tgt = load(args.source), load(args.target)
    mask = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    t0, b0, l0, r0 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = T.RandomResizedCrop(RES, scale=tuple(args.crop_scale), antialias=True)

    os.makedirs(args.outdir, exist_ok=True)

    def save(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    save(x0, f"{args.outdir}/baseline.png")
    p0 = zero_shot(x0).cpu().numpy()
    print(f"baseline global P({args.object})={p0[0]:.3f} argmax={labels[int(p0.argmax())]}")

    pads = [float(p) for p in args.pads.split(",")]
    rows = []
    summary = [f"{args.source}->{args.target} eps={args.eps}/255 steps={args.steps}",
               f"baseline global P({args.object})={p0[0]:.4f}"]
    for frac in pads:
        pad = int(frac * RES)
        T_, B_ = max(0, t0 - pad), min(RES, b0 + pad + 1)
        L_, R_ = max(0, l0 - pad), min(RES, r0 + pad + 1)
        region = torch.zeros((1, 1, RES, RES), device=DEVICE)
        region[:, :, T_:B_, L_:R_] = 1.0
        coverage = region.mean().item()

        # crop of the padded region (for RandomResizedCrop within it)
        def region_crop(x):
            return x[:, :, T_:B_, L_:R_]

        tgt_region = region_crop(x_tgt)
        delta = torch.zeros_like(x0, requires_grad=True)
        momentum = torch.zeros_like(x0)
        for step in range(args.steps):
            with torch.no_grad():
                tgt_feat = embed(crop(tgt_region))
            adv = torch.clamp(x0 + delta * region, 0, 1)
            src_feat = embed(crop(region_crop(adv)))
            sim = (src_feat * tgt_feat).sum()
            g, = torch.autograd.grad(sim, delta)
            with torch.no_grad():
                momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                delta.add_(alpha * momentum.sign()).clamp_(-eps, eps)
                delta.data = torch.clamp(x0 + delta * region, 0, 1) - x0
            delta.requires_grad_(True)

        adv = torch.clamp(x0 + delta.detach() * region, 0, 1)
        with torch.no_grad():
            reg_sim = (embed(region_crop(adv)) @ embed(tgt_region).T).item()
            pv = zero_shot(adv).cpu().numpy()
        tag = f"pad{frac:g}"
        save(adv, f"{args.outdir}/{tag}.png")
        rows.append(dict(frac=frac, coverage=coverage, reg_sim=reg_sim,
                         p_obj=float(pv[0]), p_couch=float(pv[2]), argmax=labels[int(pv.argmax())]))
        line = (f"pad={frac:<5g} region={coverage*100:4.1f}%  cos(region,target)={reg_sim:.3f}  "
                f"global P({args.object})={pv[0]:.3f} P(couch)={pv[2]:.3f}  argmax={labels[int(pv.argmax())]}")
        print(line); summary.append(line)

    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cov = [r["coverage"] * 100 for r in rows]
    ax[0].plot(cov, [r["p_obj"] for r in rows], "o-", color="tab:red", label=f"P({args.object})")
    ax[0].plot(cov, [r["p_couch"] for r in rows], "o-", color="tab:blue", label="P(couch)")
    ax[0].axhline(p0[0], ls=":", c="gray", label="baseline P(obj)")
    ax[0].set_xlabel("region coverage (% of image)"); ax[0].set_ylabel("global zero-shot P")
    ax[0].set_title("dose-response: CLIP-space vs padding"); ax[0].legend()
    ax[1].plot(cov, [r["reg_sim"] for r in rows], "o-", color="tab:green")
    ax[1].set_xlabel("region coverage (% of image)"); ax[1].set_ylabel("cos(region, target)")
    ax[1].set_title("region steering achieved vs padding")
    fig.suptitle("region-level M-Attack padding sweep (single surrogate = LLaVA-1.5 encoder)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/dose_response.png", dpi=130)
    print(f"\nsaved {args.outdir}/ ({len(pads)} padding levels + baseline) -> "
          f"caption all via vlm_eval on llava-1.5-7b for the LLaVA dose-response")


if __name__ == "__main__":
    main()
