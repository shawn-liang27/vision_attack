"""Stage 16: M-Attack (V1), simplified to a single surrogate = LLaVA-1.5's own
encoder, steering original.png toward removed.png. Verification of the mechanism.

Faithful to the M-Attack V1 procedure (the reference the user shared): at each
step take a RANDOM RESIZED CROP of both the adversarial (source) image and the
target image, embed each with CLIP, and MAXIMIZE cosine similarity between the
two crop embeddings via MI-FGSM under an L_inf budget. This "local-to-local"
crop matching is the whole engine. Whole-image perturbation (NOT mask-confined).

Simplifications (per request, since we know the victim's encoder):
  * single surrogate = openai/clip-vit-large-patch14-336 (LLaVA-1.5's tower),
    input 336 -- white-box best case, no ensemble;
  * no MCA / ATA / patch-momentum (those are V2).

The question: can M-Attack steer original -> removed.png strongly enough that
LLaVA-1.5 describes an empty couch / stops reporting a dog?

Example:
    uv run python m_attack.py --steps 300 --eps 16 --alpha 1 --attack mifgsm
    uv run python vlm_eval.py --images-dir results/m_attack \
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
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0, help="L_inf budget in /255")
    ap.add_argument("--alpha", type=float, default=1.0, help="step size in /255")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--attack", default="mifgsm", choices=["mifgsm", "pgd"])
    ap.add_argument("--object", default="dog")
    ap.add_argument("--outdir", default="results/m_attack")
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
    print(f"single surrogate={args.model} res={RES}  {args.source} -> {args.target}  "
          f"eps={args.eps}/255 alpha={args.alpha}/255 steps={args.steps} attack={args.attack}")

    def load(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def embed(x01):
        """CLIP global image embedding of a [0,1] tensor at RES; differentiable."""
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

    x0 = load(args.source)
    x_tgt = load(args.target)
    crop = T.RandomResizedCrop(RES, scale=tuple(args.crop_scale), antialias=True)

    tgt_global = embed(x_tgt).detach()
    p0 = zero_shot(x0).cpu().numpy()
    sim0 = (embed(x0).detach() @ tgt_global.T).item()
    print(f"baseline: cos(orig, target)={sim0:.3f}  P({args.object})={p0[0]:.3f} "
          f"argmax={labels[int(p0.argmax())]}")

    delta = torch.zeros_like(x0, requires_grad=True)
    momentum = torch.zeros_like(x0)
    optimizer = torch.optim.Adam([delta], lr=alpha) if args.attack == "pgd" else None

    traj = []
    for step in range(args.steps):
        with torch.no_grad():
            tgt_feat = embed(crop(x_tgt))          # random target crop, re-sampled each step
        adv = torch.clamp(x0 + delta, 0, 1)
        src_feat = embed(crop(adv))                # random source crop
        sim = (src_feat * tgt_feat).sum()          # local-to-local cosine

        if args.attack == "pgd":
            optimizer.zero_grad()
            (-sim).backward()
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.data = torch.clamp(x0 + delta, 0, 1) - x0
        else:  # MI-FGSM: ascend similarity
            g, = torch.autograd.grad(sim, delta)
            with torch.no_grad():
                momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                delta.add_(alpha * momentum.sign()).clamp_(-eps, eps)
                delta.data = torch.clamp(x0 + delta, 0, 1) - x0
            delta.requires_grad_(True)

        if step % 15 == 0 or step == args.steps - 1:
            with torch.no_grad():
                adv = torch.clamp(x0 + delta, 0, 1)
                gsim = (embed(adv) @ tgt_global.T).item()
                pv = zero_shot(adv).cpu().numpy()
            traj.append((step, gsim, float(pv[0]), float(pv[2])))  # step, sim->tgt, P(dog), P(couch)

    adv = torch.clamp(x0 + delta.detach(), 0, 1)
    gsim = (embed(adv) @ tgt_global.T).item()
    pv = zero_shot(adv).cpu().numpy()

    os.makedirs(args.outdir, exist_ok=True)

    def save(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    save(x0, f"{args.outdir}/baseline.png")
    save(adv, f"{args.outdir}/m_attack_eps{int(args.eps)}.png")

    summary = [
        f"single surrogate={args.model}  {args.source}->{args.target}  "
        f"eps={args.eps}/255 steps={args.steps} attack={args.attack}",
        f"cos(adv, target): {sim0:.4f} -> {gsim:.4f}   (1.0 = identical embedding)",
        f"global zero-shot P({args.object}): {p0[0]:.4f} -> {pv[0]:.4f}   "
        f"argmax {labels[int(p0.argmax())]} -> {labels[int(pv.argmax())]}",
        f"global zero-shot P(couch): {p0[2]:.4f} -> {pv[2]:.4f}",
    ]
    print("\n" + "\n".join(summary))
    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    st = [t[0] for t in traj]
    ax[0].plot(st, [t[1] for t in traj], "o-"); ax[0].axhline(sim0, ls=":", c="gray")
    ax[0].set_title("cos(adv, target) — steering progress"); ax[0].set_xlabel("step")
    ax[1].plot(st, [t[2] for t in traj], "o-", color="tab:red", label=f"P({args.object})")
    ax[1].plot(st, [t[3] for t in traj], "o-", color="tab:blue", label="P(couch)")
    ax[1].axhline(p0[0], ls=":", c="tab:red", alpha=.5); ax[1].legend()
    ax[1].set_title("global zero-shot vs step"); ax[1].set_xlabel("step")
    ax[2].imshow(adv.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax[2].set_title(f"adversarial (eps={int(args.eps)}/255)"); ax[2].axis("off")
    fig.suptitle(f"M-Attack (single surrogate = LLaVA-1.5 encoder) steering "
                 f"{args.source}->{args.target}", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/m_attack.png", dpi=130)
    print(f"\nsaved {args.outdir}/ -> caption via vlm_eval on llava-1.5-7b")


if __name__ == "__main__":
    main()
