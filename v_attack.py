"""Stage 15: V-Attack mechanism, TEXT target "cat" (capability probe).

Recreates V-Attack's core procedure -- steer the disentangled VALUE features
(v_proj of the last block, projected to CLIP joint space) with PGD under an
L_inf budget -- but:
  * NO surrogate ensemble (single white-box model = LLaVA-1.5's own encoder,
    openai/clip-vit-large-patch14-336), and
  * a TEXT target "cat" instead of a background image:
        loss = mean_ROI cos(V_joint, phi_T("dog")) - cos(V_joint, phi_T("cat"))
    i.e. push the dog-region value features away from "dog" and toward "cat".

Purpose: verify whether ANY feature-space attack moves LLaVA. dog->cat is a much
easier target than dog->background, so if this flips LLaVA's caption to "cat"
the mechanism works (and background was simply unreachable); if even this fails
white-box, no feature-space attack moves LLaVA -- a definitive negative.

Saves exact 336 squares; verify with:
    uv run python vlm_eval.py --images-dir results/v_attack_cat/square \
        --object cat --models llava-hf/llava-1.5-7b-hf

Example:
    uv run python v_attack_cat.py --budgets 8,16,32,64,128 --iters 300
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import CLIPModel, CLIPProcessor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--source", default="dog")
    ap.add_argument("--target", default="cat")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--budgets", default="8,16,32,64,128")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--outdir", default="results/v_attack_cat")
    args = ap.parse_args()
    src, tgt = args.source, args.target
    RES = args.res

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = model.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"model={args.model} res={RES} grid={GRID}  steer {src}->{tgt}")

    def to01(img):
        arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def value_joint(x01):
        """V-Attack feature: last-block VALUE features projected into CLIP joint
        space (MaskCLIP value path), L2-normalized. Differentiable."""
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        layer = vm.encoder.layers[-1]
        h_in = out.hidden_states[-2]
        v = layer.self_attn.v_proj(layer.layer_norm1(h_in))     # VALUE features
        d = model.visual_projection(vm.post_layernorm(layer.self_attn.out_proj(v)))
        return F.normalize(d[:, 1:], dim=-1).squeeze(0)

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [src, tgt, "couch", "cushion", "wall", "window"]
    label_txt = texts([f"a photo of a {l}" for l in labels])
    src_txt, tgt_txt = label_txt[0], label_txt[1]

    def zero_shot(feat):
        return ((feat @ label_txt.T) * 100).softmax(-1)

    orig_img = Image.open(args.original).convert("RGB")
    x0 = to01(orig_img)
    W0, H0 = orig_img.size
    orig_full = torch.from_numpy(np.asarray(orig_img, np.float32) / 255) \
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    mask_img = Image.open(args.mask).convert("L")
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), np.float32) / 255
    obj = (m.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5).reshape(-1)
    obj_t = torch.from_numpy(obj).to(DEVICE)
    mask_pix = torch.from_numpy(binary_dilation(
        np.asarray(mask_img.resize((RES, RES), Image.NEAREST)) > 127, iterations=2).astype(np.float32)) \
        .view(1, 1, RES, RES).to(DEVICE)

    def pooled(d):
        return F.normalize(d[obj_t].mean(0), dim=-1)

    def pgd(eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            d = value_joint(x)
            # targeted: away from source, toward target -- on VALUE features
            loss = (d[obj_t] @ src_txt).mean() - (d[obj_t] @ tgt_txt).mean()
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    def save_full(x, path):
        up = F.interpolate(x.detach() - x0, size=(H0, W0), mode="bicubic", align_corners=False)
        arr = ((orig_full + up).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def save_square(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    os.makedirs(args.outdir, exist_ok=True)
    sqdir = os.path.join(args.outdir, "square")
    os.makedirs(sqdir, exist_ok=True)
    save_square(x0, f"{sqdir}/baseline_eps0.png")

    pl0 = zero_shot(pooled(value_joint(x0).detach())).cpu().numpy()
    print(f"baseline pooled: P({src})={pl0[0]:.3f} P({tgt})={pl0[1]:.3f} argmax={labels[int(pl0.argmax())]}")
    budgets = [float(b) for b in args.budgets.split(",")]
    rows = {"b": [], "psrc": [], "ptgt": [], "argmax": []}
    summary = [f"labels={labels}",
               f"baseline pooled P({src})={pl0[0]:.4f} P({tgt})={pl0[1]:.4f}"]
    for b in budgets:
        x_adv = pgd(b / 255.0)
        d = value_joint(x_adv).detach()
        pl = zero_shot(pooled(d)).cpu().numpy()
        pp = zero_shot(d[obj_t]).mean(0).cpu().numpy()
        save_full(x_adv, f"{args.outdir}/{src}2{tgt}_eps{int(b)}.png")
        save_square(x_adv, f"{sqdir}/{src}2{tgt}_eps{int(b)}.png")
        rows["b"].append(b); rows["psrc"].append(float(pl[0])); rows["ptgt"].append(float(pl[1]))
        rows["argmax"].append(labels[int(pl.argmax())])
        line = (f"eps={b:>4.0f}/255  pooled P({src})={pl[0]:.3f} P({tgt})={pl[1]:.3f} "
                f"argmax={labels[int(pl.argmax())]}  (perpatch P({tgt})={pp[1]:.3f})")
        print(line); summary.append(line)

    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bx = [0] + [int(b) for b in budgets]
    ax[0].plot(bx, [pl0[0]] + rows["psrc"], "o-", color="tab:red", label=f"P({src})")
    ax[0].plot(bx, [pl0[1]] + rows["ptgt"], "o-", color="tab:blue", label=f"P({tgt})")
    ax[0].set_title(f"value-feature steering {src}->{tgt} (CLIP-space)")
    ax[0].set_xlabel("L_inf (/255)"); ax[0].set_ylabel("pooled-ROI zero-shot P"); ax[0].legend()
    p = f"{sqdir}/{src}2{tgt}_eps{int(budgets[-1])}.png"
    heat = zero_shot(value_joint(to01(Image.open(p).convert("RGB"))).detach())[:, 1] \
        .reshape(GRID, GRID).cpu().numpy()
    ax[1].imshow(Image.open(p).resize((RES, RES)))
    ax[1].imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[1].set_title(f'P("{tgt}") heatmap, eps{int(budgets[-1])}'); ax[1].axis("off")
    fig.suptitle(f"V-Attack (value features, text->{tgt}, single model) -- {args.model.split('/')[-1]}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/v_attack_cat.png", dpi=130)
    print(f"\nsaved {args.outdir}/ -> caption square/ via vlm_eval --object {tgt}")


if __name__ == "__main__":
    main()
