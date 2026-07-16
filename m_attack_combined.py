"""Stage 20: unified single-loss M-Attack + suppression (Option 2), same graph.

Both terms are computed on the SAME random crop's forward pass, so they share one
graph and sum into a single loss (no two-pass gradient averaging):

  per crop c = resized_crop(x_adv, box):
    steer_c = - cos( pooled(c),  pooled(resized_crop(target, box_t)) )   # M-Attack engine
    supp_c  =   cos( dense(c)[ROI-patches-in-c],  phi_T("dog") )         # ride-along suppression
    L_c     = lam_steer * steer_c + lam_supp * supp_c + lam_tv * TV(delta)

The suppression term rides on M-Attack's crops (keeps the transfer engine) and
acts only on the crop patches that overlap the object ROI. Whole-image
perturbation (M-Attack), MI-FGSM, single surrogate = LLaVA-1.5's encoder.

Because "away from dog" and "toward background" need not be the same direction,
the terms can fight. We log, each step: steer loss, supp loss, and the COSINE
between their gradients on delta (grad_align). grad_align < 0 => they oppose and
the sum partially cancels -- tune lam_supp/lam_steer or accept they conflict.

Example:
    uv run python m_attack_combined.py --lam-steer 1 --lam-supp 1 --steps 300 --eps 16
    uv run python vlm_eval.py --images-dir results/m_attack_combined \
        --object dog --models llava-hf/llava-1.5-7b-hf
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--lam-steer", type=float, default=1.0)
    ap.add_argument("--lam-supp", type=float, default=1.0)
    ap.add_argument("--lam-tv", type=float, default=0.0)
    ap.add_argument("--roi-thresh", type=float, default=0.3, help="patch counts as ROI if >thresh of it overlaps")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--outdir", default="results/m_attack_combined")
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
    PATCH = model.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"combined M-Attack+suppression  surrogate={args.model} grid={GRID}  "
          f"lam_steer={args.lam_steer} lam_supp={args.lam_supp} lam_tv={args.lam_tv}")

    def load(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def feats(x01):
        """one forward -> (pooled global embedding, dense patch features)."""
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        pooled = F.normalize(model.visual_projection(out.pooler_output), dim=-1).squeeze(0)
        layer = vm.encoder.layers[-1]
        h = out.hidden_states[-2]
        a = layer.layer_norm1(h)
        h = h + layer.self_attn.out_proj(layer.self_attn.v_proj(a))
        h = h + layer.mlp(layer.layer_norm2(h))
        dense = F.normalize(model.visual_projection(vm.post_layernorm(h))[:, 1:], dim=-1).squeeze(0)
        return pooled, dense

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [args.object, "cat", "couch", "cushion", "wall", "window"]
    label_txt = texts([f"a photo of a {l}" for l in labels])
    dog_txt = label_txt[0]

    x0, x_tgt = load(args.source), load(args.target)
    tgt_global = feats(x_tgt)[0].detach()
    mask_full = torch.from_numpy(
        (np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.NEAREST)) > 127).astype(np.float32)
    ).view(1, 1, RES, RES).to(DEVICE)
    mg = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
    obj_full = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5).reshape(-1)).to(DEVICE)
    dummy = torch.zeros(3, RES, RES)
    scale, ratio = tuple(args.crop_scale), (3 / 4, 4 / 3)

    def roi_patches_in_crop(box):
        i, j, h, w = box
        rc = TF.resized_crop(mask_full, i, j, h, w, [RES, RES], TF.InterpolationMode.NEAREST)
        frac = rc.reshape(1, 1, GRID, PATCH, GRID, PATCH).mean(dim=(3, 5)).reshape(-1)
        return frac > args.roi_thresh

    def tv(d):
        return (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean() + (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()

    @torch.no_grad()
    def full_metrics(x01):
        pooled, dense = feats(x01)
        cos_t = (pooled @ tgt_global).item()
        gpd = float(((pooled[None] @ label_txt.T) * 100).softmax(-1).squeeze(0)[0])
        rpd = ((dense[obj_full] @ label_txt.T) * 100).softmax(-1)[:, 0].mean().item()
        return cos_t, gpd, rpd

    c0 = full_metrics(x0)
    print(f"baseline: cos(orig,target)={c0[0]:.3f} globalP(dog)={c0[1]:.3f} roiP(dog)={c0[2]:.3f}")

    delta = torch.zeros_like(x0, requires_grad=True)
    momentum = torch.zeros_like(x0)
    log_every = max(1, args.steps // 30)
    traj = []
    for step in range(args.steps):
        box_s = T.RandomResizedCrop.get_params(dummy, scale, ratio)
        box_t = T.RandomResizedCrop.get_params(dummy, scale, ratio)
        adv = torch.clamp(x0 + delta, 0, 1)
        crop_adv = TF.resized_crop(adv, *box_s, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
        pooled_adv, dense_adv = feats(crop_adv)
        with torch.no_grad():
            crop_tgt = TF.resized_crop(x_tgt, *box_t, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
            pooled_tgt = feats(crop_tgt)[0]

        steer = -(pooled_adv * pooled_tgt).sum()
        roi_p = roi_patches_in_crop(box_s)
        has_roi = bool(roi_p.any())
        supp = (dense_adv[roi_p] @ dog_txt).mean() if has_roi else torch.zeros((), device=DEVICE)

        # separate gradients -> combine (gives term-cooperation for free)
        gs, = torch.autograd.grad(steer, delta, retain_graph=True)
        if has_roi and args.lam_supp > 0:
            gp, = torch.autograd.grad(supp, delta, retain_graph=(args.lam_tv > 0))
        else:
            gp = torch.zeros_like(delta)
        gt = torch.autograd.grad(tv(delta), delta)[0] if args.lam_tv > 0 else torch.zeros_like(delta)
        g = args.lam_steer * gs + args.lam_supp * gp + args.lam_tv * gt

        with torch.no_grad():
            momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
            delta.add_(-alpha * momentum.sign()).clamp_(-eps, eps)   # descend combined loss
            delta.data = torch.clamp(x0 + delta, 0, 1) - x0
        delta.requires_grad_(True)

        if step % log_every == 0 or step == args.steps - 1:
            align = float("nan")
            if has_roi and gp.abs().sum() > 0:
                align = F.cosine_similarity(gs.flatten(), gp.flatten(), dim=0).item()
            traj.append((step, steer.item(), (supp.item() if has_roi else float("nan")), align))

    adv = torch.clamp(x0 + delta.detach(), 0, 1)
    cf = full_metrics(adv)

    os.makedirs(args.outdir, exist_ok=True)
    Image.fromarray((x0.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
        f"{args.outdir}/baseline.png")
    Image.fromarray((adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
        f"{args.outdir}/combined_eps{int(args.eps)}.png")

    aligns = [t[3] for t in traj if not np.isnan(t[3])]
    mean_align = float(np.mean(aligns)) if aligns else float("nan")
    summary = [
        f"lam_steer={args.lam_steer} lam_supp={args.lam_supp} lam_tv={args.lam_tv} eps={args.eps}/255",
        f"cos(adv,target): {c0[0]:.4f} -> {cf[0]:.4f}",
        f"global P(dog):   {c0[1]:.4f} -> {cf[1]:.4f}",
        f"ROI   P(dog):    {c0[2]:.4f} -> {cf[2]:.4f}",
        f"mean grad_align(steer,supp) = {mean_align:.4f}   "
        f"({'COOPERATE' if mean_align > 0.05 else 'FIGHT' if mean_align < -0.05 else 'orthogonal'})",
    ]
    print("\n" + "\n".join(summary))
    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    st = [t[0] for t in traj]
    ax[0].plot(st, [t[1] for t in traj], "o-", color="tab:green", label="steer loss (-cos to target)")
    ax[0].plot(st, [t[2] for t in traj], "s-", color="tab:red", label="supp loss (cos to dog)")
    ax[0].set_xlabel("step"); ax[0].set_title("do both terms decrease together?"); ax[0].legend()
    ax[1].plot(st, [t[3] for t in traj], "o-", color="tab:purple")
    ax[1].axhline(0, ls=":", c="gray")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("cos(grad_steer, grad_supp)")
    ax[1].set_title(f"term cooperation (mean={mean_align:.3f}: >0 cooperate, <0 fight)")
    fig.suptitle("unified M-Attack + suppression (Option 2) — term cooperation", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/cooperation.png", dpi=130)
    print(f"\nsaved {args.outdir}/ -> caption via vlm_eval on llava-1.5-7b")


if __name__ == "__main__":
    main()
