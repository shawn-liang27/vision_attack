"""Stage 11: PGD steering with ONLY the generated-background image as anchor,
instrumented with the correct CLIP-space verification sequence.

Objective (image anchor, no text): push the ROI (dog-region) dense features
toward the aligned generated-background image's features at the same positions.

    loss = - mean_{i in ROI} cos( dense_adv[i],  dense_bg[i] )

Because the anchor is an IMAGE, the CLIP text zero-shot check below is fully
independent of the objective (no text-probe circularity).

Verification, cheapest first (reviewer's sequence):
  GATE 1  loss curve: cos(adv_ROI, bg) must climb, cos(adv_ROI, original) drop
          -- tracked per iteration. Flat => optimization bug, stop.
  GATE 2  per-patch before/after: ROI patches end close to bg anchor and far
          from where they started.
  VERIFY  CLIP zero-shot on the ROI vs a label set {dog + background classes},
          on (a) per-patch dense features and (b) the cleaner POOLED-ROI
          embedding. Success = argmax flips dog -> background.

Outputs under results/bg_anchor/: curves.csv, summary.txt, figure, adv images.

Example:
    uv run python pgd_bg_anchor.py --budgets 4,8,16 --iters 300
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
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--background", default="removed_aligned.png",
                    help="the generated dog-free background (aligned) -- the ONLY anchor")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--budgets", default="4,8,16")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--outdir", default="results/bg_anchor")
    args = ap.parse_args()
    obj_name = args.object
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
    print(f"model={args.model} res={RES} patch={PATCH} grid={GRID}")

    def to01(img):
        arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def dense(x01):
        """normalized dense (MaskCLIP) patch features in joint space; differentiable."""
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

    # zero-shot label set: object + scene background classes (independent of the image anchor)
    labels = [obj_name, "couch", "cushion", "wall", "window", "houseplant"]
    label_txt = texts([f"a photo of a {l}" for l in labels])

    def zero_shot(feat):
        """feat: (...,D) normalized. returns softmax distribution over labels."""
        return ((feat @ label_txt.T) * 100).softmax(-1)

    orig_img = Image.open(args.original).convert("RGB")
    bg_img = Image.open(args.background).convert("RGB")
    x0, x_bg = to01(orig_img), to01(bg_img)
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

    dense_bg = dense(x_bg).detach()      # anchor features
    dense_o0 = dense(x0).detach()        # original object features (t=0)

    def pooled(d):
        return F.normalize(d[obj_t].mean(0), dim=-1)

    def readout(d):
        """per-patch (mean over ROI) and pooled zero-shot distributions + P(obj)."""
        pp = zero_shot(d[obj_t]).mean(0)          # per-patch, averaged over ROI
        pl = zero_shot(pooled(d))                 # pooled-ROI (cleaner)
        return pp.cpu().numpy(), pl.cpu().numpy()

    os.makedirs(args.outdir, exist_ok=True)

    def save_full(x, path):
        up = F.interpolate(x.detach() - x0, size=(H0, W0), mode="bicubic", align_corners=False)
        arr = ((orig_full + up).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    log_every = max(1, args.iters // 25)

    def pgd(eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        traj = []
        for it in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            d = dense(x)
            loss = -(d[obj_t] * dense_bg[obj_t]).sum(-1).mean()
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
                if it % log_every == 0 or it == args.iters - 1:
                    dd = dense(torch.clamp(x0 + delta * mask_pix, 0, 1))
                    traj.append((it,
                                 (dd[obj_t] * dense_bg[obj_t]).sum(-1).mean().item(),   # cos->bg (up)
                                 (dd[obj_t] * dense_o0[obj_t]).sum(-1).mean().item(),   # cos->orig (down)
                                 float(zero_shot(pooled(dd))[0])))                      # pooled P(obj)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1), traj

    # baseline readout
    pp0, pl0 = readout(dense_o0)
    budgets = [float(b) for b in args.budgets.split(",")]
    print(f"\nGATE/VERIFY sequence (label set: {labels})")
    print(f"baseline: pooled argmax={labels[int(pl0.argmax())]}  P({obj_name})_pooled={pl0[0]:.3f} "
          f"P({obj_name})_perpatch={pp0[0]:.3f}")

    summary = [f"labels={labels}",
               f"baseline pooled P({obj_name})={pl0[0]:.4f} perpatch={pp0[0]:.4f} "
               f"argmax={labels[int(pl0.argmax())]}"]
    trajs, readouts, adv_dense = {}, {}, {}
    curve_rows = ["budget,iter,cos_to_bg,cos_to_orig,pooled_p_obj"]
    for b in budgets:
        x_adv, traj = pgd(b / 255.0)
        d_adv = dense(x_adv).detach()
        pp, pl = readout(d_adv)
        trajs[b], readouts[b], adv_dense[b] = traj, (pp, pl), d_adv
        save_full(x_adv, f"{args.outdir}/bg_anchor_eps{int(b)}.png")
        for it, cb, co, pobj in traj:
            curve_rows.append(f"{int(b)},{it},{cb:.4f},{co:.4f},{pobj:.4f}")

        # GATE 1: did the loss move?
        cb0, cbT = traj[0][1], traj[-1][1]
        co0, coT = traj[0][2], traj[-1][2]
        # GATE 2: per-patch after
        to_bg_after = (d_adv[obj_t] * dense_bg[obj_t]).sum(-1)
        to_orig_after = (d_adv[obj_t] * dense_o0[obj_t]).sum(-1)
        flip = labels[int(pl.argmax())]
        line = (f"eps={b:>4.0f}/255 | GATE1 cos->bg {cb0:.3f}->{cbT:.3f} "
                f"cos->orig {co0:.3f}->{coT:.3f} | GATE2 ROI->bg {to_bg_after.mean():.3f} "
                f"ROI->orig {to_orig_after.mean():.3f} | VERIFY pooled P({obj_name}) "
                f"{pl0[0]:.3f}->{pl[0]:.3f} argmax={flip}")
        print(line)
        summary.append(line)

    with open(f"{args.outdir}/curves.csv", "w") as f:
        f.write("\n".join(curve_rows) + "\n")
    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(budgets)))

    for b, c in zip(budgets, colors):
        it = [r[0] for r in trajs[b]]
        ax[0, 0].plot(it, [r[1] for r in trajs[b]], "-", color=c, label=f"eps{int(b)} ->bg")
        ax[0, 0].plot(it, [r[2] for r in trajs[b]], "--", color=c, label=f"eps{int(b)} ->orig")
    ax[0, 0].set_title("GATE 1: cos(adv ROI, bg) up / cos(adv ROI, original) down")
    ax[0, 0].set_xlabel("iteration"); ax[0, 0].set_ylabel("mean cosine"); ax[0, 0].legend(fontsize=7)

    bx = [0] + [int(b) for b in budgets]
    ax[0, 1].plot(bx, [pl0[0]] + [readouts[b][1][0] for b in budgets], "o-", label="pooled ROI")
    ax[0, 1].plot(bx, [pp0[0]] + [readouts[b][0][0] for b in budgets], "s--", label="per-patch ROI")
    ax[0, 1].set_title(f'VERIFY: zero-shot P("{obj_name}") on ROI'); ax[0, 1].set_xlabel("L_inf (/255)")
    ax[0, 1].set_ylabel(f"P({obj_name})"); ax[0, 1].legend(fontsize=8)

    bmax = budgets[-1]
    xlab = np.arange(len(labels)); w = 0.35
    ax[0, 2].bar(xlab - w / 2, pl0, w, label="baseline")
    ax[0, 2].bar(xlab + w / 2, readouts[bmax][1], w, label=f"eps{int(bmax)}")
    ax[0, 2].set_xticks(xlab); ax[0, 2].set_xticklabels(labels, rotation=30, ha="right")
    ax[0, 2].set_title(f"pooled-ROI zero-shot distribution (eps{int(bmax)})"); ax[0, 2].legend(fontsize=8)

    tb = (adv_dense[bmax][obj_t] * dense_bg[obj_t]).sum(-1).cpu().numpy()
    tb0 = (dense_o0[obj_t] * dense_bg[obj_t]).sum(-1).cpu().numpy()
    ax[1, 0].hist(tb0, bins=25, alpha=0.6, label="before"); ax[1, 0].hist(tb, bins=25, alpha=0.6, label="after")
    ax[1, 0].set_title("GATE 2: per-patch cos(ROI, bg anchor)"); ax[1, 0].legend(fontsize=8)

    to = (adv_dense[bmax][obj_t] * dense_o0[obj_t]).sum(-1).cpu().numpy()
    ax[1, 1].hist(to, bins=25, color="tab:red", alpha=0.75)
    ax[1, 1].axvline(1.0, ls=":", c="gray"); ax[1, 1].set_title("per-patch cos(ROI, original) after (1=unmoved)")

    ax[1, 2].imshow(Image.open(f"{args.outdir}/bg_anchor_eps{int(bmax)}.png").resize((RES, RES)))
    heat = zero_shot(adv_dense[bmax])[:, 0].reshape(GRID, GRID).cpu().numpy()
    ax[1, 2].imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[1, 2].set_title(f'P("{obj_name}") heatmap, eps{int(bmax)}'); ax[1, 2].axis("off")

    fig.suptitle(f"PGD steering to generated-background anchor -- CLIP-space verification "
                 f"({args.model.split('/')[-1]})", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/verification.png", dpi=130)
    print(f"\nsaved {args.outdir}/verification.png, curves.csv, summary.txt, adv images")


if __name__ == "__main__":
    main()
