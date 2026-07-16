"""Stage 12: V-Attack -- steer VALUE features V, not patch/output features X,
plus an explicit "dog"-suppression term.

Diagnosis (stage 11): matching output/patch features X to the background moved
the ROI toward background yet P(dog) stayed high, because X mixes attention-borne
global-context channels that keep identity alive. V-Attack's fix: steer the
disentangled VALUE features V = v_proj(LN(h)) of the last block, where object
identity actually lives.

Three objectives compared on the SAME CLIP-space verification (pooled + per-patch
zero-shot P(dog)):
  X-match           : -cos(dense_adv[ROI], dense_bg[ROI])          (stage 11, output feats)
  V-match           : -cos(V_adv[ROI],     V_bg[ROI])              (the fix)
  V-match + suppress : V-match + beta * cos(dense_adv[ROI], phi_T("dog"))
                       (adds a force pointing AWAY from dog, not just toward bg)

Prediction: V-match drops P(dog) further than X-match; +suppress collapses it.

Example:
    uv run python pgd_v_attack.py --budgets 4,8,16 --iters 300 --beta 1.0
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
    ap.add_argument("--background", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--budgets", default="4,8,16")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--beta", type=float, default=1.0, help="dog-suppression weight")
    ap.add_argument("--outdir", default="results/v_attack")
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
    print(f"model={args.model} res={RES} grid={GRID} beta={args.beta}")

    def to01(img):
        arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def feats(x01):
        """one forward -> (dense joint-space output feature, raw value feature V).
        dense is the MaskCLIP value->joint feature used for readout & suppression;
        V is the raw last-block value projection (identity-disentangled)."""
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        layer = vm.encoder.layers[-1]
        h_in = out.hidden_states[-2]
        a = layer.layer_norm1(h_in)
        v = layer.self_attn.v_proj(a)                       # raw VALUE features
        h = h_in + layer.self_attn.out_proj(v)
        h = h + layer.mlp(layer.layer_norm2(h))
        d = model.visual_projection(vm.post_layernorm(h))   # joint-space output (dense)
        return (F.normalize(d[:, 1:], dim=-1).squeeze(0),
                F.normalize(v[:, 1:], dim=-1).squeeze(0))

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [obj_name, "couch", "cushion", "wall", "window", "houseplant"]
    label_txt = texts([f"a photo of a {l}" for l in labels])
    dog_txt = label_txt[0]

    def zero_shot(feat):
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

    dense_bg, V_bg = feats(x_bg)
    dense_bg, V_bg = dense_bg.detach(), V_bg.detach()

    def pooled(d):
        return F.normalize(d[obj_t].mean(0), dim=-1)

    def readout(d):
        return zero_shot(d[obj_t]).mean(0).cpu().numpy(), zero_shot(pooled(d)).cpu().numpy()

    def loss_X(dense, V):
        return -(dense[obj_t] * dense_bg[obj_t]).sum(-1).mean()

    def loss_V(dense, V):
        return -(V[obj_t] * V_bg[obj_t]).sum(-1).mean()

    def loss_Vsup(dense, V):
        return -(V[obj_t] * V_bg[obj_t]).sum(-1).mean() + args.beta * (dense[obj_t] @ dog_txt).mean()

    objectives = {"X-match": loss_X, "V-match": loss_V, "V-match+suppress": loss_Vsup}

    def pgd(loss_fn, eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            dense, V = feats(x)
            g, = torch.autograd.grad(loss_fn(dense, V), delta)
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

    os.makedirs(args.outdir, exist_ok=True)
    dense0, V0 = feats(x0)
    dense0, V0 = dense0.detach(), V0.detach()
    pp0, pl0 = readout(dense0)
    budgets = [float(b) for b in args.budgets.split(",")]
    print(f"labels={labels}")
    print(f"baseline pooled P({obj_name})={pl0[0]:.3f} perpatch={pp0[0]:.3f}")

    res = {n: {"pp": [], "pl": [], "cos_bg": []} for n in objectives}
    dist_at_max = {}
    summary = [f"labels={labels}", f"baseline pooled P({obj_name})={pl0[0]:.4f} perpatch={pp0[0]:.4f}"]
    for name, lf in objectives.items():
        for b in budgets:
            x_adv = pgd(lf, b / 255.0)
            dense_a, V_a = feats(x_adv)
            dense_a, V_a = dense_a.detach(), V_a.detach()
            pp, pl = readout(dense_a)
            # gate: similarity to bg in the space that objective matched
            cbg = ((V_a[obj_t] * V_bg[obj_t]).sum(-1).mean().item() if "V" in name
                   else (dense_a[obj_t] * dense_bg[obj_t]).sum(-1).mean().item())
            res[name]["pp"].append(pp[0]); res[name]["pl"].append(pl[0]); res[name]["cos_bg"].append(cbg)
            save_full(x_adv, f"{args.outdir}/{name.replace('+', '_').replace('-', '_')}_eps{int(b)}.png")
            if b == budgets[-1]:
                dist_at_max[name] = pl
            line = (f"{name:<18} eps={b:>4.0f}/255  pooled P({obj_name})={pl[0]:.3f} "
                    f"perpatch={pp[0]:.3f}  argmax={labels[int(pl.argmax())]}  cos->bg={cbg:.3f}")
            print(line); summary.append(line)

    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    bx = [0] + [int(b) for b in budgets]
    col = {"X-match": "tab:orange", "V-match": "tab:blue", "V-match+suppress": "tab:green"}

    for n in objectives:
        ax[0, 0].plot(bx, [pl0[0]] + res[n]["pl"], "o-", color=col[n], label=n)
        ax[0, 1].plot(bx, [pp0[0]] + res[n]["pp"], "o-", color=col[n], label=n)
    for a, t in [(ax[0, 0], "pooled ROI"), (ax[0, 1], "per-patch ROI")]:
        a.axhline(pl0[0] if t == "pooled ROI" else pp0[0], ls=":", c="gray")
        a.set_title(f'VERIFY: zero-shot P("{obj_name}") -- {t}'); a.set_xlabel("L_inf (/255)")
        a.set_ylabel(f"P({obj_name})"); a.legend(fontsize=8)

    for n in objectives:
        ax[0, 2].plot(bx[1:], res[n]["cos_bg"], "o-", color=col[n], label=n)
    ax[0, 2].set_title("GATE: cos(adv ROI, bg) in matched space"); ax[0, 2].set_xlabel("L_inf (/255)")
    ax[0, 2].legend(fontsize=8)

    xl = np.arange(len(labels)); w = 0.2
    ax[1, 0].bar(xl - 1.5 * w, pl0, w, label="baseline")
    for i, n in enumerate(objectives):
        ax[1, 0].bar(xl + (i - 0.5) * w, dist_at_max[n], w, label=n)
    ax[1, 0].set_xticks(xl); ax[1, 0].set_xticklabels(labels, rotation=30, ha="right")
    ax[1, 0].set_title(f"pooled-ROI zero-shot dist (eps{int(budgets[-1])})"); ax[1, 0].legend(fontsize=7)

    # heatmaps for X-match vs V-match+suppress at max budget
    for a, n in [(ax[1, 1], "X-match"), (ax[1, 2], "V-match+suppress")]:
        p = f"{args.outdir}/{n.replace('+', '_').replace('-', '_')}_eps{int(budgets[-1])}.png"
        d_a, _ = feats(to01(Image.open(p).convert("RGB")))
        heat = zero_shot(d_a.detach())[:, 0].reshape(GRID, GRID).cpu().numpy()
        a.imshow(Image.open(p).resize((RES, RES)))
        a.imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        a.set_title(f'{n}: P("{obj_name}") heatmap'); a.axis("off")

    fig.suptitle(f"V-Attack: steer VALUE features (+dog suppression) vs X-match "
                 f"-- {args.model.split('/')[-1]}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/v_attack.png", dpi=130)
    print(f"\nsaved {args.outdir}/v_attack.png, summary.txt, adv images")


if __name__ == "__main__":
    main()
