"""Stage 5: mask-confined white-box PGD on CLIP vs the interpolation oracle.

Answers: is the ~150/255 needed to flip "dog" a property of CLIP's geometry,
or just of the (unoptimized) interpolation direction? PGD searches for the
*best* mask-confined perturbation under an L_inf budget; interpolation follows
the fixed natural-image direction. Overlaying both on the same L_inf axis
separates "direction is bad" from "background has no low-norm representation".

Threat-model note: this is WHITE-BOX PGD on CLIP itself, the cleanest probe of
direction-vs-geometry. It is NOT the black-box transfer setting of M-Attack-V2;
the 8/255 comparison is about the perturbation budget, not their success metric.

Objectives (--objective):
  suppress    : minimize cos(CLS, "a photo of a dog")                  [untargeted]
  target_text : minimize cos(CLS, dog) + maximize cos(CLS, "empty couch")
  target_img  : maximize cos(CLS, removed-image CLS)   [embedding match, M-Attack-like]

The perturbation is confined to the dilated object mask and clamped to an
L_inf ball in [0,1] pixel space, exactly like a real attack image.

Example:
    uv run python pgd_attack.py --objective target_img \
        --budgets 4,8,16,32,64 --iters 300
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
from transformers import CLIPModel, CLIPProcessor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch16"
RES = 448
PATCH = 16
GRID = RES // PATCH

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

_ip = processor.image_processor
MEAN = torch.tensor(_ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(_ip.image_std, device=DEVICE).view(1, 3, 1, 1)


def to_pixel01(img):
    """PIL -> (1,3,RES,RES) float tensor in [0,1] (before CLIP normalization)."""
    arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def clip_normalize(x01):
    return (x01 - MEAN) / STD


def cls_embed(x01):
    """Differentiable CLS embedding in CLIP joint space (L2-normalized)."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True)
    cls = model.visual_projection(vm.post_layernorm(out.last_hidden_state)[:, 0])
    return F.normalize(cls, dim=-1)


@torch.no_grad()
def full_embed(x01):
    """(cls, standard patches, dense MaskCLIP patches), all L2-normalized."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True, output_hidden_states=True)
    hidden = vm.post_layernorm(out.last_hidden_state)
    proj = model.visual_projection(hidden)
    cls = F.normalize(proj[:, 0], dim=-1).squeeze(0)
    patches = F.normalize(proj[:, 1:], dim=-1).squeeze(0)
    layer = vm.encoder.layers[-1]
    h = out.hidden_states[-2]
    x = layer.layer_norm1(h)
    x = layer.self_attn.out_proj(layer.self_attn.v_proj(x))
    h = h + x
    h = h + layer.mlp(layer.layer_norm2(h))
    dense = model.visual_projection(vm.post_layernorm(h))
    dense = F.normalize(dense[:, 1:], dim=-1).squeeze(0)
    return cls, patches, dense


@torch.no_grad()
def embed_texts(prompts):
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    pooled = model.text_model(**tok).pooler_output
    return F.normalize(model.text_projection(pooled), dim=-1)


def mask_grid_and_regions(mask01_res):
    frac = mask01_res.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))
    obj = frac > 0.5
    dist = distance_transform_edt(~obj)
    return obj, (~obj) & (dist <= 2), (~obj) & (dist > 2)


def pgd(x0, mask_t, loss_fn, eps, iters, step):
    """Mask-confined L_inf PGD. Minimizes loss_fn(cls_embed(x)). Returns x*."""
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(iters):
        x = torch.clamp(x0 + delta * mask_t, 0, 1)
        loss = loss_fn(cls_embed(x))
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * grad.sign()
            delta.clamp_(-eps, eps)
        delta.requires_grad_(True)
    return torch.clamp(x0 + delta.detach() * mask_t, 0, 1)


def metrics(x01, patch_0, patch_1, obj, ring, far, loc_txt, cls_txt, obj_name):
    cls, patch, _ = full_embed(x01)
    _, _, dense = full_embed(x01)
    prob = ((dense @ loc_txt.T) * 100).softmax(-1)[:, 0]
    obj_flat = torch.from_numpy(obj.reshape(-1)).to(DEVICE)
    drift = (1 - (patch_0 * patch).sum(-1)).reshape(GRID, GRID).cpu().numpy()
    sims = (cls @ cls_txt.T).cpu().numpy()
    return dict(
        p_obj_mean=prob[obj_flat].mean().item(),
        p_obj_max=prob[obj_flat].max().item(),
        cls_sim_obj=float(sims[0]),
        cls_sim_couch=float(sims[1]),
        cos_to_removed=(patch * patch_1).sum(-1)[obj_flat].mean().item(),
        drift_obj=drift[obj].mean(),
        drift_far=drift[far].mean(),
    ), prob.reshape(GRID, GRID).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--removed", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--objective", default="target_img",
                    choices=["suppress", "target_text", "target_img"])
    ap.add_argument("--budgets", default="2,4,8,16,32,64,128",
                    help="comma L_inf budgets in /255")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.objective
    obj_name = args.object

    orig_img = Image.open(args.original).convert("RGB")
    rem_img = Image.open(args.removed).convert("RGB")
    x0 = to_pixel01(orig_img)
    x_rem = to_pixel01(rem_img)

    mask_full = np.asarray(Image.open(args.mask).convert("L").resize((RES, RES), Image.NEAREST)) > 127
    mask_dil = binary_dilation(mask_full, iterations=2)
    mask_t = torch.from_numpy(mask_dil.astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)
    obj, ring, far = mask_grid_and_regions(mask_full.astype(np.float32))

    loc_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch",
                           "a photo of a wall", "a photo of a window"])
    cls_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch"])
    dog_txt, couch_txt = cls_txt[0], cls_txt[1]
    rem_cls = cls_embed(x_rem).detach().squeeze(0)

    if args.objective == "suppress":
        loss_fn = lambda c: (c.squeeze(0) @ dog_txt)
    elif args.objective == "target_text":
        loss_fn = lambda c: (c.squeeze(0) @ dog_txt) - (c.squeeze(0) @ couch_txt)
    else:  # target_img
        loss_fn = lambda c: -(c.squeeze(0) @ rem_cls)

    patch_0, patch_1 = full_embed(x0)[1], full_embed(x_rem)[1]

    budgets = [float(b) for b in args.budgets.split(",")]
    # baseline (eps=0) + interpolation-oracle reference at matched L_inf
    delta_px = (x_rem - x0)  # in [0,1]; interpolation direction
    dmax = delta_px.abs().max().item()

    rows, frames = [], {}
    base, _ = metrics(x0, patch_0, patch_1, obj, ring, far, loc_txt, cls_txt, obj_name)
    base.update(eps=0.0, kind="baseline")
    rows.append(base)
    print(f"baseline: P({obj_name})={base['p_obj_mean']:.3f} "
          f"sim_{obj_name}={base['cls_sim_obj']:.4f}")

    for b in budgets:
        eps = b / 255.0
        step = 2.5 * eps / args.iters
        x_adv = pgd(x0, mask_t, loss_fn, eps, args.iters, step)
        m_pgd, heat = metrics(x_adv, patch_0, patch_1, obj, ring, far, loc_txt, cls_txt, obj_name)
        m_pgd.update(eps=b, kind="pgd")
        rows.append(m_pgd)

        # interpolation oracle at the SAME L_inf budget: t s.t. t*dmax = eps
        t = min(eps / dmax, 1.0)
        x_int = torch.clamp(x0 + t * delta_px * mask_t, 0, 1)
        m_int, _ = metrics(x_int, patch_0, patch_1, obj, ring, far, loc_txt, cls_txt, obj_name)
        m_int.update(eps=b, kind="interp", t=t)
        rows.append(m_int)

        print(f"eps={b:>5.0f}/255  PGD P({obj_name})={m_pgd['p_obj_mean']:.3f} "
              f"sim_{obj_name}={m_pgd['cls_sim_obj']:.4f} far_drift={m_pgd['drift_far']:.4f}"
              f"  |  interp(t={t:.3f}) P={m_int['p_obj_mean']:.3f}")

        if b in (8, 64) or b == budgets[-1]:
            frames[b] = (x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy(), heat)

    os.makedirs("results", exist_ok=True)
    keys = ["eps", "kind", "p_obj_mean", "p_obj_max", "cls_sim_obj", "cls_sim_couch",
            "cos_to_removed", "drift_obj", "drift_far"]
    with open(f"results/pgd_{tag}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # --- figure ---------------------------------------------------------------
    pgd_rows = [r for r in rows if r["kind"] == "pgd"]
    int_rows = [r for r in rows if r["kind"] == "interp"]
    eps_pgd = [r["eps"] for r in pgd_rows]
    eps_int = [r["eps"] for r in int_rows]

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.semilogx(eps_pgd, [r["p_obj_mean"] for r in pgd_rows], "o-", label=f"PGD ({args.objective})")
    ax.semilogx(eps_int, [r["p_obj_mean"] for r in int_rows], "s--", label="interpolation oracle")
    ax.axhline(base["p_obj_mean"], ls=":", color="gray", label="baseline")
    for e in (8, 16):
        ax.axvline(e, ls=":", color="crimson", alpha=0.5)
    ax.set_xlabel("L_inf budget (/255, log)"); ax.set_ylabel(f'dense P("{obj_name}") in mask')
    ax.set_title("dog signal: optimized vs interpolation"); ax.legend()

    ax = fig.add_subplot(gs[0, 2:4])
    ax.semilogx(eps_pgd, [r["cls_sim_obj"] for r in pgd_rows], "o-", label=f'PGD: "{obj_name}"')
    ax.semilogx(eps_pgd, [r["cls_sim_couch"] for r in pgd_rows], "o-", label='PGD: "couch"')
    ax.semilogx(eps_int, [r["cls_sim_obj"] for r in int_rows], "s--", label=f'interp: "{obj_name}"')
    ax.axvline(8, ls=":", color="crimson", alpha=0.5)
    ax.set_xlabel("L_inf budget (/255, log)"); ax.set_ylabel("CLS image-text cos")
    ax.set_title("global semantics vs budget"); ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(eps_pgd, [r["drift_far"] for r in pgd_rows], "o-", label="PGD")
    ax.semilogx(eps_int, [r["drift_far"] for r in int_rows], "s--", label="interp")
    ax.set_xlabel("L_inf (/255)"); ax.set_ylabel("far-bg drift")
    ax.set_title("attention bleed"); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(eps_pgd, [r["drift_obj"] for r in pgd_rows], "o-", label="PGD")
    ax.semilogx(eps_int, [r["drift_obj"] for r in int_rows], "s--", label="interp")
    ax.set_xlabel("L_inf (/255)"); ax.set_ylabel("object drift")
    ax.set_title("in-mask token drift"); ax.legend()

    for i, (b, (img, heat)) in enumerate(sorted(frames.items())[:2]):
        ax = fig.add_subplot(gs[1, 2 + i])
        ax.imshow(np.clip(img, 0, 1))
        ax.imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"PGD eps={b:.0f}/255", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"mask-confined PGD on CLIP ({args.objective}) vs interpolation oracle",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f"results/pgd_{tag}.png", dpi=130)
    print(f"\nsaved results/pgd_{tag}.csv and results/pgd_{tag}.png")


if __name__ == "__main__":
    main()
