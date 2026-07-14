"""Stage 8: LOCAL patch-token camouflage (the VLM-faithful attack).

VLMs feed the grid of PATCH tokens (CLS discarded) into the LLM, so "do you see
a dog?" is answered from the local visual tokens over the object region -- not a
pooled global embedding. The correct attack therefore steers the dog-region
patch tokens away from "dog" (and toward "couch"), i.e. local camouflage.

This directly contrasts a LOCAL patch objective against the GLOBAL CLS objective
at matched L_inf, both mask-confined, and reads out dog-ness with an INDEPENDENT
dense (MaskCLIP) probe -- so a P(dog) drop reflects genuine local suppression,
not overfitting to the attacked head.

    loss_local = mean_{i in obj}  cos(patch_i, "dog") - cos(patch_i, "couch")
    loss_cls   =                  cos(CLS,     "dog") - cos(CLS,     "couch")

Example:
    uv run python pgd_patch.py --budgets 2,4,8,16 --iters 300
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
RES, PATCH, GRID = 448, 16, 28

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
_ip = processor.image_processor
MEAN = torch.tensor(_ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(_ip.image_std, device=DEVICE).view(1, 3, 1, 1)


def to_pixel01(img):
    arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def clip_normalize(x01):
    return (x01 - MEAN) / STD


def embed_grad(x01):
    """Differentiable normalized (cls, standard patch tokens)."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True)
    proj = model.visual_projection(vm.post_layernorm(out.last_hidden_state))
    return F.normalize(proj[:, 0], dim=-1).squeeze(0), F.normalize(proj[:, 1:], dim=-1).squeeze(0)


@torch.no_grad()
def embed_dense(x01):
    """INDEPENDENT dense readout (MaskCLIP value-projection, penultimate layer)."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True, output_hidden_states=True)
    layer = vm.encoder.layers[-1]
    h = out.hidden_states[-2]
    x = layer.layer_norm1(h)
    x = layer.self_attn.out_proj(layer.self_attn.v_proj(x))
    h = h + x
    h = h + layer.mlp(layer.layer_norm2(h))
    dense = model.visual_projection(vm.post_layernorm(h))
    return F.normalize(dense[:, 1:], dim=-1).squeeze(0), F.normalize(out.hidden_states[-2][:, 1:], dim=-1).squeeze(0)


def dense_grad(x01):
    """Differentiable dense (MaskCLIP) patch features -- the localizing head a
    VLM projector is closest to. Same math as embed_dense, grad enabled."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True, output_hidden_states=True)
    layer = vm.encoder.layers[-1]
    h = out.hidden_states[-2]
    x = layer.layer_norm1(h)
    x = layer.self_attn.out_proj(layer.self_attn.v_proj(x))
    h = h + x
    h = h + layer.mlp(layer.layer_norm2(h))
    dense = model.visual_projection(vm.post_layernorm(h))
    return F.normalize(dense[:, 1:], dim=-1).squeeze(0)


@torch.no_grad()
def embed_texts(prompts):
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)


def regions(mask_img):
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), np.float32) / 255.0
    frac = m.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
    obj = frac > 0.5
    dist = distance_transform_edt(~obj)
    return obj, (~obj) & (dist > 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--target", default="removed_aligned.png",
                    help="dog-free target image (aligned); the image anchor")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--budgets", default="2,4,8,16")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    obj_name = args.object

    orig_img = Image.open(args.original).convert("RGB")
    tgt_img = Image.open(args.target).convert("RGB")
    x0 = to_pixel01(orig_img)
    x_tgt = to_pixel01(tgt_img)
    mask_img = Image.open(args.mask).convert("L")
    obj, far = regions(mask_img)
    obj_t = torch.from_numpy(obj.reshape(-1)).to(DEVICE)
    far_t = torch.from_numpy(far.reshape(-1)).to(DEVICE)
    mask_pix = torch.from_numpy(binary_dilation(np.asarray(
        mask_img.resize((RES, RES), Image.NEAREST)) > 127, iterations=2).astype(np.float32)) \
        .view(1, 1, RES, RES).to(DEVICE)

    # ATTACK prompts (used to build the loss)
    dog_txt, couch_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch"])
    # HELD-OUT measurement prompts (synonyms) -- so a P(dog) drop cannot be
    # explained by overfitting the single attack prompt vector
    meas_dog = ["a corgi", "a puppy", "a dog sitting on a sofa"]
    meas_bg = ["an empty sofa", "a cushion", "living room furniture", "a bare couch"]
    meas_txt = embed_texts(meas_dog + meas_bg)
    nd = len(meas_dog)

    _, patch0 = embed_grad(x0)
    patch0 = patch0.detach()
    # IMAGE ANCHOR: features of the aligned dog-free target scene
    cls_tgt, _ = embed_grad(x_tgt)
    cls_tgt = cls_tgt.detach()
    dense_tgt, _ = embed_dense(x_tgt)
    dense_tgt = dense_tgt.detach()

    def p_dog(dense):
        """held-out P(dog-group): softmax over synonym prompts, sum over dog terms."""
        logits = (dense @ meas_txt.T) * 100
        return logits.softmax(-1)[:, :nd].sum(-1)

    def pgd(loss_fn, eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            cls, patch = embed_grad(x)
            dense = dense_grad(x)
            loss = loss_fn(cls, patch, dense)
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    objectives = {
        # IMAGE ANCHOR (the sensible one): match dog-region dense tokens to the
        # aligned dog-free target's tokens at the SAME positions. No text, no crops.
        "img-target (local)": lambda cls, patch, dense:
            -(dense[obj_t] * dense_tgt[obj_t]).sum(-1).mean(),
        # TEXT ANCHOR (contrast): suppress "dog"/promote "couch" on the same head
        "text (local)": lambda cls, patch, dense:
            (dense[obj_t] @ dog_txt).mean() - (dense[obj_t] @ couch_txt).mean(),
        # IMAGE ANCHOR on CLS (global, for contrast)
        "img-target (CLS)": lambda cls, patch, dense:
            -(cls @ cls_tgt),
    }

    @torch.no_grad()
    def evaluate(x_adv):
        cls, patch = embed_grad(x_adv)
        dense, penult = embed_dense(x_adv)
        pv = p_dog(dense)
        # penultimate-layer patch drift = what a VLM projector actually consumes
        vlm_drift = (1 - (penult[obj_t] * F.normalize(
            embed_dense(x0)[1][obj_t], dim=-1)).sum(-1)).mean().item()
        return dict(
            p_dog_mean=pv[obj_t].mean().item(),
            p_dog_max=pv[obj_t].max().item(),
            cos_to_target=(dense[obj_t] * dense_tgt[obj_t]).sum(-1).mean().item(),
            cls_dog=(cls @ dog_txt).item(),
            far_drift=(1 - (patch[far_t] * patch0[far_t]).sum(-1)).mean().item(),
            vlm_patch_drift=vlm_drift,
        ), pv.reshape(GRID, GRID).cpu().numpy()

    def save_image(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def slug(name):
        return name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

    img_dir = "results/adv_images"
    os.makedirs(img_dir, exist_ok=True)
    save_image(x0, f"{img_dir}/baseline.png")

    base, base_heat = evaluate(x0)
    print(f"baseline: P({obj_name}) mean={base['p_dog_mean']:.3f} max={base['p_dog_max']:.3f}")
    budgets = [float(b) for b in args.budgets.split(",")]
    rows, frames = {}, {}
    for name, lf in objectives.items():
        for b in budgets:
            x_adv = pgd(lf, b / 255.0)
            m, heat = evaluate(x_adv)
            rows.setdefault(name, []).append((b, m))
            # save the actual manipulated image the model would ingest
            save_image(x_adv, f"{img_dir}/{slug(name)}_eps{int(b)}.png")
            print(f"{name:<18} eps={b:>4.0f}/255  P({obj_name}) mean={m['p_dog_mean']:.3f} "
                  f"max={m['p_dog_max']:.3f}  cos->target={m['cos_to_target']:.3f}  "
                  f"far_drift={m['far_drift']:.3f}")
            if b == 8:
                frames[name] = (x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy(), heat)

    os.makedirs("results", exist_ok=True)
    with open("results/pgd_patch.txt", "w") as f:
        f.write(f"baseline P({obj_name}) mean={base['p_dog_mean']:.4f} max={base['p_dog_max']:.4f}\n")
        for name, rr in rows.items():
            for b, m in rr:
                f.write(f"{name} eps={b:.0f} " +
                        " ".join(f"{k}={v:.4f}" for k, v in m.items()) + "\n")

    # ---- figure ---------------------------------------------------------------
    n = len(objectives)
    fig = plt.figure(figsize=(6 * max(n, 2), 9))
    gs = fig.add_gridspec(2, n)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    c = {name: palette[i % len(palette)] for i, name in enumerate(objectives)}

    ax = fig.add_subplot(gs[0, 0:max(1, n // 2)])
    for name, rr in rows.items():
        ax.plot([b for b, _ in rr], [m["p_dog_mean"] for _, m in rr], "o-", color=c[name], label=name + " mean")
        ax.plot([b for b, _ in rr], [m["p_dog_max"] for _, m in rr], "s--", color=c[name], alpha=0.6, label=name + " max")
    ax.axhline(base["p_dog_mean"], ls=":", c="gray", label="baseline")
    ax.set_xlabel("L_inf (/255)"); ax.set_ylabel(f'held-out P("{obj_name}") in mask')
    ax.set_title("does an INDEPENDENT probe still see the dog?"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, max(1, n // 2):n])
    for name, rr in rows.items():
        ax.plot([b for b, _ in rr], [m["vlm_patch_drift"] for _, m in rr], "o-", color=c[name], label=name)
    ax.set_xlabel("L_inf (/255)"); ax.set_ylabel("penultimate patch drift (VLM tap)")
    ax.set_title("change in the tokens a VLM projector consumes"); ax.legend(fontsize=8)

    for i, (name, (img, heat)) in enumerate(frames.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(np.clip(img, 0, 1))
        ax.imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f'{name} eps=8 -> P("{obj_name}")', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("local patch camouflage: attacking the localizing head vs std tokens vs CLS "
                 "(held-out probe)", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/pgd_patch.png", dpi=130)
    print("\nsaved results/pgd_patch.{txt,png}")


if __name__ == "__main__":
    main()
