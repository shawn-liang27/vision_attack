"""Stage 6: is the inpaint fill THIS specific background, and is it recoverable
from surrounding context WITHOUT regenerating the background pixels?

Two parts:

  (A) Descriptive. Compare, in patch-token space, the masked region's embedding
      against three references: the surrounding ring of visible couch patches
      (context, observed — no generation), a far-away couch region, and the
      generic "a photo of a couch" text. If the honest fill matches the
      *surrounding* ring best, the removal is a SPECIFIC local continuation,
      not a generic couch.

  (B) Steering. Mask-confined PGD that pushes the masked patch tokens toward the
      mean embedding of the surrounding ring (target built only from observed
      context — the "supposed background" is never generated). Compared at
      matched L_inf against:
        target_img  : the real regenerated removed background (uses generation)
        target_text : the generic "empty couch" text
      If target_surround reproduces the removal effect (dog signal down, and the
      result lands near the real removed embedding), then the specific occluded
      background is inferable from context alone.

Example:
    uv run python context_completion.py --budgets 4,8,16 --iters 300
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
    """Differentiable (cls, patches), L2-normalized, standard last-layer tokens."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True)
    proj = model.visual_projection(vm.post_layernorm(out.last_hidden_state))
    cls = F.normalize(proj[:, 0], dim=-1).squeeze(0)
    patches = F.normalize(proj[:, 1:], dim=-1).squeeze(0)
    return cls, patches


@torch.no_grad()
def embed_dense(x01):
    """Dense MaskCLIP patch features (L2-normalized) for P(dog) readout."""
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
    pooled = model.text_model(**tok).pooler_output
    return F.normalize(model.text_projection(pooled), dim=-1)


def regions(mask_img):
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), np.float32) / 255.0
    frac = m.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
    obj = frac > 0.5
    dist = distance_transform_edt(~obj)
    surround = (~obj) & (dist >= 1) & (dist <= 3)   # visible couch ring around dog
    far = (~obj) & (dist > 8)                        # distant background
    return obj, surround, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--removed", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--budgets", default="4,8,16")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    obj_name = args.object

    orig_img = Image.open(args.original).convert("RGB")
    rem_img = Image.open(args.removed).convert("RGB")
    x0, x_rem = to_pixel01(orig_img), to_pixel01(rem_img)

    mask_img = Image.open(args.mask).convert("L")
    obj, surround, far = regions(mask_img)
    obj_t = torch.from_numpy(obj.reshape(-1)).to(DEVICE)
    sur_t = torch.from_numpy(surround.reshape(-1)).to(DEVICE)
    far_t = torch.from_numpy(far.reshape(-1)).to(DEVICE)

    mask_full = np.asarray(mask_img.resize((RES, RES), Image.NEAREST)) > 127
    mask_dil = binary_dilation(mask_full, iterations=2)
    mask_pix = torch.from_numpy(mask_dil.astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)

    cls_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch"])
    dog_txt, couch_txt = cls_txt[0], cls_txt[1]
    loc_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch",
                           "a photo of a wall", "a photo of a window"])

    cls_o, patch_o = embed_grad(x0)
    cls_r, patch_r = embed_grad(x_rem)
    cls_o, patch_o = cls_o.detach(), patch_o.detach()
    cls_r, patch_r = cls_r.detach(), patch_r.detach()

    def unit(v):
        return F.normalize(v, dim=-1)

    # context target: mean of the SURROUNDING visible couch patches (observed only)
    ctx_target = unit(patch_o[sur_t].mean(0))
    far_target = unit(patch_o[far_t].mean(0))

    def mean_cos(P, t):
        return (P @ t).mean().item()

    # ---- (A) descriptive: is the honest fill specifically the surround? -------
    print("=" * 68)
    print("(A) what does the honest removal fill the masked region with?")
    print(f"  masked-region patches, mean cosine to reference targets")
    print(f"  {'reference':<26}{'original':>12}{'removed':>12}")
    for name, tgt in [("surrounding couch ring", ctx_target),
                      ("far couch/background", far_target),
                      ('text "a photo of a couch"', couch_txt),
                      ('text "a photo of a '+obj_name+'"', dog_txt)]:
        print(f"  {name:<26}{mean_cos(patch_o[obj_t], tgt):>12.4f}"
              f"{mean_cos(patch_r[obj_t], tgt):>12.4f}")
    # how well does the surround predict the ACTUAL fill? (specificity)
    fill_mean = unit(patch_r[obj_t].mean(0))
    print(f"\n  cos(mean filled patch, surrounding-ring target) = "
          f"{(fill_mean @ ctx_target).item():.4f}")
    print(f"  cos(mean filled patch, far-background target)    = "
          f"{(fill_mean @ far_target).item():.4f}")
    print(f"  -> fill matches the SURROUND more than the far bg by "
          f"{(fill_mean @ ctx_target).item() - (fill_mean @ far_target).item():+.4f}")

    dense_o = embed_dense(x0)
    dense_r = embed_dense(x_rem)
    p_dog = lambda d: ((d @ loc_txt.T) * 100).softmax(-1)[:, 0]
    base_pdog = p_dog(dense_o)[obj_t].mean().item()
    rem_pdog = p_dog(dense_r)[obj_t].mean().item()

    # ---- (B) steering: reach the fill from context only, no regeneration ------
    def pgd(loss_fn, eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            loss = loss_fn(*embed_grad(x))
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    objectives = {
        # context-only: pull masked patches toward the observed surrounding ring
        "target_surround": lambda cls, P: -(P[obj_t] @ ctx_target).mean(),
        # generic text
        "target_text": lambda cls, P: (cls @ dog_txt) - (cls @ couch_txt),
        # regenerated background (uses generation) — the reference upper bound
        "target_img": lambda cls, P: -(cls @ cls_r),
    }

    budgets = [float(b) for b in args.budgets.split(",")]
    results = {k: [] for k in objectives}
    print("\n" + "=" * 68)
    print("(B) steering the masked patches (matched L_inf), effect on the scene")
    for name, lf in objectives.items():
        for b in budgets:
            x_adv = pgd(lf, b / 255.0)
            cls_a, patch_a = embed_grad(x_adv)
            dense_a = embed_dense(x_adv)
            row = dict(
                eps=b,
                p_dog=p_dog(dense_a)[obj_t].mean().item(),
                cos_to_ctx=(patch_a[obj_t] @ ctx_target).mean().item(),
                cos_to_removed_patch=(patch_a[obj_t] * patch_r[obj_t]).sum(-1).mean().item(),
                cls_to_removed=(cls_a @ cls_r).item(),
                cls_sim_dog=(cls_a @ dog_txt).item(),
                cls_sim_couch=(cls_a @ couch_txt).item(),
            )
            results[name].append(row)
            print(f"  {name:<16} eps={b:>4.0f}/255  "
                  f"P({obj_name})={row['p_dog']:.3f}  "
                  f"cos->ctx={row['cos_to_ctx']:.3f}  "
                  f"CLS->removed={row['cls_to_removed']:.3f}  "
                  f"couch={row['cls_sim_couch']:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/context_completion.txt", "w") as f:
        f.write(f"baseline P({obj_name}) in mask = {base_pdog:.4f}\n")
        f.write(f"honest-removal P({obj_name}) in mask = {rem_pdog:.4f}\n")
        f.write(f"cos(mean fill, surround) = {(fill_mean @ ctx_target).item():.4f}, "
                f"cos(mean fill, far bg) = {(fill_mean @ far_target).item():.4f}\n\n")
        for name in objectives:
            f.write(f"[{name}]\n")
            for r in results[name]:
                f.write("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in r.items()) + "\n")

    # ---- figure ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"target_surround": "tab:blue", "target_text": "tab:green", "target_img": "tab:red"}
    labels = {"target_surround": "context ring (no gen)",
              "target_text": "generic couch text",
              "target_img": "real removed bg (gen)"}
    for name in objectives:
        e = [r["eps"] for r in results[name]]
        ax[0].plot(e, [r["p_dog"] for r in results[name]], "o-", color=colors[name], label=labels[name])
        ax[1].plot(e, [r["cls_to_removed"] for r in results[name]], "o-", color=colors[name], label=labels[name])
        ax[2].plot(e, [r["cls_sim_couch"] for r in results[name]], "o-", color=colors[name], label=labels[name])
    ax[0].axhline(base_pdog, ls=":", c="gray", label="baseline")
    ax[0].axhline(rem_pdog, ls="--", c="k", label="honest removal")
    ax[0].set_title(f'dense P("{obj_name}") in mask'); ax[0].set_ylabel("P(dog)")
    ax[1].set_title("CLS cosine to the real removed image")
    ax[2].set_title('CLS cosine to "a photo of a couch"')
    for a in ax:
        a.set_xlabel("L_inf budget (/255)"); a.legend(fontsize=8)
    fig.suptitle("recovering the occluded background from context vs regeneration", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/context_completion.png", dpi=130)
    print("\nsaved results/context_completion.{txt,png}")


if __name__ == "__main__":
    main()
