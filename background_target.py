"""Stage 7: better background targets, no full reconstruction.

Addresses two questions:

  Q1 (attention-ablated target). Build an object-free target by masking the
      object patch tokens OUT of attention (no query, incl. CLS, attends to
      them). The resulting CLS is a genuine context-only reading of the scene,
      derived from the real image with zero pixels generated. Steering the real
      image's CLS toward it is a principled generation-free "remove object"
      target -- and, unlike the neighbor-patch mean, it is not a degenerate
      anisotropic-cone target.

  Q2 (how generic can the substitute be?). Build the target from a set of
      generic couch images (--anchors <dir>) instead of the specific
      reconstructed background. Sweep the anchor-set size to expose the
      genericity/specificity tradeoff: does any-couch suffice to remove the dog,
      and how close does it get to the SPECIFIC removed scene?

All targets are compared at matched L_inf against the established bounds:
  target_text (generic text, lower bound) and target_img (real removed, upper).

Example:
    uv run python background_target.py --budgets 8,16 --iters 200 \
        --anchors couch_anchors/
"""

import argparse
import glob
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
    """Differentiable normalized (cls, patches), full attention (normal forward)."""
    vm = model.vision_model
    out = vm(clip_normalize(x01), interpolate_pos_encoding=True)
    proj = model.visual_projection(vm.post_layernorm(out.last_hidden_state))
    return F.normalize(proj[:, 0], dim=-1).squeeze(0), F.normalize(proj[:, 1:], dim=-1).squeeze(0)


@torch.no_grad()
def embed_ablated_cls(x01, drop_grid):
    """CLS with object patch tokens masked OUT of attention (as keys).

    drop_grid: (GRID,GRID) bool, True where the object is. Returns normalized CLS
    of the scene as if those tokens contributed nothing -- generation-free.
    """
    vm = model.vision_model
    emb = vm.embeddings(clip_normalize(x01), interpolate_pos_encoding=True)
    pre = getattr(vm, "pre_layrnorm", getattr(vm, "pre_layernorm", None))
    hidden = pre(emb) if pre is not None else emb

    seq = hidden.shape[1]  # 1 + GRID*GRID
    keep = torch.ones(seq, dtype=torch.bool, device=DEVICE)
    keep[1:] = ~torch.from_numpy(drop_grid.reshape(-1)).to(DEVICE)  # CLS kept
    bias = torch.zeros(seq, device=DEVICE, dtype=hidden.dtype)
    bias[~keep] = torch.finfo(hidden.dtype).min
    attn_mask = bias.view(1, 1, 1, seq).expand(1, 1, seq, seq)

    enc = vm.encoder(inputs_embeds=hidden, attention_mask=attn_mask)
    last = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
    cls = model.visual_projection(vm.post_layernorm(last))[:, 0]
    return F.normalize(cls, dim=-1).squeeze(0)


@torch.no_grad()
def embed_dense(x01):
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


@torch.no_grad()
def embed_image_cls(img):
    return embed_grad(to_pixel01(img))[0]


def obj_grid(mask_img):
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), np.float32) / 255.0
    frac = m.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
    return frac > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--removed", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--anchors", default=None, help="dir of generic couch images (Q2)")
    ap.add_argument("--budgets", default="8,16")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    obj_name = args.object

    orig_img = Image.open(args.original).convert("RGB")
    rem_img = Image.open(args.removed).convert("RGB")
    x0, x_rem = to_pixel01(orig_img), to_pixel01(rem_img)

    mask_img = Image.open(args.mask).convert("L")
    obj = obj_grid(mask_img)
    obj_t = torch.from_numpy(obj.reshape(-1)).to(DEVICE)
    mask_full = np.asarray(mask_img.resize((RES, RES), Image.NEAREST)) > 127
    mask_pix = torch.from_numpy(binary_dilation(mask_full, iterations=2).astype(np.float32)) \
        .view(1, 1, RES, RES).to(DEVICE)

    cls_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch"])
    dog_txt, couch_txt = cls_txt[0], cls_txt[1]
    loc_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch",
                           "a photo of a wall", "a photo of a window"])

    cls_r = embed_grad(x_rem)[0].detach()
    p_dog = lambda d: ((d @ loc_txt.T) * 100).softmax(-1)[:, 0]
    base_pdog = p_dog(embed_dense(x0))[obj_t].mean().item()
    rem_pdog = p_dog(embed_dense(x_rem))[obj_t].mean().item()

    # ---- build the candidate targets ----------------------------------------
    targets = {}
    try:
        abl = embed_ablated_cls(x0, obj)
        targets["ablate (no gen)"] = abl
        print(f"attention-ablated target built. cos(ablated, real removed CLS) = "
              f"{(abl @ cls_r).item():.4f}  cos(ablated, orig CLS) = "
              f"{(abl @ embed_grad(x0)[0]).item():.4f}")
    except Exception as e:
        print(f"WARNING: attention-ablation failed ({type(e).__name__}: {e}); skipping Q1")

    targets["generic text"] = None  # handled via loss below
    targets["real removed (gen)"] = cls_r

    # Q2 anchors: centroid of generic couch images, swept by set size
    anchor_cls = None
    if args.anchors:
        paths = sorted(sum([glob.glob(os.path.join(args.anchors, e))
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.webp")], []))
        if paths:
            anchor_cls = torch.stack([embed_image_cls(Image.open(p).convert("RGB")) for p in paths])
            print(f"loaded {len(paths)} couch anchors. "
                  f"mean cos(anchor, real removed CLS) = "
                  f"{(anchor_cls @ cls_r).mean().item():.4f}")
        else:
            print(f"no images found in {args.anchors}; skipping Q2 anchor sweep")

    # ---- PGD -----------------------------------------------------------------
    def pgd(loss_fn, eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            cls, _ = embed_grad(x)
            loss = loss_fn(cls)
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    def evaluate(x_adv, target_cls=None):
        cls, _ = embed_grad(x_adv)
        return dict(
            p_dog=p_dog(embed_dense(x_adv))[obj_t].mean().item(),
            cls_to_removed=(cls @ cls_r).item(),
            cls_sim_dog=(cls @ dog_txt).item(),
            cls_sim_couch=(cls @ couch_txt).item(),
            cls_to_target=(cls @ target_cls).item() if target_cls is not None else float("nan"),
        )

    budgets = [float(b) for b in args.budgets.split(",")]
    lines = [f"baseline P({obj_name})={base_pdog:.4f}  honest-removal P({obj_name})={rem_pdog:.4f}\n"]
    print("\n" + "=" * 70)
    fig_rows = {}
    for name, tgt in targets.items():
        for b in budgets:
            eps = b / 255.0
            if name == "generic text":
                lf = lambda c: (c @ dog_txt) - (c @ couch_txt)
                tc = couch_txt
            else:
                lf = (lambda t: (lambda c: -(c @ t)))(tgt)
                tc = tgt
            m = evaluate(pgd(lf, eps), tc)
            fig_rows.setdefault(name, []).append((b, m))
            line = (f"{name:<20} eps={b:>4.0f}/255  P({obj_name})={m['p_dog']:.3f}  "
                    f"CLS->removed={m['cls_to_removed']:.3f}  couch={m['cls_sim_couch']:.3f}  "
                    f"dog={m['cls_sim_dog']:.3f}")
            print(line); lines.append(line + "\n")

    # Q2 genericity sweep: target = centroid of first-k anchors, at max budget
    sweep = []
    if anchor_cls is not None:
        b = budgets[-1]
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= len(anchor_cls)] + [len(anchor_cls)]
        ks = sorted(set(ks))
        print("\n" + "-" * 70 + f"\nQ2 anchor genericity sweep (eps={b:.0f}/255):")
        for k in ks:
            centroid = F.normalize(anchor_cls[:k].mean(0), dim=-1)
            lf = (lambda t: (lambda c: -(c @ t)))(centroid)
            m = evaluate(pgd(lf, b / 255.0), centroid)
            m["cos_centroid_removed"] = (centroid @ cls_r).item()
            sweep.append((k, m))
            line = (f"  k={k:<3} P({obj_name})={m['p_dog']:.3f}  CLS->removed={m['cls_to_removed']:.3f}"
                    f"  couch={m['cls_sim_couch']:.3f}  cos(centroid,removed)={m['cos_centroid_removed']:.3f}")
            print(line); lines.append(line + "\n")

    os.makedirs("results", exist_ok=True)
    with open("results/background_target.txt", "w") as f:
        f.writelines(lines)

    # ---- figure ---------------------------------------------------------------
    ncol = 3 if sweep else 2
    fig, ax = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5))
    for name, rows in fig_rows.items():
        e = [r[0] for r in rows]
        ax[0].plot(e, [r[1]["p_dog"] for r in rows], "o-", label=name)
        ax[1].plot(e, [r[1]["cls_to_removed"] for r in rows], "o-", label=name)
    ax[0].axhline(base_pdog, ls=":", c="gray"); ax[0].axhline(rem_pdog, ls="--", c="k", label="honest removal")
    ax[0].set_title(f'dense P("{obj_name}") in mask'); ax[0].set_xlabel("L_inf (/255)")
    ax[1].set_title("CLS cosine to real removed scene"); ax[1].set_xlabel("L_inf (/255)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    if sweep:
        k = [s[0] for s in sweep]
        ax[2].plot(k, [s[1]["p_dog"] for s in sweep], "o-", label=f"P({obj_name})")
        ax[2].plot(k, [s[1]["cls_to_removed"] for s in sweep], "s-", label="CLS->removed")
        ax[2].plot(k, [s[1]["cos_centroid_removed"] for s in sweep], "^--", label="cos(centroid,removed)")
        ax[2].set_xscale("log", base=2); ax[2].set_xlabel("# couch anchors")
        ax[2].set_title("genericity/specificity tradeoff"); ax[2].legend(fontsize=8)
    fig.suptitle("generation-free background targets: attention-ablation & generic anchors", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/background_target.png", dpi=130)
    print("\nsaved results/background_target.{txt,png}")


if __name__ == "__main__":
    main()
