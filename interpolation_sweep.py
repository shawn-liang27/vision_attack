"""Stage 4: interpolation sweep — steer original patch tokens toward the
removed-image tokens by scaling the removal delta, adversarial-attack style.

x_t = quantize(original + t * (removed - original)),  t in [0, 1]

The delta is nonzero only inside the (dilated) object mask, so this is a
purely local perturbation ramp whose L_inf budget scales linearly with t —
an oracle version of the steering direction a black-box attack (e.g.
M-Attack-style token alignment) has to *search* for under budget.

Tracked per t:
  * dense (MaskCLIP) P("dog") inside the mask — mean and max
  * global CLS image-text similarity for dog / couch prompts
  * steering progress: mean cos of mask patch tokens to their t=0 and t=1 values
  * regional patch drift vs t=0 (object / ring / far background)
  * perturbation budget: L_inf and L2 of the pixel delta

Outputs: results/sweep_<tag>.csv, results/sweep_<tag>.png

Example:
    uv run python interpolation_sweep.py --tag removal
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
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
processor = CLIPProcessor.from_pretrained(MODEL_ID)


def preprocess(img):
    inputs = processor(
        images=img.resize((RES, RES), Image.BICUBIC),
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return inputs["pixel_values"].to(DEVICE)


@torch.no_grad()
def embed(pixel_values):
    """(cls, standard patch tokens, dense MaskCLIP tokens), all L2-normalized."""
    vm = model.vision_model
    out = vm(pixel_values, interpolate_pos_encoding=True, output_hidden_states=True)
    hidden = vm.post_layernorm(out.last_hidden_state)
    proj = model.visual_projection(hidden)
    cls_emb = F.normalize(proj[:, 0], dim=-1).squeeze(0)
    patches = F.normalize(proj[:, 1:], dim=-1).squeeze(0)

    layer = vm.encoder.layers[-1]
    h = out.hidden_states[-2]
    x = layer.layer_norm1(h)
    x = layer.self_attn.out_proj(layer.self_attn.v_proj(x))
    h = h + x
    h = h + layer.mlp(layer.layer_norm2(h))
    dense = model.visual_projection(vm.post_layernorm(h))
    dense = F.normalize(dense[:, 1:], dim=-1).squeeze(0)
    return cls_emb, patches, dense


@torch.no_grad()
def embed_texts(prompts):
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    pooled = model.text_model(**tok).pooler_output
    return F.normalize(model.text_projection(pooled), dim=-1)


def mask_to_grid(mask_img):
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), dtype=np.float32) / 255.0
    return m.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--edited", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--tag", default="removal")
    ap.add_argument("--steps", type=int, default=21, help="linear grid size on [0,1]")
    args = ap.parse_args()

    orig_img = Image.open(args.original).convert("RGB")
    edit_img = Image.open(args.edited).convert("RGB")
    assert orig_img.size == edit_img.size
    mask_img = Image.open(args.mask).convert("L")

    orig_np = np.array(orig_img, dtype=np.float32)
    edit_np = np.array(edit_img, dtype=np.float32)
    delta = edit_np - orig_np
    print(f"delta: {np.any(delta != 0, axis=2).mean():.1%} of pixels nonzero, "
          f"max|delta|={np.abs(delta).max():.0f}/255")

    # linear grid plus dense small-t samples for the adversarial-budget regime
    ts = sorted(set(np.round(np.linspace(0, 1, args.steps), 4))
                | {0.01, 0.02, 0.03, 0.05, 0.08, 0.12})

    frac = mask_to_grid(mask_img)
    obj = frac > 0.5
    dist = distance_transform_edt(~obj)
    ring = (~obj) & (dist <= 2)
    far = (~obj) & (dist > 2)
    obj_t = torch.from_numpy(obj.reshape(-1)).to(DEVICE)

    obj_name = args.object
    loc_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch",
                           "a photo of a wall", "a photo of a window"])
    cls_txt = embed_texts([f"a photo of a {obj_name}", "a photo of a couch"])

    # endpoints for steering-progress measurement
    _, patch_0, _ = embed(preprocess(orig_img))
    _, patch_1, _ = embed(preprocess(edit_img))

    rows = []
    frames = {}
    for t in ts:
        x_t = Image.fromarray(np.clip(orig_np + t * delta, 0, 255).astype(np.uint8))
        cls_e, patch_e, dense_e = embed(preprocess(x_t))

        prob = ((dense_e @ loc_txt.T) * 100).softmax(-1)[:, 0]
        p_obj = prob[obj_t]
        drift = (1.0 - (patch_0 * patch_e).sum(-1)).reshape(GRID, GRID).cpu().numpy()
        sims = (cls_e @ cls_txt.T).cpu().numpy()

        row = dict(
            t=t,
            linf=t * np.abs(delta).max() / 255.0,
            l2=t * np.linalg.norm(delta / 255.0),
            p_obj_mean=p_obj.mean().item(),
            p_obj_max=p_obj.max().item(),
            cls_sim_obj=float(sims[0]),
            cls_sim_couch=float(sims[1]),
            cos_to_orig=(patch_e * patch_0).sum(-1)[obj_t].mean().item(),
            cos_to_removed=(patch_e * patch_1).sum(-1)[obj_t].mean().item(),
            drift_obj=drift[obj].mean(),
            drift_ring=drift[ring].mean(),
            drift_far=drift[far].mean(),
        )
        rows.append(row)
        print(f"t={t:.3f}  P({obj_name})={row['p_obj_mean']:.3f}  "
              f"sim_{obj_name}={row['cls_sim_obj']:.4f}  far_drift={row['drift_far']:.4f}")

        if round(t, 2) in (0.0, 0.25, 0.5, 0.75, 1.0):
            heat = prob.reshape(GRID, GRID).cpu().numpy()
            frames[round(t, 2)] = (x_t, heat)

    os.makedirs("results", exist_ok=True)
    keys = list(rows[0].keys())
    with open(f"results/sweep_{args.tag}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" for k in keys) + "\n")

    # --- figure ---------------------------------------------------------------
    T = [r["t"] for r in rows]
    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9])

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(T, [r["p_obj_mean"] for r in rows], "o-", label="mean")
    ax.plot(T, [r["p_obj_max"] for r in rows], "s--", label="max")
    ax.set_xlabel("t"); ax.set_ylabel(f'dense P("{obj_name}") in mask')
    ax.set_title(f"{obj_name} signal vs perturbation scale"); ax.legend()

    ax = fig.add_subplot(gs[0, 2:4])
    ax.plot(T, [r["cls_sim_obj"] for r in rows], "o-", label=f'"a photo of a {obj_name}"')
    ax.plot(T, [r["cls_sim_couch"] for r in rows], "s-", label='"a photo of a couch"')
    ax.set_xlabel("t"); ax.set_ylabel("CLS image-text cos")
    ax.set_title("global semantics vs t"); ax.legend()

    ax = fig.add_subplot(gs[0, 4:6])
    ax.plot(T, [r["cos_to_orig"] for r in rows], "o-", label="cos to t=0 tokens")
    ax.plot(T, [r["cos_to_removed"] for r in rows], "s-", label="cos to t=1 tokens")
    ax.set_xlabel("t"); ax.set_ylabel("mean cos (mask patches)")
    ax.set_title("steering progress in token space"); ax.legend()

    ax = fig.add_subplot(gs[1, 0:2])
    for k, lbl in [("drift_obj", "object"), ("drift_ring", "ring"), ("drift_far", "far bg")]:
        ax.plot(T, [r[k] for r in rows], "o-", label=lbl)
    ax.set_xlabel("t"); ax.set_ylabel("mean drift vs t=0")
    ax.set_title("regional drift vs t"); ax.legend()

    ax = fig.add_subplot(gs[1, 2:4])
    eps = [r["linf"] * 255 for r in rows[1:]]
    ax.semilogx(eps, [r["p_obj_mean"] for r in rows[1:]], "o-")
    for e in (4, 8, 16):
        ax.axvline(e, ls=":", color="gray")
        ax.text(e, ax.get_ylim()[1] * 0.95, f" {e}/255", fontsize=8, color="gray")
    ax.set_xlabel("L_inf budget (/255, log)"); ax.set_ylabel(f'mean dense P("{obj_name}")')
    ax.set_title("adversarial-budget view")

    ax = fig.add_subplot(gs[1, 4:6])
    ax.semilogx(eps, [r["drift_far"] for r in rows[1:]], "o-", color="tab:red")
    ax.set_xlabel("L_inf budget (/255, log)"); ax.set_ylabel("far-background drift")
    ax.set_title("attention bleed vs budget")

    for i, (tv, (img, heat)) in enumerate(sorted(frames.items())):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(img.resize((RES, RES)))
        ax.imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap="jet",
                  alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"t={tv}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"steering original -> removed ({args.tag}): "
                 "patch tokens vs perturbation scale", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"results/sweep_{args.tag}.png", dpi=130)
    print(f"\nsaved results/sweep_{args.tag}.csv and results/sweep_{args.tag}.png")


if __name__ == "__main__":
    main()
