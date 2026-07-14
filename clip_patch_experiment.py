"""Stage 3: CLIP patch-level analysis of a pixel-aligned edit, with GT mask.

Measures, for an (original, edited, mask) triple:
  * global CLS embedding shift and image-text similarity deltas
  * per-patch embedding drift, split into object / boundary-ring / background
  * drift as a function of grid distance from the mask (context bleed)
  * residual object signal in the edited image (MaskCLIP-style dense features)

Dense localization uses the MaskCLIP value-projection trick: the last attention
block's softmax mixing is replaced by each token's own value vector, which gives
spatially faithful patch features (naive last-layer tokens do not localize).

Examples:
    uv run python clip_patch_experiment.py --edited removed_aligned.png --tag removal
    uv run python clip_patch_experiment.py --edited control_inpaint.png \
        --mask masks/control_mask.png --tag control
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
GRID = RES // PATCH  # 28

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)


def preprocess(img):
    """Squash-resize to RES x RES so the patch grid maps to the full frame."""
    inputs = processor(
        images=img.resize((RES, RES), Image.BICUBIC),
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return inputs["pixel_values"].to(DEVICE)


@torch.no_grad()
def embed(pixel_values):
    """Returns (cls, patches, dense) — all L2-normalized, in CLIP joint space.

    patches: standard last-layer tokens (what downstream models consume)
    dense:   MaskCLIP value-projection features (spatially faithful)
    """
    vm = model.vision_model
    out = vm(pixel_values, interpolate_pos_encoding=True, output_hidden_states=True)

    hidden = vm.post_layernorm(out.last_hidden_state)
    proj = model.visual_projection(hidden)
    cls_emb = F.normalize(proj[:, 0], dim=-1).squeeze(0)
    patches = F.normalize(proj[:, 1:], dim=-1).squeeze(0)

    # MaskCLIP trick: recompute the last encoder layer with value-only attention
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
    # explicit text tower + projection: get_text_features returns a tensor in
    # transformers v4 but a model-output object in v5
    pooled = model.text_model(**tok).pooler_output
    return F.normalize(model.text_projection(pooled), dim=-1)


def mask_to_grid(mask_img):
    """Full-res binary mask -> per-patch object fraction on the GRID x GRID grid."""
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), dtype=np.float32) / 255.0
    return m.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--edited", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog", help="text name of the edited object")
    ap.add_argument("--tag", default="removal", help="suffix for output files")
    args = ap.parse_args()

    orig_img = Image.open(args.original).convert("RGB")
    edit_img = Image.open(args.edited).convert("RGB")
    assert orig_img.size == edit_img.size, "pair must be pixel-aligned (same size)"
    mask_img = Image.open(args.mask).convert("L")

    px_o, px_e = preprocess(orig_img), preprocess(edit_img)
    mae = (px_o - px_e).abs().mean().item()

    cls_o, patch_o, dense_o = embed(px_o)
    cls_e, patch_e, dense_e = embed(px_e)

    # --- regions from the GT mask -------------------------------------------
    frac = mask_to_grid(mask_img)
    obj = frac > 0.5
    dist = distance_transform_edt(~obj)          # grid cells from the object
    ring = (~obj) & (dist <= 2)                  # boundary ring (<=2 patches out)
    far = (~obj) & (dist > 2)

    # --- global comparison ----------------------------------------------------
    global_sim = (cls_o @ cls_e).item()
    obj_name = args.object
    text_prompts = [f"a photo of a {obj_name}", f"a {obj_name} on a couch",
                    "an empty couch", "a photo of a couch"]
    txt = embed_texts(text_prompts)
    sims_o = (cls_o @ txt.T).cpu().numpy()
    sims_e = (cls_e @ txt.T).cpu().numpy()

    # --- per-patch drift (standard tokens, what downstream tasks see) --------
    drift = (1.0 - (patch_o * patch_e).sum(-1)).reshape(GRID, GRID).cpu().numpy()
    drift_dense = (1.0 - (dense_o * dense_e).sum(-1)).reshape(GRID, GRID).cpu().numpy()

    # drift vs distance from mask (context bleed on the standard tokens)
    max_d = int(dist.max())
    bleed = [(d, drift[(dist > d - 1) & (dist <= d)].mean(),
              ((dist > d - 1) & (dist <= d)).sum()) for d in range(1, max_d + 1)]

    # --- residual object signal via dense features ----------------------------
    loc_prompts = [f"a photo of a {obj_name}", "a photo of a couch",
                   "a photo of a wall", "a photo of a window"]
    loc_txt = embed_texts(loc_prompts)
    prob_o = ((dense_o @ loc_txt.T) * 100).softmax(-1)[:, 0].reshape(GRID, GRID).cpu().numpy()
    prob_e = ((dense_e @ loc_txt.T) * 100).softmax(-1)[:, 0].reshape(GRID, GRID).cpu().numpy()

    # sanity: does the dense map agree with the GT mask? (IoU at 0.5)
    pred = prob_o > 0.5
    iou = (pred & obj).sum() / max((pred | obj).sum(), 1)

    # --- report ----------------------------------------------------------------
    L = []
    L.append(f"model={MODEL_ID} input={RES} grid={GRID}x{GRID}  tag={args.tag}")
    L.append(f"pair: {args.original} vs {args.edited}  mask: {args.mask}")
    L.append(f"pixel MAE (preprocessed): {mae:.4f}")
    L.append(f"\nGlobal CLS cosine(original, edited) = {global_sim:.4f}")
    L.append(f"{'prompt':<28}{'original':>10}{'edited':>10}{'delta':>10}")
    for p, so, se in zip(text_prompts, sims_o, sims_e):
        L.append(f"{p:<28}{so:>10.4f}{se:>10.4f}{se-so:>+10.4f}")
    L.append(f"\nGT mask: {obj.sum()} object patches, {ring.sum()} ring, {far.sum()} far bg"
             f"  (dense-map IoU vs GT: {iou:.3f})")
    L.append("mean patch drift (1-cos), standard tokens:")
    L.append(f"  object region:   {drift[obj].mean():.4f}")
    L.append(f"  boundary ring:   {drift[ring].mean():.4f}")
    L.append(f"  far background:  {drift[far].mean():.4f}")
    L.append(f"  object/far ratio: {drift[obj].mean()/max(drift[far].mean(),1e-6):.2f}x")
    L.append("mean patch drift (1-cos), dense (MaskCLIP) features:")
    L.append(f"  object region:   {drift_dense[obj].mean():.4f}")
    L.append(f"  boundary ring:   {drift_dense[ring].mean():.4f}")
    L.append(f"  far background:  {drift_dense[far].mean():.4f}")
    L.append("\ndrift vs grid-distance from mask (standard tokens):")
    for d, v, n in bleed:
        L.append(f"  d={d:>2}: {v:.4f}  (n={n})")
    L.append(f"\nresidual P({obj_name}) inside mask, dense features: "
             f"orig mean={prob_o[obj].mean():.4f} max={prob_o[obj].max():.4f} -> "
             f"edited mean={prob_e[obj].mean():.4f} max={prob_e[obj].max():.4f}")

    report = "\n".join(L)
    print(report)
    os.makedirs("results", exist_ok=True)
    with open(f"results/stats_{args.tag}.txt", "w") as f:
        f.write(report + "\n")

    # --- figure ------------------------------------------------------------------
    def up(a):
        return np.kron(a, np.ones((PATCH, PATCH)))

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    panels = [
        (axes[0, 0], orig_img, None, "original", {}),
        (axes[0, 1], edit_img, None, f"edited ({args.tag})", {}),
        (axes[0, 2], orig_img, frac, "GT mask (patch fraction)", dict(cmap="viridis", vmin=0, vmax=1)),
        (axes[0, 3], orig_img, prob_o, f'dense P("{obj_name}") — original', dict(cmap="jet", vmin=0, vmax=1)),
        (axes[1, 0], edit_img, prob_e, f'dense P("{obj_name}") — edited', dict(cmap="jet", vmin=0, vmax=1)),
        (axes[1, 1], orig_img, drift, "patch drift — standard tokens", dict(cmap="inferno", vmin=0)),
        (axes[1, 2], orig_img, drift_dense, "patch drift — dense features", dict(cmap="inferno", vmin=0)),
    ]
    for ax, img, heat, title, kw in panels:
        ax.imshow(img.resize((RES, RES)))
        if heat is not None:
            hm = ax.imshow(up(heat), alpha=0.55, **kw)
            fig.colorbar(hm, ax=ax, fraction=0.03)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 3]
    ds = [d for d, _, _ in bleed]
    vs = [v for _, v, _ in bleed]
    ax.plot(ds, vs, "o-", color="tab:red", label="background")
    ax.axhline(drift[obj].mean(), ls="--", color="tab:blue", label="object region")
    ax.set_xlabel("grid distance from mask")
    ax.set_ylabel("mean drift (1−cos)")
    ax.set_title("context bleed: drift vs distance", fontsize=11)
    ax.legend()

    fig.suptitle(f"CLIP patch-level effect of edit — {args.tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"results/patch_analysis_{args.tag}.png", dpi=130)
    print(f"\nsaved results/patch_analysis_{args.tag}.png")


if __name__ == "__main__":
    main()
