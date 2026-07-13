"""CLIP patch-level analysis of object removal.

Embeds original.png (corgi on couch) and removed.png (dog inpainted away),
localizes the dog in the original via text->patch similarity, then measures
how much each spatial patch embedding moved after removal.

Outputs: results/patch_analysis.png, results/stats.txt
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch16"
RES = 448  # input resolution; pos-embeddings are interpolated to this size
PATCH = 16

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

orig_img = Image.open("original.png").convert("RGB")
rem_img = Image.open("removed.png").convert("RGB")


def preprocess(img, res):
    """Resize (squash) to res x res so the patch grid maps to the full frame."""
    inputs = processor(
        images=img.resize((res, res), Image.BICUBIC),
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return inputs["pixel_values"].to(DEVICE)


@torch.no_grad()
def embed_patches(pixel_values):
    """Return (grid, D) patch embeddings projected into CLIP joint space, plus CLS."""
    out = model.vision_model(pixel_values, interpolate_pos_encoding=True)
    hidden = out.last_hidden_state  # (1, 1+N, 768)
    hidden = model.vision_model.post_layernorm(hidden)
    proj = model.visual_projection(hidden)  # (1, 1+N, 512)
    cls_emb = F.normalize(proj[:, 0], dim=-1)
    patches = F.normalize(proj[:, 1:], dim=-1)
    return cls_emb.squeeze(0), patches.squeeze(0)


@torch.no_grad()
def embed_texts(prompts):
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    feats = model.get_text_features(**tok)
    return F.normalize(feats, dim=-1)


grid = RES // PATCH  # 28

px_o = preprocess(orig_img, RES)
px_r = preprocess(rem_img, RES)

# --- pixel-level alignment sanity check -------------------------------------
mae = (px_o - px_r).abs().mean().item()

cls_o, patch_o = embed_patches(px_o)
cls_r, patch_r = embed_patches(px_r)

# --- global (CLS) comparison -------------------------------------------------
global_sim = (cls_o @ cls_r).item()

text_prompts = [
    "a photo of a dog",
    "a dog on a couch",
    "an empty couch",
    "a photo of a couch",
    "an orange wall",
    "a window",
]
txt = embed_texts(text_prompts)
sims_o = (cls_o @ txt.T).cpu().numpy()
sims_r = (cls_r @ txt.T).cpu().numpy()

# --- dog localization on original via patch->text similarity -----------------
# softmax over a prompt set per patch gives a cleaner map than raw cosine
loc_prompts = ["a photo of a dog", "a photo of a couch",
               "a photo of a wall", "a photo of a window"]
loc_txt = embed_texts(loc_prompts)
logits_o = (patch_o @ loc_txt.T) * 100.0  # CLIP logit scale ~100
probs_o = logits_o.softmax(dim=-1)[:, 0].reshape(grid, grid).cpu().numpy()
dog_cos = (patch_o @ loc_txt[0]).reshape(grid, grid).cpu().numpy()

dog_mask = probs_o > 0.5  # patches classified as dog

# --- per-patch drift between original and removed -----------------------------
patch_sim = (patch_o * patch_r).sum(-1).reshape(grid, grid).cpu().numpy()
drift = 1.0 - patch_sim

drift_dog = drift[dog_mask].mean()
drift_bg = drift[~dog_mask].mean()

# where did the removed image's dog-region patches go semantically?
patch_r_grid = patch_r.reshape(grid, grid, -1)
dog_patches_after = patch_r_grid[torch.from_numpy(dog_mask).to(DEVICE)]
after_sims = (dog_patches_after @ loc_txt.T).mean(0).cpu().numpy()
before_sims = (patch_o.reshape(grid, grid, -1)[torch.from_numpy(dog_mask).to(DEVICE)]
               @ loc_txt.T).mean(0).cpu().numpy()

# residual dog signal in removed image
logits_r = (patch_r @ loc_txt.T) * 100.0
probs_r = logits_r.softmax(dim=-1)[:, 0].reshape(grid, grid).cpu().numpy()

# --- report -------------------------------------------------------------------
lines = []
lines.append(f"model={MODEL_ID}  input={RES}x{RES}  grid={grid}x{grid} ({grid*grid} patches)")
lines.append(f"pixel MAE between preprocessed images: {mae:.4f} (normalized units)")
lines.append(f"\nGlobal CLS cosine(original, removed) = {global_sim:.4f}")
lines.append("\nGlobal image-text cosine similarities:")
lines.append(f"{'prompt':<24}{'original':>10}{'removed':>10}{'delta':>10}")
for p, so, sr in zip(text_prompts, sims_o, sims_r):
    lines.append(f"{p:<24}{so:>10.4f}{sr:>10.4f}{sr-so:>+10.4f}")
lines.append(f"\nDog patches (softmax>0.5): {dog_mask.sum()} / {grid*grid}")
lines.append(f"mean patch drift (1-cos) inside dog region:  {drift_dog:.4f}")
lines.append(f"mean patch drift (1-cos) outside dog region: {drift_bg:.4f}")
lines.append(f"ratio: {drift_dog/drift_bg:.2f}x")
lines.append("\nMean text-similarity of DOG-REGION patches (before -> after removal):")
for i, p in enumerate(loc_prompts):
    lines.append(f"  {p:<24}{before_sims[i]:.4f} -> {after_sims[i]:.4f}")
lines.append(f"\nmax dog prob in removed image: {probs_r.max():.4f} "
             f"(original: {probs_o.max():.4f})")

report = "\n".join(lines)
print(report)

import os
os.makedirs("results", exist_ok=True)
with open("results/stats.txt", "w") as f:
    f.write(report + "\n")

# --- figure -------------------------------------------------------------------
def show(ax, img):
    ax.imshow(img.resize((RES, RES)))
    ax.set_xticks([]); ax.set_yticks([])

def overlay(ax, img, heat, title, cmap="jet", vmin=None, vmax=None):
    show(ax, img)
    hm = ax.imshow(np.kron(heat, np.ones((PATCH, PATCH))), cmap=cmap,
                   alpha=0.55, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    return hm

fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

show(axes[0, 0], orig_img); axes[0, 0].set_title("original", fontsize=11)
show(axes[0, 1], rem_img); axes[0, 1].set_title("removed (inpainted)", fontsize=11)

hm = overlay(axes[0, 2], orig_img, probs_o, 'P("dog") per patch — original', vmin=0, vmax=1)
fig.colorbar(hm, ax=axes[0, 2], fraction=0.03)

hm = overlay(axes[1, 0], rem_img, probs_r, 'P("dog") per patch — removed', vmin=0, vmax=1)
fig.colorbar(hm, ax=axes[1, 0], fraction=0.03)

hm = overlay(axes[1, 1], orig_img, drift, "patch drift 1−cos(orig, removed)",
             cmap="inferno", vmin=0)
fig.colorbar(hm, ax=axes[1, 1], fraction=0.03)

ax = axes[1, 2]
ax.scatter(probs_o.flatten(), drift.flatten(), s=12, alpha=0.5, c="tab:blue")
ax.set_xlabel('P("dog") in original patch')
ax.set_ylabel("patch drift (1−cos)")
ax.set_title("dog-ness vs. embedding drift", fontsize=11)

fig.suptitle("CLIP ViT-B/16 patch-level effect of object removal", fontsize=13)
fig.tight_layout()
fig.savefig("results/patch_analysis.png", dpi=130)
print("\nsaved results/patch_analysis.png")
