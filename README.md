# vision_attack

Experiments on how object removal changes CLIP embeddings, at the patch level.

A first qualitative pair (`original.png` = corgi on a couch, `removed.png` = tool-generated
removal) showed a clean **global** semantic shift (dog→couch in CLIP text-similarity space),
but the patch-level comparison was confounded: the removal tool also re-framed the image
(pixel MAE 0.21), and naive CLIP last-layer patch tokens don't localize objects. This
pipeline removes both confounds.

## Pipeline

Controlled triple: ground-truth mask → pixel-aligned removal → mask-aware patch analysis.

```bash
uv sync  # needs a GPU box; SDXL inpainting + SAM3 + CLIP

# 1. GT object mask from SAM 3 (text-prompted)
uv run python generate_mask.py --image original.png --prompt dog --out masks/dog_mask.png

# 2a. pixel-aligned removal (SDXL inpainting, composited back — only masked px change)
uv run python generate_removed.py --image original.png --mask masks/dog_mask.png \
    --out removed_aligned.png --prompt "an empty couch in a living room, fabric cushions"

# 2a-alt. if SDXL hallucinates an object in the hole (it's a replacement, not a removal),
# composite an externally-edited image (e.g. Gemini) through the mask instead —
# alignment stays guaranteed because only masked pixels are taken from the edit
uv run python generate_removed.py --image original.png --mask masks/dog_mask.png \
    --out removed_aligned.png --external gemini_removed.png

# 2b. control: inpaint an equal-shape background region (drift from inpainting alone)
uv run python generate_removed.py --image original.png --mask masks/dog_mask.png \
    --out control_inpaint.png --control

# 3. CLIP patch analysis (removal condition + control condition)
uv run python clip_patch_experiment.py --edited removed_aligned.png --tag removal
uv run python clip_patch_experiment.py --edited control_inpaint.png \
    --mask masks/control_mask.png --tag control

# 4. interpolation sweep: scale the removal delta like an adversarial budget,
#    x_t = original + t*(removed - original), and track patch tokens vs t
uv run python interpolation_sweep.py --tag removal

# 5. white-box PGD on CLIP vs the interpolation oracle at matched L_inf:
#    is ~150/255 a property of CLIP geometry or just the unoptimized direction?
uv run python pgd_attack.py --objective target_img --budgets 2,4,8,16,32,64,128 --iters 300
uv run python pgd_attack.py --objective suppress    # untargeted dog suppression

# 6. is the fill THIS specific background, and recoverable from context alone?
#    steers masked patches toward the surrounding visible couch ring (no
#    regenerated pixels), vs generic couch text, vs the real removed bg
uv run python context_completion.py --budgets 4,8,16 --iters 300
```

Outputs land in `results/`: `stats_<tag>.txt` and `patch_analysis_<tag>.png`.

## What stage 3 measures

- **Global**: CLS cosine between original/edited; image-text similarity deltas
  ("a photo of a dog" vs "an empty couch", …).
- **Patch drift** (1 − cos per grid cell), split by the GT mask into object /
  boundary ring / far background — for both the standard last-layer tokens and
  MaskCLIP-style dense features (value-projection of the last attention block,
  which localizes; naive tokens don't).
- **Context bleed**: drift as a function of grid distance from the mask.
- **Residual object signal**: dense P("dog") inside the mask before vs after removal.

Comparing the `removal` run against the `control` run separates *object-semantics*
drift from generic *inpainting-texture* drift.

## Notes

- Models: `openai/clip-vit-base-patch16` (448px, interpolated pos-embeddings, 28×28 grid),
  `facebook/sam3`, `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.
- `removed.png` (the original tool-generated edit) is kept for reference; the analysis
  should use `removed_aligned.png`, which is aligned by construction.
