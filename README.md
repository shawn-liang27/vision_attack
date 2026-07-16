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

# 7. generation-free background targets:
#    Q1 attention-ablated object-free CLS (no patch attends to the object)
#    Q2 generic couch anchors, swept by set size (genericity vs specificity)
uv run python background_target.py --budgets 8,16 --iters 200 --anchors couch_anchors/

# 8. LOCAL patch-token camouflage (VLM-faithful) vs global CLS attack:
#    VLMs read the patch grid, not CLS -- attack the dog-region patch tokens
#    saves the manipulated images to results/adv_images/
uv run python pgd_patch.py --budgets 2,4,8,16 --iters 300

# 9. transfer test: feed the saved adv images to real ~7B VLMs, ask if the
#    dog is still visible (the non-circular judge). Gemini saw it in all.
uv run python vlm_eval.py   # Qwen2-VL-7B + LLaVA-1.6-7B by default

# 9b. why it failed: does the attack survive save/reload + resize on the SAME
#     CLIP? (it dies on resize -> resolution fragility, not surrogates)
uv run python verify_attack.py

# 10. correct method: resize-robust (EOT) white-box attack on the target VLM's
#     OWN encoder at its native resolution (LLaVA tower = CLIP ViT-L/14-336)
uv run python pgd_vlm_encoder.py --budgets 8,16 --iters 250 --eot 8
uv run python vlm_eval.py --images-dir results/adv_images_vlm \
    --models llava-hf/llava-v1.6-mistral-7b-hf

# 11. focused: generated-background image as the ONLY PGD anchor, with the
#     correct CLIP-space verification (loss-climb gate -> per-patch gate ->
#     zero-shot ROI object->background flip, per-patch AND pooled)
uv run python pgd_bg_anchor.py --budgets 4,8,16 --iters 300

# 12. V-Attack: steer VALUE features V (not output feats X) + dog-suppression,
#     three objectives compared on the same verification (V should drop P(dog)
#     where X did not; +suppress should collapse it)
uv run python pgd_v_attack.py --budgets 4,8,16 --iters 300 --beta 1.0

# 13. SAME-ENCODER VLM test (does CLIP-space concealment change the caption?):
#     regenerate V-attack on LLaVA-1.5's OWN encoder (clip-vit-large-patch14-336,
#     simple 336 resize, no AnyRes), then caption the EXACT 336 squares so the
#     VLM's resize doesn't resample the perturbation away.
uv run python pgd_v_attack.py --model openai/clip-vit-large-patch14-336 --res 336 \
    --budgets 4,8,16 --iters 300 --beta 1.0 --outdir results/v_attack_llava
uv run python vlm_eval.py --images-dir results/v_attack_llava/square \
    --models llava-hf/llava-1.5-7b-hf

# 14. representation-target fix: stage 13 showed CLIP-space concealment succeeds
#     on LLaVA's own encoder yet the caption still says dog -- because the
#     projector reads the RAW penultimate patch tokens, not CLIP's zero-shot
#     head. Steer THAT representation instead.
uv run python pgd_projector_target.py --budgets 8,16,32 --iters 300
uv run python vlm_eval.py --images-dir results/projector_target/square \
    --models llava-hf/llava-1.5-7b-hf

# 15. capability probe: V-Attack mechanism (value features, PGD/L_inf, single
#     model, NO surrogates) with a TEXT target "cat" instead of background.
#     dog->cat is far easier than dog->absent; does ANY feature attack move LLaVA?
uv run python v_attack_cat.py --budgets 8,16,32,64,128 --iters 300
uv run python vlm_eval.py --images-dir results/v_attack_cat/square \
    --object cat --models llava-hf/llava-1.5-7b-hf

# 16. M-Attack (V1) verification: whole-image, crop-matched embedding steering
#     original -> removed.png, single surrogate = LLaVA-1.5's own encoder.
#     Does whole-image crop-matching (more powerful than masked nudges) flip LLaVA?
uv run python m_attack.py --steps 300 --eps 16 --alpha 1 --attack mifgsm
uv run python vlm_eval.py --images-dir results/m_attack \
    --object dog --models llava-hf/llava-1.5-7b-hf

# 17. region-level M-Attack PADDING SWEEP (dose-response): confine the crops +
#     perturbation to the ROI expanded by a margin; sweep tight->whole-image to
#     find the transition where LLaVA stops seeing the dog.
uv run python m_attack_roi.py --pads 0,0.05,0.15,0.35,1.0 --steps 300 --eps 16
uv run python vlm_eval.py --images-dir results/m_attack_roi \
    --object dog --models llava-hf/llava-1.5-7b-hf

# 18. H1 test: whole-image crop-matching LOSS (global signal, matches what LLaVA
#     reads) but ROI-masked UPDATE (perturbation only in ROI). "Mask the update,
#     not the loss." Padding sweep; compare to #17 (local signal) at same pads.
uv run python m_attack_h1.py --pads 0,0.05,0.15,0.35,1.0 --steps 300 --eps 16
uv run python vlm_eval.py --images-dir results/m_attack_h1 \
    --object dog --models llava-hf/llava-1.5-7b-hf

# 19. SEEDED sweep (rigor): N seeds/padding -> success RATE, continuous signals
#     (ROI P(dog), cos-to-target), and per-pixel perturbation density. Run both
#     modes (h1 global-signal, local local-signal) for the seed-matched compare.
uv run python m_attack_sweep.py --mode h1    --seeds 5 --steps 300 --eps 16
uv run python m_attack_sweep.py --mode local --seeds 5 --steps 300 --eps 16
uv run python vlm_eval.py --images-dir results/sweep_h1 --object dog \
    --models llava-hf/llava-1.5-7b-hf
uv run python vlm_eval.py --images-dir results/sweep_local --object dog \
    --models llava-hf/llava-1.5-7b-hf
uv run python summarize_sweep.py --dir results/sweep_h1
uv run python summarize_sweep.py --dir results/sweep_local

# 20. unified single loss (Option 2): M-Attack crop-steering + per-crop ROI
#     suppression on the SAME crop forward pass; logs whether the two terms
#     cooperate or fight (cosine of their gradients on delta).
uv run python m_attack_combined.py --lam-steer 1 --lam-supp 1 --steps 300 --eps 16
uv run python vlm_eval.py --images-dir results/m_attack_combined \
    --object dog --models llava-hf/llava-1.5-7b-hf
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
