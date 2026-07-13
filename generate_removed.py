"""Stage 2: pixel-aligned object removal via SDXL inpainting.

Only pixels inside the (dilated) mask are ever touched: the diffusion output is
composited back into the original at full resolution, so the pair is aligned by
construction. Also supports --control, which shifts the object mask onto a
background region and inpaints there — a baseline for "drift caused by
inpainting texture alone".

Examples:
    uv run python generate_removed.py --image original.png --mask masks/dog_mask.png \
        --out removed_aligned.png --prompt "an empty green couch in a living room"
    uv run python generate_removed.py --image original.png --mask masks/dog_mask.png \
        --out control_inpaint.png --control --control-mask-out masks/control_mask.png
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from diffusers import AutoPipelineForInpainting

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
WORK = 1024  # SDXL native working resolution


def square_crop_around_mask(mask, margin=1.5):
    """Square crop box (l, t, r, b) around mask bbox, clipped to the image."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    t, b, l, r = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (t + b) / 2, (l + r) / 2
    side = int(min(max(b - t, r - l) * margin, min(H, W)))
    half = side // 2
    cx = int(np.clip(cx, half, W - half))
    cy = int(np.clip(cy, half, H - half))
    return cx - half, cy - half, cx + half, cy + half


def shift_mask_to_background(mask):
    """Translate the object mask to a non-overlapping background spot (control)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    h_obj, w_obj = ys.max() - ys.min(), xs.max() - xs.min()
    for dx, dy in [(int(W * 0.3), -int(H * 0.15)), (-int(W * 0.3), -int(H * 0.15)),
                   (int(W * 0.35), 0), (-int(W * 0.35), 0)]:
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        if ok.mean() < 0.98:
            continue
        shifted = np.zeros_like(mask)
        shifted[ny[ok], nx[ok]] = True
        if not (shifted & mask).any():
            return shifted
    raise SystemExit("could not place control mask without overlapping the object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="original.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--out", default="removed_aligned.png")
    ap.add_argument("--prompt", default="an empty couch in a living room, fabric cushions")
    ap.add_argument("--negative", default="dog, animal, pet, person, object")
    ap.add_argument("--dilate", type=int, default=24, help="mask dilation in px (covers shadows/fur edges)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="inpaint a shifted copy of the mask on background instead")
    ap.add_argument("--control-mask-out", default="masks/control_mask.png")
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    W, H = image.size
    mask = np.array(Image.open(args.mask).convert("L")) > 127

    if args.control:
        mask = shift_mask_to_background(mask)
        os.makedirs(os.path.dirname(args.control_mask_out) or ".", exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.control_mask_out)
        print(f"saved control mask -> {args.control_mask_out}")

    mask_dil = binary_dilation(mask, iterations=args.dilate)

    l, t, r, b = square_crop_around_mask(mask_dil)
    crop_img = image.crop((l, t, r, b)).resize((WORK, WORK), Image.LANCZOS)
    crop_mask = Image.fromarray(mask_dil.astype(np.uint8) * 255).crop((l, t, r, b)) \
                     .resize((WORK, WORK), Image.NEAREST)

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, variant="fp16" if DEVICE == "cuda" else None
    ).to(DEVICE)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image=crop_img,
        mask_image=crop_mask,
        height=WORK,
        width=WORK,
        strength=0.99,
        guidance_scale=8.0,
        num_inference_steps=args.steps,
        generator=gen,
    ).images[0]

    # composite back: only dilated-mask pixels change, everything else is the
    # original image byte-for-byte -> the pair is pixel-aligned by construction
    result_full = result.resize((r - l, b - t), Image.LANCZOS)
    out_np = np.array(image)
    crop_np = np.array(result_full)
    region = mask_dil[t:b, l:r]
    out_np[t:b, l:r][region] = crop_np[region]

    Image.fromarray(out_np).save(args.out)
    changed = mask_dil.mean()
    print(f"saved {args.out}  ({W}x{H}, {changed:.1%} of pixels replaced)")


if __name__ == "__main__":
    main()
