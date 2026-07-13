"""Stage 1: ground-truth object mask via SAM 3 (text-prompted segmentation).

Example:
    uv run python generate_mask.py --image original.png --prompt dog --out masks/dog_mask.png
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="original.png")
    ap.add_argument("--prompt", default="dog")
    ap.add_argument("--out", default="masks/dog_mask.png")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")

    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks, scores = results["masks"], results["scores"]
    if len(masks) == 0:
        raise SystemExit(f"SAM3 found no '{args.prompt}' above threshold {args.threshold}")

    print(f"{len(masks)} instance(s) of '{args.prompt}', scores: "
          f"{[round(float(s), 3) for s in scores]}")

    # union of all detected instances
    union = torch.zeros_like(masks[0], dtype=torch.bool)
    for m in masks:
        union |= m.bool()
    mask_np = (union.cpu().numpy().astype(np.uint8)) * 255

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(mask_np, mode="L").save(args.out)
    cov = (mask_np > 0).mean()
    print(f"saved {args.out}  ({image.size[0]}x{image.size[1]}, {cov:.1%} of image)")


if __name__ == "__main__":
    main()
