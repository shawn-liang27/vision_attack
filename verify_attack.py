"""Stage 9b: does the attack survive save->reload on the SAME CLIP?

Before blaming transfer/surrogates, check the most basic thing: reload each
saved adversarial PNG, run it back through the attacked encoder (ViT-B/16) with
the exact attack preprocessing, and measure P(dog) in the mask. If P(dog) has
snapped back toward baseline, the save (delta upsampled to full-res) + any
re-resize destroyed the high-frequency perturbation -- i.e. the failure is
pipeline fragility, not lack of surrogates. Also reports P(dog) when the image
is first downsampled to 336 (LLaVA) / 224 to show resolution sensitivity.

    uv run python verify_attack.py
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
RES, PATCH, GRID = 448, 16, 28

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
ip = processor.image_processor
MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def dense(img, pre_resize=None):
    """dense (MaskCLIP) patch features. pre_resize simulates a VLM first
    downsampling the image to a smaller resolution before it reaches us."""
    if pre_resize:
        img = img.resize((pre_resize, pre_resize), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255) \
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    vm = model.vision_model
    out = vm((x - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
    layer = vm.encoder.layers[-1]
    h = out.hidden_states[-2]
    a = layer.layer_norm1(h)
    h = h + layer.self_attn.out_proj(layer.self_attn.v_proj(a))
    h = h + layer.mlp(layer.layer_norm2(h))
    d = model.visual_projection(vm.post_layernorm(h))
    return F.normalize(d[:, 1:], dim=-1).squeeze(0)


@torch.no_grad()
def texts(prompts):
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)


meas = texts(["a corgi", "a puppy", "a dog sitting on a sofa",
              "an empty sofa", "a cushion", "living room furniture", "a bare couch"])
nd = 3


def p_dog(img, pre_resize=None):
    logits = (dense(img, pre_resize) @ meas.T) * 100
    return logits.softmax(-1)[:, :nd].sum(-1)


mask = np.array(Image.open("masks/dog_mask.png").convert("L").resize((RES, RES), Image.BILINEAR),
                np.float32) / 255
obj = (mask.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5).reshape(-1)
obj_t = torch.from_numpy(obj).to(DEVICE)

paths = sorted(glob.glob("results/adv_images/*.png"))
print(f"{'image':<30}{'P(dog)@448':>12}{'@336':>8}{'@224':>8}   (mean in mask)")
lines = ["image,p_dog_448,p_dog_336,p_dog_224,max_448"]
for p in paths:
    img = Image.open(p).convert("RGB")
    pv = p_dog(img)
    p336 = p_dog(img, 336)[obj_t].mean().item()
    p224 = p_dog(img, 224)[obj_t].mean().item()
    m, mx = pv[obj_t].mean().item(), pv[obj_t].max().item()
    print(f"{os.path.basename(p):<30}{m:>12.3f}{p336:>8.3f}{p224:>8.3f}   (max {mx:.3f})")
    lines.append(f"{os.path.basename(p)},{m:.4f},{p336:.4f},{p224:.4f},{mx:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/verify_attack.csv", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nIf adv rows ~= baseline row, the save/reload+resize destroyed the attack.")
print("saved results/verify_attack.csv")
