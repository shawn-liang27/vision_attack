"""Stage 10: resize-robust camouflage on the target VLM's own encoder.

Diagnosis from verify_attack.py: the attack survives at its native resolution
but is destroyed by any resize (gone by 224). So the correct method is (1)
attack the target VLM's OWN vision encoder at its native resolution -- for
LLaVA-1.6 that is openai/clip-vit-large-patch14-336 -- and (2) use Expectation
over Transformation (EOT): at each step average the loss over several random
resizes so the perturbation survives the victim's preprocessing. No surrogates:
white-box on the real encoder.

Objective: image-anchor (match dog-region dense tokens to the aligned dog-free
target), mask-confined, L_inf bounded. Saves full-res images to
results/adv_images_vlm/ for vlm_eval.py.

Example:
    uv run python pgd_vlm_encoder.py --budgets 8,16 --iters 250 --eot 8
    uv run python vlm_eval.py --images-dir results/adv_images_vlm \
        --models llava-hf/llava-v1.6-mistral-7b-hf
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                    help="the target VLM's vision tower (LLaVA-1.6 uses this)")
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--target", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--budgets", default="8,16")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--eot", type=int, default=8, help="EOT samples (random resizes) per step")
    ap.add_argument("--eot-min", type=float, default=0.5, help="min resize fraction")
    ap.add_argument("--outdir", default="results/adv_images_vlm")
    args = ap.parse_args()
    obj_name = args.object

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    RES = model.config.vision_config.image_size          # 336
    PATCH = model.config.vision_config.patch_size        # 14
    GRID = RES // PATCH                                   # 24
    print(f"model={args.model}  res={RES} patch={PATCH} grid={GRID}")

    def to01(img):
        arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def dense(x01):
        """dense (MaskCLIP) patch features; differentiable."""
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
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

    def p_dog(x01):
        return ((dense(x01) @ meas.T) * 100).softmax(-1)[:, :nd].sum(-1)

    orig_img = Image.open(args.original).convert("RGB")
    tgt_img = Image.open(args.target).convert("RGB")
    x0, x_tgt = to01(orig_img), to01(tgt_img)
    W0, H0 = orig_img.size
    orig_full = torch.from_numpy(np.asarray(orig_img, np.float32) / 255) \
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    mask_img = Image.open(args.mask).convert("L")
    m = np.array(mask_img.resize((RES, RES), Image.BILINEAR), np.float32) / 255
    obj = (m.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5).reshape(-1)
    obj_t = torch.from_numpy(obj).to(DEVICE)
    mask_pix = torch.from_numpy(binary_dilation(
        np.asarray(mask_img.resize((RES, RES), Image.NEAREST)) > 127, iterations=2).astype(np.float32)) \
        .view(1, 1, RES, RES).to(DEVICE)

    dense_tgt = dense(x_tgt).detach()

    def eot(x):
        """random-resize a copy (differentiable) to force resize-robustness."""
        s = int(torch.randint(int(RES * args.eot_min), RES + 1, (1,)).item())
        down = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(down, size=(RES, RES), mode="bilinear", align_corners=False)

    def loss_fn(x):
        # image-anchor: pull dog-region dense tokens toward the dog-free target
        return -(dense(x)[obj_t] * dense_tgt[obj_t]).sum(-1).mean()

    def pgd(eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            loss = sum(loss_fn(eot(x)) for _ in range(args.eot)) / args.eot
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    def save_full(x, path):
        delta = x.detach() - x0
        up = F.interpolate(delta, size=(H0, W0), mode="bicubic", align_corners=False)
        out = (orig_full + up).clamp(0, 1)
        arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    @torch.no_grad()
    def robustness(x):
        """P(dog) at native res and after a hard downsample -- did EOT help?"""
        native = p_dog(x)[obj_t].mean().item()
        small = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        small = F.interpolate(small, size=(RES, RES), mode="bilinear", align_corners=False)
        return native, p_dog(small)[obj_t].mean().item()

    os.makedirs(args.outdir, exist_ok=True)
    save_full(x0, f"{args.outdir}/baseline.png")
    bn, bs = robustness(x0)
    print(f"baseline P({obj_name}): native={bn:.3f}  @224={bs:.3f}")
    for b in [float(x) for x in args.budgets.split(",")]:
        x_adv = pgd(b / 255.0)
        save_full(x_adv, f"{args.outdir}/imgtarget_eot_eps{int(b)}.png")
        n, s = robustness(x_adv)
        print(f"eps={b:>4.0f}/255  P({obj_name}) native={n:.3f}  @224(EOT-robust?)={s:.3f}")

    print(f"\nsaved images to {args.outdir}/  -> run vlm_eval.py on this dir")


if __name__ == "__main__":
    main()
