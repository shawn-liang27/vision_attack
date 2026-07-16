"""Stage 14: attack the representation the VLM projector actually consumes.

Stage 13 result: on LLaVA-1.5's OWN encoder, CLIP-space concealment succeeded
(pooled P(dog) 0.95->0.09, zero-shot argmax dog->cushion) yet LLaVA said "dog"
10/10. Diagnosis (confirmed, same encoder): we concealed the wrong
representation. LLaVA's projector reads the RAW penultimate patch hidden states
(vision_feature_layer=-2, drop CLS) -- NOT CLIP's visual_projection/zero-shot
head. So we must steer THAT representation.

Objective (image anchor, projector's actual input):
    loss = - mean_{i in ROI} cos( H_adv^{(-2)}[i],  H_bg^{(-2)}[i] )

where H^{(-2)} is the raw hidden state LLaVA feeds its projector. Verified with
BOTH the projector-space gate (penult cos-to-bg, the one that matters) AND the
old CLIP zero-shot P(dog) (for contrast -- may move differently). Saves exact
336 squares for the same-encoder VLM caption test.

Example:
    uv run python pgd_projector_target.py --budgets 8,16,32 --iters 300
    uv run python vlm_eval.py --images-dir results/projector_target/square \
        --models llava-hf/llava-1.5-7b-hf
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import CLIPModel, CLIPProcessor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                    help="the VLM's vision tower (LLaVA-1.5)")
    ap.add_argument("--original", default="original.png")
    ap.add_argument("--background", default="removed_aligned.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--feature-layer", type=int, default=-2,
                    help="hidden-state index the projector reads (LLaVA-1.5 = -2)")
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--outdir", default="results/projector_target")
    args = ap.parse_args()
    obj_name = args.object
    RES, FL = args.res, args.feature_layer

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = model.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"model={args.model} res={RES} grid={GRID} feature_layer={FL}")

    def to01(img):
        arr = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def feats(x01):
        """(dense joint feature for zero-shot readout, RAW penultimate patch tokens
        = the projector's actual input). Both differentiable."""
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        penult = out.hidden_states[FL][:, 1:]                 # RAW, drop CLS (projector input)
        layer = vm.encoder.layers[-1]
        h_in = out.hidden_states[-2]
        a = layer.layer_norm1(h_in)
        h = h_in + layer.self_attn.out_proj(layer.self_attn.v_proj(a))
        h = h + layer.mlp(layer.layer_norm2(h))
        d = model.visual_projection(vm.post_layernorm(h))     # CLIP zero-shot head (for contrast)
        return (F.normalize(d[:, 1:], dim=-1).squeeze(0),
                F.normalize(penult, dim=-1).squeeze(0))

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [obj_name, "couch", "cushion", "wall", "window", "houseplant"]
    label_txt = texts([f"a photo of a {l}" for l in labels])

    def zero_shot(feat):
        return ((feat @ label_txt.T) * 100).softmax(-1)

    orig_img = Image.open(args.original).convert("RGB")
    bg_img = Image.open(args.background).convert("RGB")
    x0, x_bg = to01(orig_img), to01(bg_img)
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

    dense_bg, penult_bg = feats(x_bg)
    dense_bg, penult_bg = dense_bg.detach(), penult_bg.detach()
    dense_o0, penult_o0 = feats(x0)
    dense_o0, penult_o0 = dense_o0.detach(), penult_o0.detach()

    def pooled(d):
        return F.normalize(d[obj_t].mean(0), dim=-1)

    def pgd(eps):
        step = 2.5 * eps / args.iters
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(args.iters):
            x = torch.clamp(x0 + delta * mask_pix, 0, 1)
            _, penult = feats(x)
            loss = -(penult[obj_t] * penult_bg[obj_t]).sum(-1).mean()   # projector-input match
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        return torch.clamp(x0 + delta.detach() * mask_pix, 0, 1)

    def save_full(x, path):
        up = F.interpolate(x.detach() - x0, size=(H0, W0), mode="bicubic", align_corners=False)
        arr = ((orig_full + up).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def save_square(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255) \
            .round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    os.makedirs(args.outdir, exist_ok=True)
    sqdir = os.path.join(args.outdir, "square")
    os.makedirs(sqdir, exist_ok=True)
    save_square(x0, f"{sqdir}/baseline_eps0.png")

    # baseline: how similar are the ORIGINAL penult ROI tokens to the bg already?
    base_penult_bg = (penult_o0[obj_t] * penult_bg[obj_t]).sum(-1).mean().item()
    pl0 = zero_shot(pooled(dense_o0)).cpu().numpy()
    print(f"baseline: penult cos(orig ROI, bg)={base_penult_bg:.3f}  "
          f"pooled P({obj_name})={pl0[0]:.3f}")

    budgets = [float(b) for b in args.budgets.split(",")]
    rows = {"budget": [], "penult_to_bg": [], "penult_to_orig": [], "pdog_pooled": [], "argmax": []}
    summary = [f"labels={labels}",
               f"baseline penult cos(orig ROI,bg)={base_penult_bg:.4f} pooled P({obj_name})={pl0[0]:.4f}"]
    for b in budgets:
        x_adv = pgd(b / 255.0)
        dense_a, penult_a = feats(x_adv)
        dense_a, penult_a = dense_a.detach(), penult_a.detach()
        to_bg = (penult_a[obj_t] * penult_bg[obj_t]).sum(-1).mean().item()      # GATE (right space)
        to_orig = (penult_a[obj_t] * penult_o0[obj_t]).sum(-1).mean().item()
        pl = zero_shot(pooled(dense_a)).cpu().numpy()
        slug = f"penult_eps{int(b)}"
        save_full(x_adv, f"{args.outdir}/{slug}.png")
        save_square(x_adv, f"{sqdir}/{slug}.png")
        rows["budget"].append(b); rows["penult_to_bg"].append(to_bg)
        rows["penult_to_orig"].append(to_orig); rows["pdog_pooled"].append(float(pl[0]))
        rows["argmax"].append(labels[int(pl.argmax())])
        line = (f"eps={b:>4.0f}/255  GATE penult cos->bg {base_penult_bg:.3f}->{to_bg:.3f} "
                f"(->orig {to_orig:.3f}) | CLIP zero-shot P({obj_name}) {pl0[0]:.3f}->{pl[0]:.3f} "
                f"argmax={labels[int(pl.argmax())]}")
        print(line); summary.append(line)

    with open(f"{args.outdir}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bx = rows["budget"]
    ax[0].axhline(base_penult_bg, ls=":", c="gray", label="orig ROI vs bg")
    ax[0].plot(bx, rows["penult_to_bg"], "o-", label="adv ROI vs bg (gate)")
    ax[0].plot(bx, rows["penult_to_orig"], "s--", label="adv ROI vs orig")
    ax[0].set_title("GATE: penultimate-token match (projector input)")
    ax[0].set_xlabel("L_inf (/255)"); ax[0].set_ylabel("mean cosine"); ax[0].legend(fontsize=8)
    ax[1].axhline(pl0[0], ls=":", c="gray", label="baseline")
    ax[1].plot(bx, rows["pdog_pooled"], "o-", color="tab:red")
    ax[1].set_title(f'CLIP zero-shot P("{obj_name}") (contrast head)')
    ax[1].set_xlabel("L_inf (/255)"); ax[1].legend(fontsize=8)
    fig.suptitle("attacking the projector's actual input (penultimate patch tokens)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/projector_target.png", dpi=130)
    print(f"\nsaved {args.outdir}/ -> caption square/ via vlm_eval on llava-1.5-7b")


if __name__ == "__main__":
    main()
