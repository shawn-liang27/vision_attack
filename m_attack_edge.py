"""Stage 21: edge-emphasis test -- does concentrating the perturbation on the
object CONTOUR let a tight ROI (pad0) recover what the pad0.15 margin bought?

Hypothesis (data-grounded: pad0 fails, pad0.15 works): the margin's job is
reaching the object edges. So emphasizing the boundary should recover pad0's
failure without a full margin.

All arms use the H1 global signal: whole-image M-Attack crop-matching loss
(-cos(pooled(crop(adv)), pooled(crop(target)))), MI-FGSM ascent; only the UPDATE
region/weighting differs:
  pad0      : binary tight bbox            (expected failing baseline)
  pad0.15   : binary bbox + 0.15*RES margin (expected working baseline)
  edge_v1   : continuous edge weight (Gaussian annulus at the SAM contour) used
              as a PER-PIXEL STEP SIZE -- reweighting, no new loss term, cannot
              fight M-Attack (interpretation 1).
  edge_v2   : pad0 support + an added boundary term matching contour-band
              penultimate patch features (layer -2, what LLaVA reads) to the
              target's -- a second term, so we LOG grad cooperation (may fight).

N seeds/arm -> success RATE. Confirmed if edge_v1 climbs from pad0's rate toward
pad0.15's.

Example:
    uv run python m_attack_edge.py --seeds 5 --steps 300 --eps 16
    uv run python vlm_eval.py --images-dir results/edge --outdir results/edge \
        --object dog --models llava-hf/llava-1.5-7b-hf
    uv run python summarize_edge.py --dir results/edge
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import distance_transform_edt
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--source", default="original.png")
    ap.add_argument("--target", default="removed.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default="pad0,pad0.15,edge_v1,edge_v2")
    ap.add_argument("--edge-sigma", type=float, default=0.06, help="contour band width, fraction of RES")
    ap.add_argument("--edge-maxr", type=float, default=3.0, help="clip edge weight beyond maxr*sigma")
    ap.add_argument("--lam-edge", type=float, default=1.0, help="edge_v2 boundary-term weight")
    ap.add_argument("--feature-layer", type=int, default=-2)
    ap.add_argument("--object", default="dog")
    ap.add_argument("--outdir", default="results/edge")
    args = ap.parse_args()
    RES, FL = args.input_res, args.feature_layer
    eps, alpha = args.eps / 255.0, args.alpha / 255.0

    model = CLIPModel.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model)
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = model.config.vision_config.patch_size
    GRID = RES // PATCH
    print(f"edge test surrogate={args.model} grid={GRID} arms={args.arms} seeds={args.seeds}")

    def load(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def embed_global(x01):
        v = model.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True).pooler_output
        return F.normalize(model.visual_projection(v), dim=-1)

    def embed_penult(x01):
        out = model.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        return F.normalize(out.hidden_states[FL][:, 1:], dim=-1).squeeze(0)

    def embed_dense(x01):
        vm = model.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        layer = vm.encoder.layers[-1]
        h = out.hidden_states[-2]
        a = layer.layer_norm1(h)
        h = h + layer.self_attn.out_proj(layer.self_attn.v_proj(a))
        h = h + layer.mlp(layer.layer_norm2(h))
        return F.normalize(model.visual_projection(vm.post_layernorm(h))[:, 1:], dim=-1).squeeze(0)

    @torch.no_grad()
    def texts(prompts):
        tok = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(model.text_projection(model.text_model(**tok).pooler_output), dim=-1)

    labels = [args.object, "cat", "couch", "cushion", "wall", "window"]
    label_txt = texts([f"a photo of a {l}" for l in labels])

    x0, x_tgt = load(args.source), load(args.target)
    tgt_global = embed_global(x_tgt).detach()
    tgt_penult = embed_penult(x_tgt).detach()

    # --- masks / edge weighting from the SAM segmentation ---------------------
    M = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.NEAREST)) > 127
    ys, xs = np.nonzero(M)
    t0, b0, l0, r0 = ys.min(), ys.max(), xs.min(), xs.max()
    dist = np.where(M, distance_transform_edt(M), distance_transform_edt(~M))  # unsigned dist to contour
    sigma = args.edge_sigma * RES
    ew = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    ew[dist > args.edge_maxr * sigma] = 0.0
    ew = ew / ew.max()
    edge_weight = torch.from_numpy(ew.astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)

    def bbox_region(pad_frac):
        pad = int(pad_frac * RES)
        reg = torch.zeros((1, 1, RES, RES), device=DEVICE)
        reg[:, :, max(0, t0 - pad):min(RES, b0 + pad + 1), max(0, l0 - pad):min(RES, r0 + pad + 1)] = 1.0
        return reg

    # grid-level edge / adjacent-background patches for edge_v2
    mg = np.array(Image.open(args.mask).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
    Mg = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5
    dg_out = distance_transform_edt(~Mg)
    dg_in = distance_transform_edt(Mg)
    dg = np.where(Mg, dg_in, dg_out)
    edge_patch = torch.from_numpy((dg <= 1.5).reshape(-1)).to(DEVICE)          # contour band (grid)
    bg_patch = torch.from_numpy((~Mg & (dg_out <= 2.5)).reshape(-1)).to(DEVICE)  # bg just outside
    obj_patch = torch.from_numpy(Mg.reshape(-1)).to(DEVICE)
    dummy = torch.zeros(3, RES, RES)
    scale, ratio = tuple(args.crop_scale), (3 / 4, 4 / 3)

    os.makedirs(args.outdir, exist_ok=True)

    def save(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)
    save(x0, f"{args.outdir}/baseline.png")

    @torch.no_grad()
    def metrics(x01):
        cos_t = (embed_global(x01) @ tgt_global.T).item()
        gpd = float(((embed_global(x01) @ label_txt.T) * 100).softmax(-1).squeeze(0)[0])
        rpd = ((embed_dense(x01)[obj_patch] @ label_txt.T) * 100).softmax(-1)[:, 0].mean().item()
        return cos_t, gpd, rpd

    c0 = metrics(x0)
    print(f"baseline cos(orig,target)={c0[0]:.3f} gP(dog)={c0[1]:.3f} roiP(dog)={c0[2]:.3f}")

    arms = [a.strip() for a in args.arms.split(",")]
    rows = ["arm,seed,filename,cos_to_target,global_p_dog,roi_p_dog,grad_align"]
    for arm in arms:
        # perturbation support + per-pixel step weighting for this arm
        if arm == "pad0":
            support, stepw = bbox_region(0.0), None
        elif arm == "pad0.15":
            support, stepw = bbox_region(0.15), None
        elif arm == "edge_v1":
            support, stepw = (edge_weight > 0.01).float(), edge_weight   # step scaled by edge weight
        elif arm == "edge_v2":
            support, stepw = bbox_region(0.0), None                      # tight support + added term
        else:
            raise SystemExit(f"unknown arm {arm}")

        for seed in range(args.seeds):
            seed_all(2000 + seed)
            delta = torch.zeros_like(x0, requires_grad=True)
            momentum = torch.zeros_like(x0)
            aligns = []
            for step in range(args.steps):
                box_s = T.RandomResizedCrop.get_params(dummy, scale, ratio)
                box_t = T.RandomResizedCrop.get_params(dummy, scale, ratio)
                adv = torch.clamp(x0 + delta * support, 0, 1)
                crop_adv = TF.resized_crop(adv, *box_s, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
                sim = (embed_global(crop_adv) * embed_global(
                    TF.resized_crop(x_tgt, *box_t, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
                    ).detach()).sum()

                if arm == "edge_v2" and args.lam_edge > 0:
                    penult = embed_penult(adv)                          # full-image forward, layer LLaVA reads
                    edge_loss = ((penult[edge_patch] - tgt_penult[edge_patch]) ** 2).mean()
                    g_sim, = torch.autograd.grad(sim, delta, retain_graph=True)
                    g_edge, = torch.autograd.grad(edge_loss, delta)
                    if g_sim.abs().sum() > 0 and g_edge.abs().sum() > 0:
                        aligns.append(F.cosine_similarity(g_sim.flatten(), (-g_edge).flatten(), dim=0).item())
                    g = g_sim - args.lam_edge * g_edge                  # ascend sim, descend edge_loss
                else:
                    g, = torch.autograd.grad(sim, delta)                # pure M-Attack ascent

                with torch.no_grad():
                    g = g * support
                    momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                    step_scale = alpha if stepw is None else alpha * stepw
                    delta.add_(step_scale * momentum.sign() * support).clamp_(-eps, eps)
                    delta.data = torch.clamp(x0 + delta * support, 0, 1) - x0
                delta.requires_grad_(True)

            adv = torch.clamp(x0 + delta.detach() * support, 0, 1)
            fn = f"{arm}_seed{seed}.png"
            save(adv, f"{args.outdir}/{fn}")
            m = metrics(adv)
            ga = float(np.mean(aligns)) if aligns else float("nan")
            rows.append(f"{arm},{seed},{fn},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{ga:.4f}")
            print(f"{arm:<9} seed={seed} cos->tgt={m[0]:.3f} gP(dog)={m[1]:.3f} roiP(dog)={m[2]:.3f}"
                  + (f" grad_align={ga:.3f}" if not np.isnan(ga) else ""))

    with open(f"{args.outdir}/metrics.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nsaved {args.outdir}/metrics.csv + images. Next: vlm_eval (--outdir {args.outdir}), "
          f"then summarize_edge.py --dir {args.outdir}")


if __name__ == "__main__":
    main()
