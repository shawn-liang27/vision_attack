"""Stage 21: edge-emphasis test -- does concentrating the perturbation on the
object CONTOUR let a tight ROI (pad0) recover what the pad0.15 margin bought?

Now runs over a DATASET (dataset.jsonl) for the generalization test: the same
4-arm comparison per object. Hypothesis generalizes if edge_v1 recovers pad0.15
across objects, not just the dog.

Arms (all use the H1 global M-Attack crop-matching loss; only the UPDATE differs):
  pad0     : binary tight bbox                       (expected failing baseline)
  pad0.15  : binary bbox + 0.15*RES margin           (expected working baseline)
  edge_v1  : Gaussian edge weight at the SAM contour as a PER-PIXEL STEP SIZE
             (reweighting, no new term -- can't fight M-Attack)
  edge_v2  : pad0 support + boundary penult-patch term to target (grad-logged)

Examples:
    uv run python m_attack_edge.py --dataset dataset.jsonl --seeds 5 --steps 300 --eps 16
    uv run python m_attack_edge.py --source original.png --target removed.png \
        --mask masks/dog_mask.png --object dog          # single-sample mode
"""

import argparse
import json
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
DISTRACTORS = ["background", "wall", "floor", "furniture", "sky"]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--dataset", default=None, help="dataset.jsonl; if omitted, single-sample mode")
    ap.add_argument("--source", default="original.png")
    ap.add_argument("--target", default="removed.png")
    ap.add_argument("--mask", default="masks/dog_mask.png")
    ap.add_argument("--object", default="dog")
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default="pad0.15,edge_v1")
    ap.add_argument("--edge-sigma", type=float, default=0.06)
    ap.add_argument("--edge-maxr", type=float, default=3.0)
    ap.add_argument("--lam-edge", type=float, default=1.0)
    ap.add_argument("--feature-layer", type=int, default=-2)
    ap.add_argument("--outdir", default="results/edge_gen")
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
    dummy = torch.zeros(3, RES, RES)
    scale, ratio = tuple(args.crop_scale), (3 / 4, 4 / 3)
    arms = [a.strip() for a in args.arms.split(",")]
    print(f"edge generalization surrogate={args.model} grid={GRID} arms={arms} seeds={args.seeds}")

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

    os.makedirs(args.outdir, exist_ok=True)

    def save(x, path):
        arr = (x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(path)

    def run_sample(sid, source, target, mask_path, obj):
        x0, x_tgt = load(source), load(target)
        tgt_global = embed_global(x_tgt).detach()
        tgt_penult = embed_penult(x_tgt).detach()
        label_txt = texts([f"a photo of a {l}" for l in [obj] + DISTRACTORS])

        M = np.array(Image.open(mask_path).convert("L").resize((RES, RES), Image.NEAREST)) > 127
        if M.sum() == 0:
            print(f"  [{sid}] empty mask, skipping"); return []
        ys, xs = np.nonzero(M)
        t0, b0, l0, r0 = ys.min(), ys.max(), xs.min(), xs.max()
        dist = np.where(M, distance_transform_edt(M), distance_transform_edt(~M))
        sigma = args.edge_sigma * RES
        ew = np.exp(-(dist ** 2) / (2 * sigma ** 2)); ew[dist > args.edge_maxr * sigma] = 0.0
        edge_weight = torch.from_numpy((ew / ew.max()).astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)

        def bbox_region(pf):
            pad = int(pf * RES); reg = torch.zeros((1, 1, RES, RES), device=DEVICE)
            reg[:, :, max(0, t0 - pad):min(RES, b0 + pad + 1), max(0, l0 - pad):min(RES, r0 + pad + 1)] = 1.0
            return reg

        mg = np.array(Image.open(mask_path).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        Mg = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.5
        dg_out = distance_transform_edt(~Mg); dg_in = distance_transform_edt(Mg)
        dg = np.where(Mg, dg_in, dg_out)
        edge_patch = torch.from_numpy((dg <= 1.5).reshape(-1)).to(DEVICE)
        obj_patch = torch.from_numpy(Mg.reshape(-1)).to(DEVICE)

        @torch.no_grad()
        def metrics(x01):
            cos_t = (embed_global(x01) @ tgt_global.T).item()
            gp = float(((embed_global(x01) @ label_txt.T) * 100).softmax(-1).squeeze(0)[0])
            rp = ((embed_dense(x01)[obj_patch] @ label_txt.T) * 100).softmax(-1)[:, 0].mean().item()
            return cos_t, gp, rp

        save(x0, f"{args.outdir}/{sid}__baseline.png")
        rows = []
        for arm in arms:
            if arm == "pad0":
                support, stepw = bbox_region(0.0), None
            elif arm == "pad0.15":
                support, stepw = bbox_region(0.15), None
            elif arm == "edge_v1":
                support, stepw = (edge_weight > 0.01).float(), edge_weight
            elif arm == "edge_v2":
                support, stepw = bbox_region(0.0), None
            else:
                raise SystemExit(f"unknown arm {arm}")

            for seed in range(args.seeds):
                seed_all(2000 + seed)
                delta = torch.zeros_like(x0, requires_grad=True)
                momentum = torch.zeros_like(x0)
                aligns = []
                for _ in range(args.steps):
                    box_s = T.RandomResizedCrop.get_params(dummy, scale, ratio)
                    box_t = T.RandomResizedCrop.get_params(dummy, scale, ratio)
                    adv = torch.clamp(x0 + delta * support, 0, 1)
                    crop_adv = TF.resized_crop(adv, *box_s, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
                    sim = (embed_global(crop_adv) * embed_global(
                        TF.resized_crop(x_tgt, *box_t, [RES, RES], TF.InterpolationMode.BICUBIC, antialias=True)
                        ).detach()).sum()
                    if arm == "edge_v2" and args.lam_edge > 0:
                        penult = embed_penult(adv)
                        edge_loss = ((penult[edge_patch] - tgt_penult[edge_patch]) ** 2).mean()
                        g_sim, = torch.autograd.grad(sim, delta, retain_graph=True)
                        g_edge, = torch.autograd.grad(edge_loss, delta)
                        if g_sim.abs().sum() > 0 and g_edge.abs().sum() > 0:
                            aligns.append(F.cosine_similarity(g_sim.flatten(), (-g_edge).flatten(), dim=0).item())
                        g = g_sim - args.lam_edge * g_edge
                    else:
                        g, = torch.autograd.grad(sim, delta)
                    with torch.no_grad():
                        g = g * support
                        momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                        stepv = alpha if stepw is None else alpha * stepw
                        delta.add_(stepv * momentum.sign() * support).clamp_(-eps, eps)
                        delta.data = torch.clamp(x0 + delta * support, 0, 1) - x0
                    delta.requires_grad_(True)

                adv = torch.clamp(x0 + delta.detach() * support, 0, 1)
                fn = f"{sid}__{arm}_seed{seed}.png"
                save(adv, f"{args.outdir}/{fn}")
                m = metrics(adv)
                ga = float(np.mean(aligns)) if aligns else float("nan")
                rows.append(f"{sid},{obj},{arm},{seed},{fn},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{ga:.4f}")
                print(f"  [{sid}] {arm:<9} seed={seed} cos->tgt={m[0]:.3f} gP={m[1]:.3f} roiP={m[2]:.3f}"
                      + (f" ga={ga:.3f}" if not np.isnan(ga) else ""))
        return rows

    # gather samples
    if args.dataset:
        with open(args.dataset) as f:
            samples = [json.loads(l) for l in f if l.strip()]
    else:
        sid = os.path.splitext(os.path.basename(args.source))[0]
        samples = [{"id": sid, "image": args.source, "object": args.object,
                    "mask": args.mask, "target": args.target}]

    all_rows = ["id,object,arm,seed,filename,cos_to_target,global_p_obj,roi_p_obj,grad_align"]
    for s in samples:
        print(f"\n=== sample {s['id']}  object='{s['object']}' ===")
        for miss in ("mask", "target"):
            if not os.path.exists(s[miss]):
                print(f"  MISSING {miss}={s[miss]} (run prep_dataset.py first); skipping sample"); break
        else:
            all_rows += run_sample(s["id"], s["image"], s["target"], s["mask"], s["object"])

    with open(f"{args.outdir}/metrics.csv", "w") as f:
        f.write("\n".join(all_rows) + "\n")
    print(f"\nsaved {args.outdir}/metrics.csv + images. Next: vlm_eval_dataset.py then summarize_edge.py")


if __name__ == "__main__":
    main()
