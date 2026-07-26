"""Stage 25 + 26: AdvEDM-R (patch-token nulling) with optional MULTI-LAYER DEEP
FEATURE ERASURE (deep supervision), tested under STRICT source-absence.

AdvEDM-R (Eqs 3-8): steer the object-region patch tokens toward a REMOVAL TARGET,
suppress CLS-vs-object-text, and fixate attention to preserve the rest. Target =
the INPAINT image (object -> plausible background; --target inpaint, DEFAULT) -- the
sound "object-absent" counterfactual -- or the zeroed black box (--target zeroed).
  L_cls = CS(cls_adv, E_t(obj))                                      # push CLS off object-text (min)
  L_p   = -mean_{i in obj} CS(patch_adv[-1][i], patch_tgt[-1][i])    # steer obj tokens -> target @ ENCODER OUTPUT
  L_fix = -mean_{j in bg}  A_clean[j] * CS(patch_adv[-1][j], patch_clean[-1][j])  # attn-weighted preserve
  L = w1 L_cls + w2 L_p + w3 L_fix     (w1=0.5, w2=2, w3=0.2), Adam lr=5e-3, 500 it, L2 ball.

DEEP SUPERVISION (arm 'deep', --w-inter>0): ALSO steer obj tokens at INTERMEDIATE
layers l in --inter-layers. Truly-shallow layers (e.g. 2,4,6 of 24) keep patch tokens
spatially local / least globally entangled -- the fairest test of whether erasing the
object early cuts its info flow before deeper layers can reconstruct it:
  L_p_inter = mean_l [ -mean_{i in obj} CS(patch_adv[l][i], patch_tgt[l][i]) ]
  L = ... + w_inter * L_p_inter
Run --arms base,deep to A/B last-layer-only vs +intermediate under identical seeds.

CRITICAL LAYER NOTE: AdvEDM matches the CLIP ENCODER-OUTPUT tokens (hidden_states
[-1]); LLaVA's decoder reads the PENULTIMATE (-2) tokens. We attack [-1] (+inter)
and DIAGNOSE at [-2] (what the victim reads). The deep-supervision hypothesis is
that erasing early forces [-2] object-free too; diagnostics test it directly
(objpatch_cos_tgt_inter = did inter steering reach the target; roiP@-2 = did it reach LLaVA's layer).

Region mask: (a) 'seg' = segmentation -> token grid; (b) 'sim' = top-20% patches by
CS(patch, E_t(obj)). Eval: WEAK = object in "Describe this image."? vs STRICT =
absent under ALL of {direct, list, presup, detail, read-text(sign)}. 5 seeds, LLaVA-1.5-7b.

    uv run python advedm_r.py --dataset dataset.jsonl --regions seg,sim \\
        --budgets 20,40,80 --seeds 5 --arms base,deep --inter-layers 2,4,6 --target inpaint
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine", "pet", "animal"], "cat": ["cat", "kitten", "feline", "pet", "animal"],
       "car": ["car", "vehicle", "truck", "sedan", "suv", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "plane": ["plane", "airplane", "aircraft", "jet"], "sign": ["sign", "signage", "placard"]}
CAT = {"dog": "animal", "cat": "animal", "car": "vehicle", "airplane": "aircraft", "plane": "aircraft", "sign": "sign"}


def present(text, obj):
    return any(s in text.lower() for s in SYN.get(obj, [obj]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--regions", default="seg,sim")
    ap.add_argument("--budgets", default="20,40,80", help="L2 epsilon values")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--w2", type=float, default=2.0)
    ap.add_argument("--w3", type=float, default=0.2)
    ap.add_argument("--topk", type=float, default=0.20)
    ap.add_argument("--target", default="inpaint", choices=["inpaint", "zeroed"],
                    help="removal target for L_p/L_p_inter: 'inpaint'=object->background (sound), 'zeroed'=black box")
    ap.add_argument("--inter-layers", default="",
                    help="intermediate layers for deep supervision: ints (abs idx) or floats in (0,1) (depth frac); empty=L/3,2L/3")
    ap.add_argument("--w-inter", type=float, default=1.0, help="weight alpha on intermediate-layer nulling loss")
    ap.add_argument("--arms", default="deep", help="comma list of {base,deep}: base=last-layer-only (w_inter=0), deep=+intermediate")
    ap.add_argument("--outdir", default="results/advedm_r")
    args = ap.parse_args()
    RES = args.res

    # eager attention so output_attentions actually returns weights (SDPA returns None) -- needed for L_fix
    clip = CLIPModel.from_pretrained(args.surrogate, attn_implementation="eager").to(DEVICE).eval()
    clip.requires_grad_(False)
    cproc = CLIPProcessor.from_pretrained(args.surrogate)
    ipc = cproc.image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH

    victim = AutoModelForImageTextToText.from_pretrained(args.victim, torch_dtype=DTYPE, device_map=DEVICE).eval()
    victim.requires_grad_(False)
    vproc = AutoProcessor.from_pretrained(args.victim)
    vtok = vproc.tokenizer
    print(f"surrogate={args.surrogate} grid={GRID}; victim={args.victim}")
    print("LAYER: attack matches encoder-output patch tokens hidden_states[-1]; "
          "diagnostics read penultimate hidden_states[-2] (LLaVA's input)")

    NL = clip.config.vision_config.num_hidden_layers  # 24 for ViT-L/14; hidden_states has NL+1 entries (0=embeds)

    def parse_layers(spec):
        if not spec.strip():
            return sorted({round(NL / 3), round(2 * NL / 3)})
        out = []
        for t in spec.split(","):
            t = t.strip()
            if not t:
                continue
            v = float(t)
            out.append(max(1, min(NL, round(v * NL) if 0 < v < 1 else int(v))))
        return sorted(set(out))
    inter_layers = parse_layers(args.inter_layers)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"deep-supervision layers (of {NL}): {inter_layers} "
          f"(depths {[round(l / NL, 2) for l in inter_layers]}); arms={arms} w_inter={args.w_inter}")
    print(f"removal target = {args.target} ({'inpaint image, object->background' if args.target == 'inpaint' else 'zeroed black box'})")

    def txt_emb(words):
        tk = cproc(text=words, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(clip.text_projection(clip.text_model(**tk).pooler_output), dim=-1)

    def vis(x01, want_attn=False):
        """returns dict: cls(image-embed), patch_last(-1, raw), patch_penult(-2, raw), attn(CLS->patch)."""
        vm = clip.vision_model
        out = vm((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                 output_hidden_states=True, output_attentions=want_attn)
        cls = F.normalize(clip.visual_projection(out.pooler_output), dim=-1)      # image embedding
        patch_last = out.last_hidden_state[:, 1:]                                  # encoder output (-1)
        patch_penult = out.hidden_states[-2][:, 1:]                                # LLaVA's input (-2)
        A = None
        if want_attn:
            att = torch.stack(out.attentions).mean(0).mean(1)   # mean over layers, then heads -> (1,seq,seq)
            A = att[:, 0, 1:]                                    # CLS -> patch attention (1, P)
        return dict(cls=cls, last=patch_last, penult=patch_penult, attn=A, hs=out.hidden_states)

    def load01(p):
        img = Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def gen(x01, prompt):
        img = Image.fromarray((x01.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = vproc.apply_chat_template(messages, add_generation_prompt=True)
        inp = vproc(images=img, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            g = victim.generate(**inp, max_new_tokens=48, do_sample=False)
        return vtok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    regions = args.regions.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    ans_rows = [["arm", "object", "region", "eps", "seed", "criterion", "prompt", "answer_text", "object_present"]]
    diag_rows = [["arm", "object", "region", "eps", "seed", "cls_cos_obj", "objpatch_cos_tgt",
                  "objpatch_cos_tgt_inter", "roiP_penult_layer-2"]]
    curve_rows = [["arm", "object", "region", "eps", "seed", "iter", "L_cls", "L_p", "L_p_inter", "L_fix"]]
    printed_shapes = False

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not os.path.exists(s["mask"]):
            print(f"[{sid}] no mask; skip"); continue
        x0 = load01(s["image"])
        E_t = txt_emb([f"a photo of a {obj}"])[0]
        label_txt = txt_emb([f"a photo of a {obj}"] + [f"a photo of {d}" for d in
                             ["background", "a wall", "a floor", "furniture", "the sky"]])
        # removal reference R: inpaint image (object -> plausible background, sound counterfactual) or zeroed (black box)
        if args.target == "inpaint" and s.get("target") and os.path.exists(s["target"]):
            xR = load01(s["target"])
        else:
            if args.target == "inpaint":
                print(f"[{sid}] no inpaint target file; falling back to zeroed black box")
            mpix = torch.from_numpy((np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.NEAREST)) > 127)
                                    .astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)
            xR = x0 * (1 - mpix)
        clean = vis(x0, want_attn=True)
        featR = vis(xR)
        # removal-target references at the intermediate layers (normalized, detached), for deep supervision
        featR_inter = [F.normalize(featR["hs"][l][:, 1:].squeeze(0), dim=-1).detach() for l in inter_layers]
        if not printed_shapes:
            print(f"SHAPES: patch_last={tuple(clean['last'].shape)} attn A(CLS->patch)={tuple(clean['attn'].shape)}")
            printed_shapes = True
        A_clean = clean["attn"][0].detach()   # (P,)

        # region token indices
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        seg_obj = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)
        proj_patch = F.normalize(clip.visual_projection(clean["last"]), dim=-1).squeeze(0)  # (P,D) joint
        sim = (proj_patch @ E_t)  # (P,)
        k = max(1, int(args.topk * sim.numel()))
        sim_obj = torch.zeros_like(seg_obj); sim_obj[sim.topk(k).indices] = True

        prompts = {"direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
                   "list": "List all objects you see in this image.",
                   "presup": f"What {CAT.get(obj,'object')} is in this image?",
                   "detail": "Describe this image in detail."}
        if obj == "sign":
            prompts["read"] = "Read any text visible in the image."
        weak_prompt = "Describe this image."

        for region in regions:
            obj_idx = seg_obj if region == "seg" else sim_obj
            bg_idx = ~obj_idx
            if obj_idx.sum() == 0:
                continue
            featR_last = featR["last"].squeeze(0).detach()
            clean_last = clean["last"].squeeze(0).detach()
            print(f"\n=== {sid} '{obj}' region={region} ({int(obj_idx.sum())}/{GRID*GRID} obj tokens) ===")

            for arm in arms:
                w_inter = args.w_inter if arm == "deep" else 0.0
                for b in budgets:
                    for seed in range(args.seeds):
                        g = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
                        delta = (torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * (b / (RES * 1.7))
                        delta = delta.detach().requires_grad_(True)
                        opt = torch.optim.Adam([delta], lr=args.lr)
                        for it in range(args.iters):
                            x = torch.clamp(x0 + delta, 0, 1)
                            f = vis(x, want_attn=False)   # L_fix uses fixed clean attention weights
                            L_cls = (f["cls"].squeeze(0) @ E_t)                                    # min -> away from obj text
                            pl = F.normalize(f["last"].squeeze(0), dim=-1)
                            Lp = -(pl[obj_idx] * F.normalize(featR_last, dim=-1)[obj_idx]).sum(-1).mean()
                            # deep supervision: null obj tokens at intermediate layers toward the zeroed image
                            Lp_inter = x0.new_zeros(())
                            if w_inter and inter_layers:
                                for i_l, l in enumerate(inter_layers):
                                    pil = F.normalize(f["hs"][l][:, 1:].squeeze(0), dim=-1)
                                    Lp_inter = Lp_inter - (pil[obj_idx] * featR_inter[i_l][obj_idx]).sum(-1).mean()
                                Lp_inter = Lp_inter / len(inter_layers)
                            w = A_clean[bg_idx]; w = w / w.sum().clamp_min(1e-9)
                            cs_bg = (pl[bg_idx] * F.normalize(clean_last, dim=-1)[bg_idx]).sum(-1)
                            Lfix = -(w * cs_bg).sum()
                            L = args.w1 * L_cls + args.w2 * Lp + w_inter * Lp_inter + args.w3 * Lfix
                            opt.zero_grad(); L.backward(); opt.step()
                            with torch.no_grad():
                                n = delta.flatten().norm(2)
                                if n > b:
                                    delta.mul_(b / n)
                                delta.data = torch.clamp(x0 + delta, 0, 1) - x0
                            if it % 50 == 0 or it == args.iters - 1:
                                curve_rows.append([arm, obj, region, b, seed, it, round(L_cls.item(), 4),
                                                   round(Lp.item(), 4), round(float(Lp_inter), 4), round(Lfix.item(), 4)])

                        adv = torch.clamp(x0 + delta.detach(), 0, 1)
                        with torch.no_grad():
                            fa = vis(adv)
                            cls_cos = (fa["cls"].squeeze(0) @ E_t).item()
                            opc = (F.normalize(fa["last"].squeeze(0), dim=-1)[obj_idx] *
                                   F.normalize(featR_last, dim=-1)[obj_idx]).sum(-1).mean().item()
                            opc_inter = float(np.mean([  # did the intermediate nulling take?
                                (F.normalize(fa["hs"][l][:, 1:].squeeze(0), dim=-1)[obj_idx] *
                                 featR_inter[i_l][obj_idx]).sum(-1).mean().item()
                                for i_l, l in enumerate(inter_layers)])) if inter_layers else float("nan")
                            pen = F.normalize(clip.visual_projection(fa["penult"].squeeze(0)), dim=-1)
                            roiP = ((pen[obj_idx] @ label_txt.T) * 100).softmax(-1)[:, 0].mean().item()
                        diag_rows.append([arm, obj, region, b, seed, round(cls_cos, 4), round(opc, 4),
                                          round(opc_inter, 4), round(roiP, 4)])

                        linf = (delta.detach().abs().max().item()) * 255
                        # WEAK (their metric)
                        wa = gen(adv, weak_prompt)
                        ans_rows.append([arm, obj, region, b, seed, "weak", "describe", wa.replace("\n", " ")[:200], present(wa, obj)])
                        # STRICT (ours)
                        strict_present = False
                        for pk, pt in prompts.items():
                            a = gen(adv, pt); p = present(a, obj); strict_present = strict_present or p
                            ans_rows.append([arm, obj, region, b, seed, "strict", pk, a.replace("\n", " ")[:200], p])
                        print(f"  [{arm}] eps={b:>4.0f} L2 seed={seed} Linf={linf:.0f}/255 | cls_cos_obj={cls_cos:.3f} "
                              f"objpatch_cos_tgt={opc:.3f} inter={opc_inter:.3f} roiP@-2={roiP:.3f} | "
                              f"weak_absent={not present(wa,obj)} strict_absent={not strict_present}")

    for name, rows in [("answers", ans_rows), ("diagnostics", diag_rows), ("curves", curve_rows)]:
        with open(f"{args.outdir}/{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)

    # ---- summary: weak vs strict removal rate per (object, region) -----------
    from collections import defaultdict
    grp = defaultdict(lambda: {"weak": defaultdict(list), "strict": defaultdict(list)})
    for r in ans_rows[1:]:
        arm, obj, region, eps, seed, crit, prompt, txt, pres = r
        grp[(arm, obj, region, eps)][crit][seed].append(pres == "True" or pres is True)
    print("\n=== WEAK (caption-mention) vs STRICT (all-probe conjunction) removal rate ===")
    print(f"{'arm':<6}{'object':<10}{'region':<6}{'eps':>5}{'weak_removal%':>15}{'strict_removal%':>17}")
    summ = [["arm", "object", "region", "eps", "weak_removal_rate", "strict_removal_rate", "n"]]
    for (arm, obj, region, eps), d in sorted(grp.items()):
        weak_rem = 100 * np.mean([not any(v) for v in d["weak"].values()])      # removed if not present
        strict_rem = 100 * np.mean([not any(v) for v in d["strict"].values()])
        n = len(d["strict"])
        summ.append([arm, obj, region, eps, round(weak_rem, 1), round(strict_rem, 1), n])
        print(f"{arm:<6}{obj:<10}{region:<6}{eps:>5}{weak_rem:>14.0f}%{strict_rem:>16.0f}%")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(summ)
    print(f"\nsaved {args.outdir}/{{answers,diagnostics,curves,summary}}.csv")
    print("READ: weak>>strict => steering only removes the caption mention, object survives direct probing "
          "(check diagnostics: objpatch_cos_tgt high [L_p reached the target @ -1] but roiP@-2 still high => "
          "didn't propagate to the layer LLaVA reads). Weight on dog/cat/sign.")
    print("DEEP-SUPERVISION A/B (base vs deep): if deep raises strict_removal AND drops roiP@-2, erasing early "
          "propagated to LLaVA's layer -> the idea works. If deep matches intermediate (objpatch_cos_tgt_inter->1) "
          "yet roiP@-2 stays high and strict_removal doesn't move -> later layers reconstruct the object through "
          "the residual stream; deep supervision doesn't reach the layer the decoder reads either.")


if __name__ == "__main__":
    main()
