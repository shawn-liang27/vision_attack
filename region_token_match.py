"""Stage 24: region-level PATCH-TOKEN matching to a dog-free inpaint target,
graded end-to-end on LLaVA-1.5 under held-out prompts.

The one untested objective: match the object-region patch tokens -- at the exact
layer LLaVA's projector consumes -- to the position-corresponding tokens of the
inpainted (object-removed) target, while preserving background tokens.

  L = mean_{i in obj} ||H_adv[i] - H_inpaint[i]||^2
    + lambda * mean_{j in bg} ||H_adv[j] - H_source[j]||^2

Attacks LLaVA-1.5's OWN vision tower (hook VERIFIED against get_image_features,
printed), whole-image L_inf perturbation, MI-FGSM, eps sweep, 5 seeds.

GRADING (the part that produced false positives before): generate text under
multiple HELD-OUT prompts (describe / list / presupposition / direct / read-text),
log the FULL caption, grade object-absent across ALL prompts (conjunction).
Keyword flag is best-effort; the text dump is authoritative (watch noun
substitution car->truck and reality-stripping dog->statue).

Companion: pixel-interpolation flip-point -- how much real change removal costs.

    uv run python region_token_match.py --dataset dataset.jsonl --budgets 8,16,32,64,128 --seeds 5
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine", "pet", "animal"],
       "cat": ["cat", "kitten", "feline", "pet", "animal"],
       "car": ["car", "vehicle", "truck", "sedan", "suv", "automobile"],
       "airplane": ["airplane", "plane", "aircraft", "jet", "airliner"],
       "plane": ["plane", "airplane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
CAT = {"dog": "animal", "cat": "animal", "car": "vehicle", "airplane": "aircraft",
       "plane": "aircraft", "sign": "sign"}
STRIP = ["statue", "sculpture", "figurine", "toy", "drawing", "painting", "cartoon"]


def present(text, obj):
    t = text.lower()
    return any(s in t for s in SYN.get(obj, [obj]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--budgets", default="8,16,32,64,128")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam-preserve", type=float, default=1.0)
    ap.add_argument("--roi-thresh", type=float, default=0.3)
    ap.add_argument("--outdir", default="results/region_token")
    args = ap.parse_args()
    RES = args.res

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map=DEVICE).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)

    cfg = model.config
    VFL = getattr(cfg, "vision_feature_layer", -2)
    VFS = getattr(cfg, "vision_feature_select_strategy", "default")
    vt = getattr(model, "vision_tower", None) or model.model.vision_tower
    PATCH = vt.config.patch_size
    GRID = RES // PATCH
    print(f"HOOK: vision_feature_layer={VFL}  select_strategy={VFS}  patch={PATCH} grid={GRID} "
          f"(tokens={GRID*GRID}); NO post-layernorm applied (raw hidden state)")

    def penult(x01, grad=False):
        """RAW patch tokens at LLaVA's hook: hidden_states[VFL], CLS dropped if 'default'."""
        pv = ((x01 - MEAN) / STD).to(DTYPE)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = vt(pv, output_hidden_states=True)
            feat = out.hidden_states[VFL]
            if VFS == "default":
                feat = feat[:, 1:]
        return feat.squeeze(0)  # (GRID*GRID, D)

    # --- verify the hook matches what the projector actually consumes ---------
    with torch.no_grad():
        probe = torch.rand(1, 3, RES, RES, device=DEVICE, dtype=DTYPE)
        mine = model.model.multi_modal_projector(penult(probe).unsqueeze(0))
        try:
            ref = model.get_image_features(pixel_values=((probe - MEAN) / STD),
                                           vision_feature_layer=VFL,
                                           vision_feature_select_strategy=VFS)
            ref = ref[0] if isinstance(ref, (list, tuple)) else ref
            diff = (mine - ref).abs().max().item()
            print(f"HOOK VERIFY: max|proj(my_tokens) - get_image_features| = {diff:.2e} "
                  f"({'MATCH' if diff < 1e-2 else 'MISMATCH -- investigate'})")
        except Exception as e:
            print(f"HOOK VERIFY skipped ({type(e).__name__}: {e}); using documented path")

    os.makedirs(args.outdir, exist_ok=True)
    sqdir = os.path.join(args.outdir, "square"); os.makedirs(sqdir, exist_ok=True)

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    def load01(path):
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)

    def prompts_for(obj):
        p = {"describe": "Describe this image in detail.",
             "list": "List all the objects you see in this image.",
             "presup": f"What {CAT.get(obj,'object')} is in this image?",
             "direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'."}
        if obj == "sign":
            p["read"] = "Read any text visible in the image."
        return p

    def generate(x01, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        img = Image.fromarray((x01.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255)
                              .round().astype(np.uint8))
        inp = processor(images=img, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**{k: v for k, v in inp.items()}, max_new_tokens=64, do_sample=False)
        return tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = [["object", "eps", "seed", "prompt", "answer_text", "object_present", "maybe_stripped"]]
    curves = [["object", "eps", "seed", "iter", "L_match", "L_preserve", "cos_obj_tgt"]]
    interp_rows = [["object", "target_says_no", "flip_alpha", "roi_linf_at_flip_/255", "max_roi_linf_/255"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/target; skip"); continue
        x_src, x_tgt = load01(s["image"]), load01(s["target"])
        H_src = penult(x_src).detach()
        H_tgt = penult(x_tgt).detach()
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        frac = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)).reshape(-1)
        obj_idx = torch.from_numpy(frac > args.roi_thresh).to(DEVICE)
        bg_idx = ~obj_idx
        if obj_idx.sum() == 0:
            print(f"[{sid}] empty object token set; skip"); continue
        prompts = prompts_for(obj)
        print(f"\n=== {sid} '{obj}' === object tokens={int(obj_idx.sum())}/{GRID*GRID}")

        # ---- companion: pixel-interpolation flip point -----------------------
        with torch.no_grad():
            tgt_no = not present(generate(x_tgt, prompts["direct"]), obj) and \
                     not present(generate(x_tgt, prompts["describe"]), obj)
            roi_pix = torch.from_numpy((mg > 0.5)).to(DEVICE)
            max_linf = ((x_tgt - x_src).abs().squeeze(0).amax(0)[roi_pix].max().item()) * 255
            flip_a, flip_linf = None, None
            for a in np.linspace(0, 1, 11):
                xa = a * x_tgt + (1 - a) * x_src
                if not present(generate(xa, prompts["direct"]), obj) and \
                   not present(generate(xa, prompts["describe"]), obj):
                    flip_a = round(float(a), 2); flip_linf = round(a * max_linf, 1); break
        interp_rows.append([obj, tgt_no, flip_a, flip_linf, round(max_linf, 1)])
        print(f"  interp: target_says_no={tgt_no} flip_alpha={flip_a} "
              f"roi_linf@flip={flip_linf}/255 (max_roi_linf={max_linf:.0f}/255)")

        # ---- attack sweep ----------------------------------------------------
        for b in [float(x) for x in args.budgets.split(",")]:
            eps = b / 255.0
            alpha = max(1.0 / 255, 2.5 * eps / args.steps)
            for seed in range(args.seeds):
                g = torch.Generator(device=DEVICE).manual_seed(1234 + seed)
                delta = (torch.rand(x_src.shape, generator=g, device=DEVICE, dtype=DTYPE) * 2 - 1) * eps
                delta = delta.detach().requires_grad_(True)
                mom = torch.zeros_like(x_src)
                for it in range(args.steps):
                    x = torch.clamp(x_src + delta, 0, 1)
                    H = penult(x, grad=True)
                    Lm = ((H[obj_idx] - H_tgt[obj_idx]) ** 2).mean()
                    Lp = ((H[bg_idx] - H_src[bg_idx]) ** 2).mean()
                    L = Lm + args.lam_preserve * Lp
                    gr, = torch.autograd.grad(L, delta)
                    with torch.no_grad():
                        mom.mul_(0.9).add_(gr / gr.abs().mean().clamp_min(1e-12))
                        delta.add_(-alpha * mom.sign()).clamp_(-eps, eps)
                        delta.data = torch.clamp(x_src + delta, 0, 1) - x_src
                    delta.requires_grad_(True)
                    if it % 30 == 0 or it == args.steps - 1:
                        with torch.no_grad():
                            co = F.cosine_similarity(H[obj_idx].float(), H_tgt[obj_idx].float(), dim=-1).mean().item()
                        curves.append([obj, b, seed, it, round(Lm.item(), 4), round(Lp.item(), 4), round(co, 4)])

                adv = torch.clamp(x_src + delta.detach(), 0, 1)
                if seed == 0:
                    Image.fromarray((adv.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255)
                                    .round().astype(np.uint8)).save(f"{sqdir}/{sid}_eps{int(b)}.png")
                # ---- GRADE under held-out prompts (full text) ----------------
                any_present = False
                for pk, pt in prompts.items():
                    ans = generate(adv, pt)
                    pres = present(ans, obj)
                    strip = any(w in ans.lower() for w in STRIP)
                    any_present = any_present or pres
                    rows.append([obj, b, seed, pk, ans.replace("\n", " ")[:200], pres, strip])
                print(f"  eps={b:>4.0f} seed={seed} concealed(all prompts)={not any_present}")

    with open(f"{args.outdir}/answers.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(f"{args.outdir}/curves.csv", "w", newline="") as f:
        csv.writer(f).writerows(curves)
    with open(f"{args.outdir}/interpolation.csv", "w", newline="") as f:
        csv.writer(f).writerows(interp_rows)

    # ---- summary: conjunction concealment rate per (object, eps) -------------
    from collections import defaultdict
    grp = defaultdict(lambda: defaultdict(list))  # (obj,eps) -> seed -> [present per prompt]
    for r in rows[1:]:
        grp[(r[0], r[1])][r[2]].append(r[5])
    print("\n=== conjunction concealment rate (object absent under ALL prompts) ===")
    print(f"{'object':<12}{'eps':>6}{'conceal% (n seeds)':>22}")
    summ = [["object", "eps", "conceal_rate", "n_seeds"]]
    for (obj, eps), seeds in sorted(grp.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        rates = [not any(pl) for pl in seeds.values()]   # concealed if no prompt shows object
        rate = 100 * np.mean(rates)
        summ.append([obj, eps, round(rate, 1), len(rates)])
        print(f"{obj:<12}{eps:>6}{rate:>15.0f}% ({len(rates)})")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(summ)
    print(f"\nsaved {args.outdir}/{{answers,curves,interpolation,summary}}.csv + square/ images")
    print("NOTE: answers.csv text is authoritative; check for noun-substitution / statue-stripping "
          "the keyword flag misses. Weight conclusions on dog/cat/sign (car/airplane multi-instance).")


if __name__ == "__main__":
    main()
