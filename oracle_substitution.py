"""Oracle test: is object-region-confined removal even SUFFICIENT to conceal the object
from LLaVA? No optimization, no pixel cost, no fidelity gap.

Run the clean image through LLaVA's OWN vision tower; at block 23 (hidden_states[-2],
exactly the layer the decoder reads) replace ONLY the object-region tokens with the
INPAINT image's tokens at the same grid positions; feed the hybrid to LLaVA and generate.
This is perfect object-region removal -- the ceiling any region-confined attack could reach.

  * If LLaVA still names the object -> NO region-confined attack can succeed; the ceiling is
    region confinement + context leakage (surrounding/boundary tokens carry the object).
    Corrected design must expand the region or drop the region confinement (whole-image delta).
  * If it removes the object -> the target is achievable and prior attacks fell short in
    fidelity (proj=0.02 etc.), not in principle.

Region variants test the "expand the region" axis and the <0.3-coverage boundary-token point:
  full    = substitute ALL 576 tokens  (SANITY: must reproduce the inpaint-image result -> hook works)
  tight   = coverage > 0.3             (the confinement used in prior runs)
  any     = coverage > 0               (every token the object touches at all)
  dilate2 = (coverage>0.3) grown by 2 token rings (adds the silhouette/boundary + margin)

Substitution is at hidden_states[vision_feature_layer]; the projector is per-token, so patching
pre-projector on the object rows == patching the projected object rows. LLaVA only reads this one
layer, so this is the entire visual input for those positions. Uses LLaVA's own vision tower (not a
separate CLIP) so features are exactly what the decoder consumes.

    uv run python oracle_substitution.py --dataset dataset.jsonl --objects dog,cat,car
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine"], "cat": ["cat", "kitten", "feline"],
       "car": ["car", "vehicle", "sedan", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
CAT = {"dog": "animal", "cat": "animal", "car": "vehicle", "airplane": "aircraft", "sign": "sign"}
NEG_CUES = ["no ", "not ", "n't", "without", "cannot", "can not", "none", "absent",
            "missing", "lack", "empty of", "free of", "devoid", "there is no",
            "there are no", "do not see", "don't see", "no sign"]


def _clauses(text):
    return re.split(r"[.,;:!?]| but | and | although | though | however ", text.lower())


def present(text, obj):
    syns = SYN.get(obj, [obj])
    return any(any(sy in cl for sy in syns) and not any(neg in cl for neg in NEG_CUES) for cl in _clauses(text))


def dilate(grid, k):
    g = grid.copy()
    for _ in range(k):
        p = g.copy()
        p[1:, :] |= g[:-1, :]; p[:-1, :] |= g[1:, :]
        p[:, 1:] |= g[:, :-1]; p[:, :-1] |= g[:, 1:]
        g = p
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat,car")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--variants", default="full,tight,any,dilate2")
    ap.add_argument("--outdir", default="results/oracle")
    args = ap.parse_args()
    RES = args.res

    model = AutoModelForImageTextToText.from_pretrained(args.victim, torch_dtype=DTYPE, device_map=DEVICE).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.victim)
    vt = getattr(model, "vision_tower", None) or getattr(model.model, "vision_tower", None)
    vdtype = next(vt.parameters()).dtype
    layer = getattr(model.config, "vision_feature_layer", -2)
    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    PATCH = vt.config.patch_size
    GRID = RES // PATCH
    NL = vt.config.num_hidden_layers
    block = NL + layer if layer < 0 else layer          # hidden_states[layer] = output of this block
    print(f"victim={args.victim}; vision_feature_layer={layer} (block {block} of {NL}) select={strategy}; "
          f"grid={GRID} ({GRID*GRID} patch tokens)")
    print(f"substitute clean object-region tokens <- inpaint tokens at hidden_states[{layer}] (block {block}) "
          f"= exactly the layer LLaVA reads")
    assert strategy == "default", f"expected select=default (CLS dropped); got {strategy}"

    def gen(img_pil, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inp = processor(images=img_pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=48, do_sample=False)
        return processor.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def load_pil(p):
        return Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)

    def inpaint_tokens(inpaint_pil):
        ipv = processor.image_processor(inpaint_pil, return_tensors="pt").pixel_values.to(DEVICE, vdtype)
        with torch.no_grad():
            out = vt(ipv, output_hidden_states=True)
        return out.hidden_states[layer]        # (1, 1+P, D)

    def make_hook(obj_1d, inp_hs):
        def hook(mod, inp, out):
            if not hasattr(out, "hidden_states") or out.hidden_states is None:
                raise RuntimeError("vision tower returned no hidden_states; cannot substitute")
            hs = list(out.hidden_states)
            patched = hs[layer].clone()
            sub = patched[:, 1:, :]                                  # (1, P, D) view over patched
            sub[:, obj_1d, :] = inp_hs[:, 1:, :][:, obj_1d, :].to(sub.dtype)
            hs[layer] = patched
            out.hidden_states = tuple(hs)
            return out
        return hook

    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    variants = args.variants.split(",")
    os.makedirs(args.outdir, exist_ok=True)
    ans_rows = [["object", "condition", "n_tokens", "prompt", "answer_text", "present"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/inpaint; skip"); continue
        clean_pil, inpaint_pil = load_pil(s["image"]), load_pil(s["target"])
        prompts = {"describe": "Describe this image.",
                   "direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
                   "list": "List all objects you see in this image.",
                   "presup": f"What {CAT.get(obj, 'object')} is in this image?",
                   "detail": "Describe this image in detail."}

        # sanity 1: inpaint image must actually remove the object (else the oracle target is invalid)
        inp_caps = {pk: gen(inpaint_pil, pt) for pk, pt in prompts.items()}
        inp_strict = any(present(c, obj) for c in inp_caps.values())
        for pk, c in inp_caps.items():
            ans_rows.append([obj, "inpaint_image", GRID * GRID, pk, c.replace("\n", " ")[:200], present(c, obj)])
        # sanity 2: clean image must show the object
        clean_caps = {pk: gen(clean_pil, pt) for pk, pt in prompts.items()}
        clean_strict = any(present(c, obj) for c in clean_caps.values())
        for pk, c in clean_caps.items():
            ans_rows.append([obj, "clean_image", 0, pk, c.replace("\n", " ")[:200], present(c, obj)])
        print(f"\n=== {sid} '{obj}' | clean_shows_obj={clean_strict} inpaint_removes_obj={not inp_strict} ===")
        if inp_strict:
            print(f"  [!] inpaint image STILL shows '{obj}' -> invalid oracle target; substitution can't beat it. "
                  f"(direct={inp_caps['direct'][:40]!r})")

        inp_hs = inpaint_tokens(inpaint_pil)

        # object coverage per token
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        cov = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))     # (GRID, GRID)
        region_grids = {"full": np.ones_like(cov, bool), "tight": cov > 0.3, "any": cov > 0,
                        "dilate1": dilate(cov > 0.3, 1), "dilate2": dilate(cov > 0.3, 2)}

        for v in variants:
            grid_bool = region_grids[v]
            obj_1d = torch.from_numpy(grid_bool.reshape(-1)).to(DEVICE)
            ntok = int(obj_1d.sum())
            h = vt.register_forward_hook(make_hook(obj_1d, inp_hs))
            try:
                strict_present = False
                for pk, pt in prompts.items():
                    a = gen(clean_pil, pt); p = present(a, obj); strict_present = strict_present or p
                    ans_rows.append([obj, f"oracle_{v}", ntok, pk, a.replace("\n", " ")[:200], p])
            finally:
                h.remove()
            print(f"  oracle_{v:<8} ({ntok:>3}/{GRID*GRID} tokens): object_removed={not strict_present}"
                  f"{'   <- SANITY: should match inpaint_removes' if v == 'full' else ''}")

    with open(f"{args.outdir}/answers.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(ans_rows)

    # summary
    from collections import defaultdict
    strict = defaultdict(lambda: {"present": False, "ntok": 0})
    for r in ans_rows[1:]:
        obj, cond, ntok, prompt, txt, pres = r
        strict[(obj, cond)]["present"] |= bool(pres); strict[(obj, cond)]["ntok"] = ntok
    summ = [["object", "condition", "n_tokens_substituted", "object_removed_strict"]]
    print("\n=== ORACLE: does perfect region-confined token substitution remove the object? ===")
    print(f"{'object':<6}{'condition':<16}{'tokens':>7}{'removed(strict)':>17}")
    for (obj, cond) in sorted(strict.keys()):
        removed = not strict[(obj, cond)]["present"]
        summ.append([obj, cond, strict[(obj, cond)]["ntok"], removed])
        print(f"{obj:<6}{cond:<16}{strict[(obj, cond)]['ntok']:>7}{str(removed):>17}")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)
    print(f"\nsaved {args.outdir}/{{answers,summary}}.csv")
    print("READ: oracle_full must == inpaint_image removal (hook works). Then: if oracle_tight/any/dilate2 do NOT "
          "remove the object while oracle_full does -> region confinement + context leakage is the ceiling; no "
          "region-confined attack can succeed; expand the region or go whole-image. If even any/dilate2 remove it, "
          "the achievable target exists and prior attacks fell short in fidelity, not in principle.")


if __name__ == "__main__":
    main()
