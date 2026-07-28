"""Oracle test + dilation sweep: how large must a PERFECTLY-removed region be to conceal
the object from LLaVA? No optimization, no pixel cost, no fidelity gap.

Hook LLaVA's own vision tower; at block 23 (hidden_states[-2], the layer the decoder reads)
replace the region's tokens of the CLEAN image with the INPAINT image's tokens at the same
grid positions; generate. Sweep the region from the tight object mask outward by token-count
MULTIPLES {1.0, 1.25, 1.5, 2.0, 2.5, ...}x -- the nearest background tokens are added first
(distance-ordered), so each level is a superset of the last. Locates the region-size
threshold at which perfect removal finally succeeds.

  * tight (1.0x) already removes -> region confinement was never the barrier (attacks failed
    on fidelity, not region).
  * removal needs Kx expansion -> the object's spatial-neighbor (halo) tokens leak it; the
    corrected attack must steer the DILATED region and must NOT pin the halo band clean.
  * even 2.5x / full fails -> context leakage is global; region expansion is insufficient.
  * oracle_full == inpaint result is the hook SANITY check.

PROBE FIX: the presupposition prompt ("What <cat> is in this image?") is dropped -- LLaVA-1.5
accepts the presupposition and answers "A car is in this image" even on an empty image
(sycophancy, not detection), which false-positives every image. Probes are now all
non-leading: describe / is-there(yes-no) / list / detail / count / main-subject. Grading is
clause-level negation-aware; generic terms (animal/pet/vehicle) are NOT object synonyms.

    uv run python oracle_substitution.py --dataset dataset.jsonl --objects dog \
        --dilations 1.0,1.25,1.5,2.0,2.5
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
       "car": ["car", "sedan", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
NEG_CUES = ["no ", "not ", "n't", "without", "cannot", "can not", "none", "absent",
            "missing", "lack", "empty of", "free of", "devoid", "there is no",
            "there are no", "do not see", "don't see", "no sign", "zero"]


def _clauses(text):
    return re.split(r"[.,;:!?]| but | and | although | though | however ", text.lower())


def present(text, obj):
    """object asserted present -- clause-level, negation-aware. No leading/generic terms."""
    syns = SYN.get(obj, [obj])
    for cl in _clauses(text):
        if any(sy in cl for sy in syns):
            if any(neg in cl for neg in NEG_CUES) or re.search(r"\b0\b", cl):  # incl. "0 dogs" count
                continue
            return True
    return False


def dilate(grid, k=1):
    g = grid.copy()
    for _ in range(k):
        p = g.copy()
        p[1:, :] |= g[:-1, :]; p[:-1, :] |= g[1:, :]
        p[:, 1:] |= g[:, :-1]; p[:, :-1] |= g[:, 1:]
        g = p
    return g


def ring_order(base_grid):
    """token indices ordered by grid-distance from the object (object tokens first)."""
    ring = np.full(base_grid.shape, 1 << 20, dtype=np.int64)
    ring[base_grid] = 0
    cur = base_grid.copy(); r = 0
    while not cur.all():
        r += 1
        nxt = dilate(cur, 1)
        newly = nxt & (~cur)
        if not newly.any():
            break
        ring[newly] = r
        cur = nxt
    return np.argsort(ring.reshape(-1), kind="stable")   # stable: ring 0 (object) first, then nearest halo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--dilations", default="1.0,1.25,1.5,2.0,2.5", help="region size as multiples of the tight object-token count")
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
    block = NL + layer if layer < 0 else layer
    print(f"victim={args.victim}; vision_feature_layer={layer} (block {block} of {NL}) select={strategy}; "
          f"grid={GRID} ({GRID*GRID} tokens); substitute clean<-inpaint at hidden_states[{layer}] = the layer LLaVA reads")
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
            return vt(ipv, output_hidden_states=True).hidden_states[layer]   # (1, 1+P, D)

    def make_hook(obj_1d, inp_hs):
        def hook(mod, inp, out):
            if not hasattr(out, "hidden_states") or out.hidden_states is None:
                raise RuntimeError("vision tower returned no hidden_states")
            hs = list(out.hidden_states)
            patched = hs[layer].clone()
            sub = patched[:, 1:, :]
            sub[:, obj_1d, :] = inp_hs[:, 1:, :][:, obj_1d, :].to(sub.dtype)
            hs[layer] = patched
            out.hidden_states = tuple(hs)
            return out
        return hook

    dilations = [float(x) for x in args.dilations.split(",") if x.strip()]
    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    ans_rows = [["object", "condition", "mult", "n_tokens", "prompt", "answer_text", "present"]]

    def probes(obj):
        return {"describe": "Describe this image.",
                "direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
                "list": "List all objects you see in this image.",
                "detail": "Describe this image in detail.",
                "count": f"How many {obj}s are in this image? If none, answer 0.",
                "subject": "What is the main subject of this image?"}

    def run(obj, img_pil, cond, mult, ntok, prompt_set):
        strict = False
        for pk, pt in prompt_set.items():
            a = gen(img_pil, pt); p = present(a, obj); strict = strict or p
            ans_rows.append([obj, cond, mult, ntok, pk, a.replace("\n", " ")[:200], p])
        return strict

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/inpaint; skip"); continue
        clean_pil, inpaint_pil = load_pil(s["image"]), load_pil(s["target"])
        pr = probes(obj)

        clean_present = run(obj, clean_pil, "clean_image", 0, 0, pr)
        inpaint_present = run(obj, inpaint_pil, "inpaint_image", -1, GRID * GRID, pr)
        print(f"\n=== {sid} '{obj}' | clean_shows={clean_present}  inpaint_removes={not inpaint_present} "
              f"(presup dropped) ===")
        if inpaint_present:
            print(f"  [!] inpaint image STILL shows '{obj}' under a non-leading probe -> invalid oracle target")

        inp_hs = inpaint_tokens(inpaint_pil)
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        base = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3     # tight object tokens
        n0 = int(base.sum())
        order = ring_order(base)

        # full-substitution sanity (must reproduce inpaint removal)
        full_1d = torch.ones(GRID * GRID, dtype=torch.bool, device=DEVICE)
        h = vt.register_forward_hook(make_hook(full_1d, inp_hs))
        try:
            full_present = run(obj, clean_pil, "oracle_full", -1, GRID * GRID, pr)
        finally:
            h.remove()
        print(f"  oracle_full ({GRID*GRID} tokens): removed={not full_present}  <- SANITY, should match inpaint_removes")

        # dilation sweep: nearest-first token-count multiples of the tight region
        print(f"  tight object tokens n0={n0}; sweeping region size as multiples of n0:")
        for m in dilations:
            T = min(GRID * GRID, max(n0, round(m * n0)))
            sel = np.zeros(GRID * GRID, bool); sel[order[:T]] = True
            obj_1d = torch.from_numpy(sel).to(DEVICE)
            h = vt.register_forward_hook(make_hook(obj_1d, inp_hs))
            try:
                strict = run(obj, clean_pil, f"oracle_{m:g}x", m, T, pr)
            finally:
                h.remove()
            print(f"    {m:>4g}x  ({T:>3}/{GRID*GRID} tokens, {T/max(n0,1):.2f}x actual): removed={not strict}")

    with open(f"{args.outdir}/answers.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(ans_rows)

    # summary + per-probe breakdown
    from collections import defaultdict
    strict = defaultdict(lambda: {"present": False, "ntok": 0, "mult": None, "probes": {}})
    for r in ans_rows[1:]:
        obj, cond, mult, ntok, prompt, txt, pres = r
        c = strict[(obj, cond)]
        c["present"] |= bool(pres); c["ntok"] = ntok; c["mult"] = mult
        c["probes"][prompt] = bool(pres)
    probe_order = ["describe", "direct", "list", "detail", "count", "subject"]
    summ = [["object", "condition", "mult", "n_tokens", "removed_strict"] + probe_order]
    print("\n=== ORACLE dilation sweep: does perfect region-confined removal conceal the object? ===")
    print(f"{'object':<5}{'condition':<15}{'tokens':>7}{'removed':>9}   per-probe present (D/Dir/L/Det/C/S)")
    for (obj, cond) in sorted(strict.keys(), key=lambda k: (k[0], str(k[1]))):
        c = strict[(obj, cond)]
        removed = not c["present"]
        pv = [c["probes"].get(p, "") for p in probe_order]
        summ.append([obj, cond, c["mult"], c["ntok"], removed] + pv)
        marks = " ".join("X" if c["probes"].get(p) else "." for p in probe_order)
        print(f"{obj:<5}{cond:<15}{c['ntok']:>7}{str(removed):>9}   {marks}")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)
    print(f"\nsaved {args.outdir}/{{answers,summary}}.csv")
    print("READ: find the smallest Kx where removed=True. tight(1.0x) removes -> region was never the barrier. "
          "Needs Kx -> the halo band leaks; corrected attack steers the Kx region and drops L_fix on the halo. "
          "Even full fails -> context leakage is global (region expansion won't fix it). oracle_full must == inpaint.")


if __name__ == "__main__":
    main()
