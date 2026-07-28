"""Stage 28: single-layer steering sweep -- does attacking EARLIER CLIP layers conceal
the object better than attacking the layer LLaVA reads (block 23)?

Hypothesis: shallow layers are less globally entangled and are the residual source
that propagates downstream, so corrupting the object there may conceal it better than
corrupting block 23 directly.

ATTACK (minimal -- no AdvEDM terms): delta over the WHOLE image, ||delta||_inf <= eps/255.
Loss = ONLY steer the object-region patch tokens toward the inpaint image's patch tokens
at ONE layer L:
    loss(delta) = - mean_{i in obj} cos( hidden[L][i](x0+delta), hidden[L][i](inpaint) )
No L_cls, no L_fix. One layer per arm; baseline = block 23 (hidden_states[-2], what the
LLaVA decoder reads). Source patches = segmentation-mask object tokens; target = the same
layer of the inpainted ("object removed") image, frozen.

CONVENTION: 'layer L' = output of the L-th transformer block (post attn+MLP+residual) =
hidden_states[L], 1-indexed (hidden_states[0]=embeddings; CLIP ViT-L/14-336 has 24 blocks).
LLaVA reads hidden_states[-2] = block 23.

Budgets L_inf in {8,16,32}/255. Objects dog,cat (car etc. via --objects). 5 seeds (delta
init only; greedy decode -> error bars are init variance). Metric: STRICT removal % = object
detected in NONE of {2 yes/no probes, describe, list, detail, count}. CORRECTED SCORING:
yes/no probes are PARSED (a bare "Yes" has no noun to keyword-match -- the direct probe was
dead in prior runs); presup dropped (sycophancy false-positive); clause-level negation-aware
incl. "0 dogs". Optimizer Adam lr=5e-3, 500 iters (carried from prior runs).

    uv run python layer_sweep.py --dataset dataset.jsonl --objects dog,cat \
        --layers 2,4,6,8,12,16,20,23 --eps 8,16,32 --seeds 5
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine"], "cat": ["cat", "kitten", "feline"],
       "car": ["car", "sedan", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
CAT = {"dog": "animal", "cat": "animal", "car": "vehicle", "airplane": "aircraft", "sign": "sign"}
NEG_CUES = ["no ", "not ", "n't", "without", "cannot", "can not", "none", "absent",
            "missing", "lack", "empty of", "free of", "devoid", "there is no",
            "there are no", "do not see", "don't see", "no sign", "zero"]
STRIP = ["statue", "stuffed", "figurine", "sculpture", " toy", "plush", "cartoon",
         "drawing", "painting", "poster", "mural", "cardboard", "plastic"]
DEGEN = ["blur", "blurr", "noise", "noisy", "static", "pixelat", "distort", "glitch",
         "abstract", "colorful pattern", "pattern of color", "corrupt", "unclear",
         "difficult to", "hard to tell", "can't tell", "not clear"]


def _clauses(text):
    return re.split(r"[.,;:!?]| but | and | although | though | however ", text.lower())


def present(text, obj):
    """object asserted present -- clause-level, negation-aware."""
    syns = SYN.get(obj, [obj])
    for cl in _clauses(text):
        if any(sy in cl for sy in syns):
            if any(neg in cl for neg in NEG_CUES) or re.search(r"\b0\b", cl):
                continue
            return True
    return False


def yesno(answer):
    a = answer.strip().lower()
    m = re.match(r"[^a-z]*([a-z]+)", a)
    first = m.group(1) if m else ""
    if first in ("yes", "yeah", "yep", "correct", "true", "sure"):
        return True
    if first in ("no", "nope", "none", "false"):
        return False
    hy, hn = re.search(r"\byes\b", a), re.search(r"\bno\b", a)
    if hy and not hn:
        return True
    if hn and not hy:
        return False
    return None


def detect(answer, obj, is_yesno):
    """yes/no probes are PARSED (a bare 'Yes' has no noun to keyword-match); open probes keyword-matched."""
    if is_yesno:
        v = yesno(answer)
        if v is not None:
            return v
    return present(answer, obj)


def reality_stripped(text, obj):
    syns = SYN.get(obj, [obj])
    return any(any(sy in cl for sy in syns) and any(st in cl for st in STRIP) for cl in _clauses(text))


def degenerate(text):
    return any(d in text.lower() for d in DEGEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--layers", default="2,4,6,8,12,16,20,23", help="single attacked layer per arm (hidden_states idx)")
    ap.add_argument("--eps", default="8,16,32", help="L_inf budgets in /255")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--outdir", default="results/layer_sweep")
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()
    clip.requires_grad_(False)
    cproc = CLIPProcessor.from_pretrained(args.surrogate)
    ipc = cproc.image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH
    NL = clip.config.vision_config.num_hidden_layers  # 24

    victim = AutoModelForImageTextToText.from_pretrained(args.victim, torch_dtype=DTYPE, device_map=DEVICE).eval()
    victim.requires_grad_(False)
    vproc = AutoProcessor.from_pretrained(args.victim)
    vtok = vproc.tokenizer

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    epslist = [float(e) for e in args.eps.split(",") if e.strip()]
    objects = [o.strip() for o in args.objects.split(",")]
    print(f"surrogate={args.surrogate} grid={GRID} blocks={NL}; victim={args.victim}")
    print(f"CONVENTION: layer L = output of block L = hidden_states[L], 1-indexed "
          f"(0=embeddings). LLaVA reads hidden_states[-2]=block {NL - 1} (the baseline).")
    print(f"layers={layers} (baseline=block {NL - 1}); eps(Linf)/255={epslist}; objects={objects}; "
          f"seeds={args.seeds} (init only); whole-image delta; loss=-cos(obj patches @L -> inpaint @L) only")

    def vis(x01):
        out = clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True, output_hidden_states=True)
        return out.hidden_states   # tuple len NL+1, each (1, 1+P, D)

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
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in objects]
    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "adv"); os.makedirs(imgdir, exist_ok=True)

    ans_rows = [["object", "eps", "layer", "seed", "criterion", "prompt", "answer_text",
                 "present", "reality_stripped", "degenerate", "linf"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not os.path.exists(s["mask"]):
            print(f"[{sid}] no mask; skip"); continue
        if not (s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] no inpaint target; skip"); continue
        x0 = load01(s["image"])
        xR = load01(s["target"])

        # E4 validity gate: the inpaint image must not itself still show the object
        ref_dir, ref_desc = gen(xR, f"Is there a {obj} in this image? Answer with only 'yes' or 'no'."), gen(xR, "Describe this image.")
        for pk, cap in [("direct", ref_dir), ("describe", ref_desc)]:
            ans_rows.append([obj, 0, "inpaint_ref", 0, "ref", pk, cap.replace("\n", " ")[:200],
                             present(cap, obj), reality_stripped(cap, obj), degenerate(cap), 0])
        if detect(ref_dir, obj, True) or present(ref_desc, obj):
            print(f"[{sid}] INPAINT STILL SHOWS '{obj}' (dir={ref_dir[:40]!r}) -> unachievable target; SKIP"); continue

        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        obj_idx = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)
        with torch.no_grad():
            hI = vis(xR)
            featI = {L: F.normalize(hI[L][:, 1:].squeeze(0), dim=-1) for L in layers}  # frozen inpaint targets
        print(f"\n=== {sid} '{obj}' ({int(obj_idx.sum())}/{GRID * GRID} obj tokens); inpaint-ref ok ===")

        # (key, prompt, is_yesno). presup dropped (sycophancy false-positive); direct is yes/no-parsed.
        prompts = [("direct1", f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                   ("direct2", f"Do you see a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                   ("describe", "Describe this image.", False),
                   ("list", "List all objects you see in this image.", False),
                   ("detail", "Describe this image in detail.", False),
                   ("count", f"How many {obj}s are in this image? If none, answer 0.", False)]

        for eps in epslist:
            e01 = eps / 255.0
            for L in layers:
                for seed in range(args.seeds):
                    g = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
                    delta = ((torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * e01).detach().requires_grad_(True)
                    opt = torch.optim.Adam([delta], lr=args.lr)
                    for it in range(args.iters):
                        x = torch.clamp(x0 + delta, 0, 1)
                        tok = F.normalize(vis(x)[L][:, 1:].squeeze(0), dim=-1)
                        loss = -(tok[obj_idx] * featI[L][obj_idx]).sum(-1).mean()   # steer obj patches -> inpaint @L
                        opt.zero_grad(); loss.backward(); opt.step()
                        with torch.no_grad():
                            delta.clamp_(-e01, e01)                                  # L_inf ball
                            delta.data = torch.clamp(x0 + delta, 0, 1) - x0          # keep adv in [0,1]

                    adv = torch.clamp(x0 + delta.detach(), 0, 1)
                    linf = round(delta.detach().abs().max().item() * 255, 1)
                    if seed == 0:
                        Image.fromarray((adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                            f"{imgdir}/{obj}_eps{eps:g}_L{L}_s0.png")
                    strict_present = False
                    for pk, pt, yn in prompts:
                        a = gen(adv, pt); p = detect(a, obj, yn); strict_present = strict_present or p
                        crit = "weak" if pk == "describe" else "strict"
                        ans_rows.append([obj, eps, L, seed, crit, pk, a.replace("\n", " ")[:200],
                                         p, reality_stripped(a, obj), degenerate(a), linf])
                    print(f"  {obj} eps{eps:g} L{L:<2} s{seed} Linf={linf:>4} | strict_absent={not strict_present}")

    with open(f"{args.outdir}/answers.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(ans_rows)

    # ---- summary: strict/weak removal % per (object, eps, layer) ----
    from collections import defaultdict
    cell = defaultdict(lambda: defaultdict(lambda: {"weak": False, "strict": False, "strip": False, "degen": False, "linf": 0.0}))
    for r in ans_rows[1:]:
        obj, eps, layer, seed, crit, prompt, txt, pres, strip, degen, linf = r
        if layer == "inpaint_ref":
            continue
        c = cell[(obj, float(eps), int(layer))][seed]
        if crit == "weak":
            c["weak"] = bool(pres); c["linf"] = float(linf)
        c["strict"] = c["strict"] or bool(pres)     # present in ANY probe (incl. describe)
        c["strip"] = c["strip"] or bool(strip); c["degen"] = c["degen"] or bool(degen)

    summ = [["object", "eps", "layer", "weak_removal_rate", "strict_removal_rate", "mean_linf",
             "reality_stripped_rate", "degen_of_removed_rate", "n"]]
    print(f"\n=== STRICT removal % vs attacked layer (baseline = block {NL - 1}) ===")
    print(f"{'object':<5}{'eps':>4}{'layer':>7}{'weak%':>8}{'strict%':>9}{'Linf':>6}{'strip%':>8}{'degenRem%':>11}")
    for key in sorted(cell.keys()):
        obj, eps, layer = key
        cs = cell[key].values()
        weak = 100 * np.mean([not c["weak"] for c in cs])
        strict = 100 * np.mean([not c["strict"] for c in cs])
        removed = [c for c in cs if not c["strict"]]
        degen_rem = 100 * np.mean([c["degen"] for c in removed]) if removed else 0.0
        strip = 100 * np.mean([c["strip"] for c in cs])
        mlinf = float(np.mean([c["linf"] for c in cs]))
        n = len(cell[key])
        summ.append([obj, eps, layer, round(weak, 1), round(strict, 1), round(mlinf, 1),
                     round(strip, 1), round(degen_rem, 1), n])
        tag = "  <- baseline" if layer == NL - 1 else ""
        print(f"{obj:<5}{eps:>4.0f}{layer:>7}{weak:>7.0f}%{strict:>8.0f}%{mlinf:>6.0f}{strip:>7.0f}%{degen_rem:>10.0f}%{tag}")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)

    # ---- plot: strict removal % vs layer, one panel per budget, dog & cat ----
    fig, axes = plt.subplots(1, len(epslist), figsize=(6 * len(epslist), 5), sharey=True, squeeze=False)
    colors = {"dog": "tab:blue", "cat": "tab:orange"}
    for ax, eps in zip(axes[0], epslist):
        for obj in objects:
            pts = sorted((layer, 100 * np.mean([not c["strict"] for c in cell[(obj, eps, layer)].values()]))
                         for (o, e, layer) in cell if o == obj and e == eps)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", color=colors.get(obj, "gray"), label=obj)
        ax.axvline(NL - 1, ls="-.", color="red", lw=1, label=f"baseline (block {NL - 1})")
        ax.set_title(f"L_inf = {eps:g}/255"); ax.set_xlabel("attacked block L"); ax.grid(alpha=0.3)
        ax.set_ylim(-5, 105)
    axes[0][0].set_ylabel("STRICT removal %"); axes[0][0].legend()
    fig.suptitle("Single-layer steering -> object patches to inpaint: strict removal vs attacked layer\n"
                 "hypothesis: earlier layers conceal better than the block-23 baseline")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/removal_vs_layer.png", dpi=130); plt.close(fig)

    print(f"\nsaved {args.outdir}/{{answers,summary}}.csv, removal_vs_layer.png, adv/*.png")
    print("READ: for each budget, does strict% rise as the attacked layer gets EARLIER (left of the red baseline)? "
          "Validity: degenRem%>0 => some 'removals' are destroyed images not concealment (check adv/*.png + captions).")


if __name__ == "__main__":
    main()
