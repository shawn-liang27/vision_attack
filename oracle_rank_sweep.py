"""Stage 32: DETECTION-relevant rank of object removal. The SVD spectrum (Stage 31) tells you
where the VARIANCE lives; this tells you where the DETECTION signal lives -- which can differ.

Sweep the rank of the removed subspace in the oracle. Over the 1.5x region, at the read layer:
    d_i    = h_clean[i] - h_inpaint[i]
    U_k    = top-k right singular vectors of {d_i} over the region   (object subspace)
    h_mod[i] = h_clean[i] - U_k U_k^T d_i        # remove only the top-k object-subspace component
  k=0        -> h_mod = h_clean            (no change; object present)
  k=full(=n) -> h_mod = h_inpaint          (exactly the working 1.5x oracle; object removed)
So k interpolates "no change" -> "full substitution", and the SMALLEST k where the object
disappears is the detection-relevant rank. Report the d-energy that top-k captures at each k
(= cumulative explained variance): if removal happens at small k with modest captured energy,
the detection signal is in the top few directions; if it needs large k, it's spread through the
tail. Even k~30 is a big win over point-matching -- zeroing 30 numbers/token, not landing a 1024-vector.

No optimization; substitute at hidden_states[-2] via the vision-tower hook. Corrected scoring
(yes/no probes parsed, presup dropped). 1.5x region (1x fails regardless). dog by default.

    uv run python oracle_rank_sweep.py --dataset dataset.jsonl --objects dog \
        --dilate 1.5 --ranks 1,3,5,10,15,22,38,68
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine"], "cat": ["cat", "kitten", "feline"],
       "car": ["car", "sedan", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
NEG_CUES = ["no ", "not ", "n't", "without", "cannot", "can not", "none", "absent", "missing",
            "lack", "empty of", "free of", "devoid", "there is no", "there are no",
            "do not see", "don't see", "no sign", "zero"]


def _clauses(t):
    return re.split(r"[.,;:!?]| but | and | although | though | however ", t.lower())


def present(text, obj):
    for cl in _clauses(text):
        if any(sy in cl for sy in SYN.get(obj, [obj])):
            if any(neg in cl for neg in NEG_CUES) or re.search(r"\b0\b", cl):
                continue
            return True
    return False


def yesno(answer):
    a = answer.strip().lower(); m = re.match(r"[^a-z]*([a-z]+)", a); first = m.group(1) if m else ""
    if first in ("yes", "yeah", "yep", "correct", "true", "sure"):
        return True
    if first in ("no", "nope", "none", "false"):
        return False
    hy, hn = re.search(r"\byes\b", a), re.search(r"\bno\b", a)
    return True if (hy and not hn) else False if (hn and not hy) else None


def detect(answer, obj, is_yesno):
    if is_yesno:
        v = yesno(answer)
        if v is not None:
            return v
    return present(answer, obj)


def dilate(grid, k=1):
    g = grid.copy()
    for _ in range(k):
        p = g.copy()
        p[1:, :] |= g[:-1, :]; p[:-1, :] |= g[1:, :]; p[:, 1:] |= g[:, :-1]; p[:, :-1] |= g[:, 1:]
        g = p
    return g


def ring_order(base):
    ring = np.full(base.shape, 1 << 20, np.int64); ring[base] = 0; cur = base.copy(); r = 0
    while not cur.all():
        r += 1; nxt = dilate(cur, 1); newly = nxt & (~cur)
        if not newly.any():
            break
        ring[newly] = r; cur = nxt
    return np.argsort(ring.reshape(-1), kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--dilate", type=float, default=1.5, help="region = this multiple of the tight object-token count")
    ap.add_argument("--ranks", default="1,3,5,10,15,22,38,68")
    ap.add_argument("--outdir", default="results/oracle_rank")
    args = ap.parse_args()
    RES = args.res

    model = AutoModelForImageTextToText.from_pretrained(args.victim, torch_dtype=DTYPE, device_map=DEVICE).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.victim)
    vt = getattr(model, "vision_tower", None) or getattr(model.model, "vision_tower", None)
    vdtype = next(vt.parameters()).dtype
    layer = getattr(model.config, "vision_feature_layer", -2)
    PATCH = vt.config.patch_size
    GRID = RES // PATCH
    print(f"victim={args.victim}; substitute at hidden_states[{layer}] (read layer); region={args.dilate}x; "
          f"h_mod = h_clean - U_k U_k^T d over region")

    def gen(img_pil, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inp = processor(images=img_pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=48, do_sample=False)
        return processor.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def load_pil(p):
        return Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)

    def tokens(pil):
        pv = processor.image_processor(pil, return_tensors="pt").pixel_values.to(DEVICE, vdtype)
        with torch.no_grad():
            return vt(pv, output_hidden_states=True).hidden_states[layer]   # (1, 1+P, D)

    def make_hook(region_1d, region_tokens):
        def hook(mod, inp, out):
            hs = list(out.hidden_states); patched = hs[layer].clone()
            sub = patched[:, 1:, :]
            sub[:, region_1d, :] = region_tokens.unsqueeze(0).to(sub.dtype)
            hs[layer] = patched; out.hidden_states = tuple(hs); return out
        return hook

    def probes(obj):
        return [("direct1", f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("direct2", f"Do you see a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("describe", "Describe this image.", False),
                ("list", "List all objects you see in this image.", False),
                ("detail", "Describe this image in detail.", False),
                ("count", f"How many {obj}s are in this image? If none, answer 0.", False)]

    ranks_req = [int(x) for x in args.ranks.split(",") if x.strip()]
    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    ans_rows = [["object", "k", "captured_energy", "prompt", "answer_text", "present"]]
    summ = [["object", "k", "captured_energy", "removed_strict"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/inpaint; skip"); continue
        clean_pil, inpaint_pil = load_pil(s["image"]), load_pil(s["target"])
        pr = probes(obj)
        # validity: inpaint must remove the object under a direct probe
        ref = gen(inpaint_pil, pr[0][1])
        if detect(ref, obj, True):
            print(f"[{sid}] inpaint still shows '{obj}' (ref={ref[:40]!r}); skip"); continue

        h_clean = tokens(clean_pil); h_inp = tokens(inpaint_pil)     # (1,577,1024)
        d_patch = (h_clean - h_inp)[0, 1:]                           # (576,1024)
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        base = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3
        n0 = int(base.sum()); order = ring_order(base)
        T = min(GRID * GRID, max(n0, round(args.dilate * n0)))
        sel = np.zeros(GRID * GRID, bool); sel[order[:T]] = True
        region_1d = torch.from_numpy(sel).to(DEVICE)
        D = d_patch[region_1d].float()                              # (n,1024) region object directions
        n = D.shape[0]
        _, _, Vh = torch.linalg.svd(D, full_matrices=False)         # Vh: (n,1024) right singular vectors
        Denergy = (D ** 2).sum().clamp_min(1e-12)
        h_clean_region = h_clean[0, 1:][region_1d].float()          # (n,1024)
        ks = [0] + [k for k in ranks_req if 0 < k < n] + [n]        # 0=clean, n=full(=inpaint oracle)
        print(f"\n=== {sid} '{obj}' region={T} tokens ({T/n0:.2f}x), rank range 0..{n} ===")

        first_removed = None
        for k in ks:
            B = Vh[:k]                                               # (k,1024) orthonormal rows
            proj = (D @ B.T) @ B if k > 0 else torch.zeros_like(D)   # U_k U_k^T d_i per region patch
            captured = (proj ** 2).sum().item() / Denergy.item()     # fraction of d-energy removed (= cum EV at k)
            h_mod_region = (h_clean_region - proj).to(vdtype)        # k=0 -> clean; k=n -> inpaint
            h = vt.register_forward_hook(make_hook(region_1d, h_mod_region))
            try:
                strict = False
                for pk, pt, yn in pr:
                    a = gen(clean_pil, pt); p = detect(a, obj, yn); strict = strict or p
                    ans_rows.append([obj, k, round(captured, 4), pk, a.replace("\n", " ")[:200], p])
            finally:
                h.remove()
            removed = not strict
            summ.append([obj, k, round(captured, 4), removed])
            if removed and k > 0 and first_removed is None:
                first_removed = (k, captured)
            tag = " <- k=0 clean (expect present)" if k == 0 else " <- k=full == 1.5x oracle (expect removed)" if k == n else ""
            print(f"  k={k:>3}  captured_d_energy={captured:5.2f}  removed={removed}{tag}")
        if first_removed:
            print(f"  => DETECTION-RELEVANT RANK: smallest k with removal = {first_removed[0]} "
                  f"(captures {first_removed[1]*100:.0f}% of d-energy)")
        else:
            print("  => no swept k removed it (signal spread through the tail; try higher ranks)")

    with open(f"{args.outdir}/answers.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(ans_rows)
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)

    # plot: captured d-energy vs k, markers colored by removed, per object
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"dog": "tab:blue", "cat": "tab:orange", "car": "tab:green"}
    from collections import defaultdict
    by = defaultdict(list)
    for r in summ[1:]:
        by[r[0]].append((int(r[1]), float(r[2]), bool(r[3])))
    for obj, rows in by.items():
        rows.sort()
        ks = [r[0] for r in rows]; en = [r[1] for r in rows]
        ax.plot(ks, en, "-", color=colors.get(obj, "gray"), alpha=0.6, label=f"{obj}: captured d-energy")
        for k, e, rem in rows:
            ax.scatter([k], [e], c=("tab:green" if rem else "tab:red"), s=60, zorder=3,
                       edgecolors="k", linewidths=0.5)
        fr = next((k for k, e, rem in rows if rem and k > 0), None)
        if fr is not None:
            ax.axvline(fr, ls="--", color=colors.get(obj, "gray"), lw=1)
    ax.set_xlabel("rank k of removed subspace"); ax.set_ylabel("fraction of d-energy removed (cumulative EV)")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax.set_title("Detection-relevant rank: green=object removed, red=still detected\n"
                 "dashed = smallest k that removes; where it sits on the energy curve = top-k vs tail")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.outdir}/rank_vs_removal.png", dpi=130); plt.close(fig)

    print(f"\nsaved {args.outdir}/{{answers,summary}}.csv, rank_vs_removal.png")
    print("READ: smallest removing k = the rank a robust removal loss must zero per token. If it sits LOW on the "
          "energy curve (small k, modest captured %), detection lives in the top few directions -> a cheap rank-k "
          "subspace loss suffices. If removal needs captured~1.0 (k~full), the signal is in the tail and subspace "
          "removal is no cheaper than full substitution.")


if __name__ == "__main__":
    main()
