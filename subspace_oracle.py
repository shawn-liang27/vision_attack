"""Stage 34: which low-rank subspace controls recognition? Four constructions, ONE oracle
validator, ranked by minimal removing rank (fewer constraints wins -- the reachability argument).

For each construction we build an ordered set of directions W (in the read-layer 1024-d space),
estimated with SINK TOKENS EXCLUDED (||h_clean||>sink_mult*median). Then the shared validator
substitutes, at the read layer, over ALL patch tokens (sinks kept in the substitution set):
    h_mod[i] = h_clean[i] - Q_k Q_k^T d_i          d_i = h_clean[i]-h_inpaint[i], Q_k = orthonormal top-k of W
k=0 -> h_clean (no change); if W spans d over the region, large k -> h_inpaint. Query the victim
(sensitive DIRECT yes/no probes + open probes) and find the minimal k that removes the object.
No optimization, no decoder gradients -- forward passes only.

Constructions:
  fisher  : rank-1 DISCRIMINATIVE. w = Sw^-1 (mu_obj - mu_bg), pooled within-class scatter Sw (reg).
            Chosen for SEPARATION, not variance -- the key contrast with PCA-1 (which failed).
  pls     : multi-direction discriminative. PLS1(patch features -> obj/bg label), top components.
  svd     : sink-free SVD of {d_i} over the region (variance-based; the existing construction, cleaned).
  text    : w = proj_to_patch( phi_T(obj) - phi_T(context) ), rank-1 (+ a few contrasts). Free: needs
            neither inpaint nor labels. Weakest (text-aligned subspace patch tokens only partly occupy).

Ranked by minimal removing rank on the DOG reference (the one that concealed under the direct probe):
a construction that fails the oracle on dog is dead regardless of elsewhere. Lowest removing rank wins.

    uv run python subspace_oracle.py --dataset dataset.jsonl --objects dog \
        --ks 0,1,2,3,5,8,10,15,22
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

from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYN = {"dog": ["dog", "puppy", "canine"], "cat": ["cat", "kitten", "feline"],
       "car": ["car", "sedan", "automobile"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "sign": ["sign", "signage", "placard"]}
CTX = {"dog": ["a couch", "a wall", "the floor", "a room"], "cat": ["a couch", "a window", "a wall", "a room"],
       "car": ["a street", "trees", "the sky", "a road"]}
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


def orthonormal(dirs):
    """dirs: (r,1024) -> orthonormal-column basis (1024, r) via QR."""
    Q, _ = torch.linalg.qr(dirs.T.float())
    return Q                                        # (1024, r)


def pls1(X, y, ncomp):
    """NIPALS PLS1: X (n,d), y (n,) -> (ncomp, d) x-weight directions (discriminative subspace)."""
    X = (X - X.mean(0)).clone(); y = (y - y.mean()).clone(); Ws = []
    for _ in range(ncomp):
        w = X.t() @ y
        nw = w.norm()
        if nw < 1e-9:
            break
        w = w / nw
        t = X @ w
        tt = (t @ t).clamp_min(1e-9)
        p = (X.t() @ t) / tt
        X = X - torch.outer(t, p)
        y = y - ((t @ y) / tt) * t
        Ws.append(w)
    return torch.stack(Ws)                          # (<=ncomp, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")   # text encoder + visual_projection
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--sink-mult", type=float, default=4.0)
    ap.add_argument("--pls-ncomp", type=int, default=15)
    ap.add_argument("--reg", type=float, default=1e-2, help="Fisher scatter ridge (fraction of mean diagonal)")
    ap.add_argument("--ks", default="0,1,2,3,5,8,10,15,22")
    ap.add_argument("--outdir", default="results/subspace_oracle")
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

    clip_s = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()          # for text + visual_projection
    clip_s.requires_grad_(False)
    cproc = CLIPProcessor.from_pretrained(args.surrogate)
    vproj_W = clip_s.visual_projection.weight.detach().float()                    # (768, 1024)
    print(f"victim={args.victim}; substitute ALL patch tokens at hidden_states[{layer}]; "
          f"estimate sink-free (||h_clean||>{args.sink_mult}x median), substitute keeps sinks; ref=dog")

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
            return vt(pv, output_hidden_states=True).hidden_states[layer]          # (1,1+P,D)

    def txt_emb(words):
        tk = cproc(text=words, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(clip_s.text_projection(clip_s.text_model(**tk).pooler_output), dim=-1).float()  # (n,768)

    def make_hook(h_mod):                          # h_mod: (P, D) -> replace ALL patch tokens
        def hook(mod, inp, out):
            hs = list(out.hidden_states); patched = hs[layer].clone()
            patched[:, 1:, :] = h_mod.unsqueeze(0).to(patched.dtype)
            hs[layer] = patched; out.hidden_states = tuple(hs); return out
        return hook

    def probes(obj):
        return [("direct1", f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("direct2", f"Do you see a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("direct3", f"Is a {obj} present in this image? Answer with only 'yes' or 'no'.", True),
                ("describe", "Describe this image.", False),
                ("list", "List all objects you see in this image.", False),
                ("count", f"How many {obj}s are in this image? If none, answer 0.", False)]

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    rows = [["object", "construction", "n_dirs", "k", "captured_region", "removed"]]
    summary = [["object", "construction", "min_removing_rank", "n_dirs"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/inpaint; skip"); continue
        clean_pil, inpaint_pil = load_pil(s["image"]), load_pil(s["target"])
        pr = probes(obj)
        ref = gen(inpaint_pil, pr[0][1])
        if detect(ref, obj, True):
            print(f"[{sid}] inpaint still shows '{obj}' -> invalid target; skip"); continue

        hcl = tokens(clean_pil)[0, 1:].float()            # (P, D) clean read-layer
        hip = tokens(inpaint_pil)[0, 1:].float()          # (P, D) inpaint
        d = hcl - hip                                     # (P, D) object direction per token
        cn = hcl.norm(dim=-1)
        sink = cn > args.sink_mult * cn.median()
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        obj_idx = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > args.thresh).reshape(-1)).to(DEVICE)
        keep = ~sink                                      # exclude sinks from ESTIMATION
        oe = obj_idx & keep                               # object, sink-free
        be = (~obj_idx) & keep                            # background, sink-free
        d_region = d[obj_idx]                             # for captured-energy denominator (region incl any sinks)
        print(f"\n=== {sid} '{obj}': {int(obj_idx.sum())} obj / {int(be.sum())} bg tokens (sink-free est), {int(sink.sum())} sinks ===")

        # ---- build the four constructions (ordered directions, float32, (r,1024)) ----
        constructions = {}
        # fisher (rank-1 discriminative)
        Xo, Xb = hcl[oe], hcl[be]
        mu = Xo.mean(0) - Xb.mean(0)
        Xoc, Xbc = Xo - Xo.mean(0), Xb - Xb.mean(0)
        Sw = (Xoc.t() @ Xoc + Xbc.t() @ Xbc) / (Xo.shape[0] + Xb.shape[0])
        Sw = Sw + args.reg * Sw.diagonal().mean() * torch.eye(Sw.shape[0], device=DEVICE)
        wf = torch.linalg.solve(Sw, mu); wf = wf / wf.norm()
        constructions["fisher"] = wf.unsqueeze(0)
        # pls (multi discriminative)
        Xp = hcl[keep]; yp = obj_idx[keep].float()
        constructions["pls"] = pls1(Xp, yp, args.pls_ncomp)
        # svd (sink-free, variance-based) of region d_i
        Dsvd = d[oe]
        _, _, Vh = torch.linalg.svd(Dsvd, full_matrices=False)
        constructions["svd"] = Vh
        # text (rank-1 + a few contrasts)
        te = txt_emb([f"a photo of a {obj}"] + [f"a photo of {c}" for c in CTX.get(obj, ["a background"])])
        tdirs = F.normalize((te[0:1] - te[1:]) @ vproj_W, dim=-1)   # (n_ctx,1024): text contrast -> patch space
        constructions["text"] = tdirs

        # ---- validate each construction through the shared oracle ----
        for name, dirs in constructions.items():
            r = dirs.shape[0]
            first = None
            for k in [kk for kk in ks if kk <= r]:
                if k == 0:
                    h_mod = hcl
                    captured = 0.0
                else:
                    Q = orthonormal(dirs[:k])                        # (1024,k)
                    proj = (d @ Q) @ Q.t()                           # (P,1024): project each d onto span
                    h_mod = hcl - proj
                    captured = ((proj[obj_idx] ** 2).sum() / (d_region ** 2).sum().clamp_min(1e-9)).item()
                h = vt.register_forward_hook(make_hook(h_mod))
                try:
                    strict = any(detect(gen(clean_pil, pt), obj, yn) for _, pt, yn in pr)
                finally:
                    h.remove()
                removed = not strict
                rows.append([obj, name, r, k, round(captured, 4), removed])
                if removed and k > 0 and first is None:
                    first = k
                print(f"  {name:<7} k={k:>2} (of {r})  captured={captured:5.2f}  removed={removed}")
            summary.append([obj, name, first if first is not None else -1, r])
            print(f"  -> {name}: min removing rank = {first if first is not None else 'none in sweep'}")

    with open(f"{args.outdir}/sweep.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(rows)
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summary)

    print("\n=== RANKING by minimal removing rank (lower = fewer constraints = wins) ===")
    for obj in dict.fromkeys(r[0] for r in summary[1:]):
        print(f"  {obj}:")
        ranked = sorted([r for r in summary[1:] if r[0] == obj], key=lambda r: (r[2] if r[2] > 0 else 1e9))
        for _, name, mr, r in ranked:
            print(f"    {name:<7} min_removing_rank={mr if mr > 0 else 'none':<6} (n_dirs={r})")
    print("\nsaved sweep.csv + summary.csv")
    print("READ: on DOG (reference), lowest min_removing_rank wins. If FISHER clears at 1-2 while SVD (PCA) needs "
          "many, discrimination beats variance -> attack with the Fisher direction (hinge + orthogonal-anchor loss). "
          "If fisher='none' but svd removes, the recognition signal isn't in the discriminative direction after all.")


if __name__ == "__main__":
    main()
