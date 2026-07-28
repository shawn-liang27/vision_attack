"""Stage 30: proof-motivated attack rebuild. Every design choice follows from a proof, not
a hypothesis:

  * REGION = 1.5x dilated object mask. The oracle (oracle_substitution.py) proved perfect
    token substitution conceals dog at 1.5x (130/576 tokens) but NOT at tight 1.0x (87) or
    1.25x (109) -- the halo band (silhouette + attention-neighbors) must also be removed.
  * L_fix preserves ONLY tokens OUTSIDE the 1.5x band. In the inpaint the object's neighbors
    differ from clean, so pinning the halo clean fought the target -- the halo is free to move.
  * LOSS = minimize DISTANCE to the inpaint tokens (not cosine). check_amplification showed the
    cosine attack reached proj~1 (object component intact/amplified) because cosine is scale-
    and direction-invariant. Two arms: L2 (per-patch squared distance, normalized by the object-
    direction scale) and PROJ (the object-direction coefficient itself, = the check's metric).

The oracle is the distance-0 limit of this attack; the question is whether a BOUNDED perturbation
(L_inf {8,16,32}/255) can drive the read-layer region tokens close enough to concede. Attack at
hidden_states[-2] (the layer LLaVA reads); whole-image delta; surrogate CLIP (float32) for the
attack, LLaVA for eval. Corrected scoring (yes/no probes PARSED, presup dropped). dog, 5 seeds.

Diagnostics per run: achieved mean proj and relative-L2 over the region (oracle target = 0) --
does the bounded attack actually reach the target the oracle needed, and does removal follow?

    uv run python dilated_attack.py --dataset dataset.jsonl --objects dog \
        --arms l2,proj --eps 8,16,32 --dilate 1.5 --seeds 5
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
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--arms", default="l2,proj")
    ap.add_argument("--eps", default="8,16,32", help="L_inf budgets /255")
    ap.add_argument("--dilate", type=float, default=1.5, help="steer region = this multiple of the tight object-token count")
    ap.add_argument("--w-fix", type=float, default=0.25, help="weight on preserve-clean OUTSIDE the dilated band")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--outdir", default="results/dilated_attack")
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate).to(DEVICE).eval()  # float32 surrogate for the attack
    clip.requires_grad_(False)
    cproc = CLIPProcessor.from_pretrained(args.surrogate)
    ipc = cproc.image_processor
    MEAN = torch.tensor(ipc.image_mean, device=DEVICE).view(1, 3, 1, 1)
    STD = torch.tensor(ipc.image_std, device=DEVICE).view(1, 3, 1, 1)
    PATCH = clip.config.vision_config.patch_size
    GRID = RES // PATCH
    NL = clip.config.vision_config.num_hidden_layers
    READ = NL - 1   # hidden_states index for LLaVA's -2 (== len(hs)-2), i.e. the layer LLaVA reads

    victim = AutoModelForImageTextToText.from_pretrained(args.victim, torch_dtype=DTYPE, device_map=DEVICE).eval()
    victim.requires_grad_(False)
    vproc = AutoProcessor.from_pretrained(args.victim)
    print(f"surrogate={args.surrogate}(f32, attack) grid={GRID}; victim={args.victim}(eval)")
    print(f"attack layer hidden_states[-2] (index {READ}); region={args.dilate}x dilated; L_fix only OUTSIDE it; "
          f"arms={args.arms}; eps(Linf)/255={args.eps}")

    def vis(x01):
        return clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                 output_hidden_states=True).hidden_states[-2][:, 1:].squeeze(0)  # (P,D) raw read-layer

    def load01(p):
        im = Image.open(p).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    def gen(x01, prompt):
        img = Image.fromarray((x01.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = vproc.apply_chat_template(messages, add_generation_prompt=True)
        inp = vproc(images=img, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            g = victim.generate(**inp, max_new_tokens=48, do_sample=False)
        return vproc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def probes(obj):
        return [("direct1", f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("direct2", f"Do you see a {obj} in this image? Answer with only 'yes' or 'no'.", True),
                ("describe", "Describe this image.", False),
                ("list", "List all objects you see in this image.", False),
                ("detail", "Describe this image in detail.", False),
                ("count", f"How many {obj}s are in this image? If none, answer 0.", False)]

    arms = args.arms.split(",")
    epslist = [float(e) for e in args.eps.split(",")]
    with open(args.dataset) as f:
        samples = [s for s in (json.loads(l) for l in f if l.strip()) if s["object"] in args.objects.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "adv"); os.makedirs(imgdir, exist_ok=True)
    ans_rows = [["object", "arm", "eps", "seed", "prompt", "answer_text", "present", "linf"]]
    diag_rows = [["object", "arm", "eps", "seed", "proj_region", "relL2_region", "strict_removed", "linf"]]

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not (os.path.exists(s["mask"]) and s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] missing mask/inpaint; skip"); continue
        x0, xR = load01(s["image"]), load01(s["target"])
        pr = probes(obj)
        with torch.no_grad():
            h_clean = vis(x0).detach()          # (P,D) raw read-layer clean
            h_inp = vis(xR).detach()            # raw read-layer inpaint (the target)
        d = (h_clean - h_inp).detach()          # object direction per patch
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        base = mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3
        n0 = int(base.sum()); order = ring_order(base)
        T = min(GRID * GRID, max(n0, round(args.dilate * n0)))
        sel = np.zeros(GRID * GRID, bool); sel[order[:T]] = True
        si = torch.from_numpy(sel).to(DEVICE)          # steer region (dilated)
        oo = torch.from_numpy(~sel).to(DEVICE)         # outside -> preserve
        dn = (d[si] ** 2).sum(-1).clamp_min(1e-8)      # ||d_i||^2 over steer region
        print(f"\n=== {sid} '{obj}' n0={n0} tight -> {T} tokens ({T/n0:.2f}x) steer; {int(oo.sum())} preserved ===")

        for arm in arms:
            for eps in epslist:
                e01 = eps / 255.0
                for seed in range(args.seeds):
                    g = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
                    delta = ((torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * e01).detach().requires_grad_(True)
                    opt = torch.optim.Adam([delta], lr=args.lr)
                    for _ in range(args.iters):
                        h = vis(torch.clamp(x0 + delta, 0, 1))
                        resid = h - h_inp
                        if arm == "l2":
                            steer = ((resid[si] ** 2).sum(-1) / dn).mean()             # per-patch relative squared distance
                        else:
                            steer = ((resid[si] * d[si]).sum(-1) / dn).mean()          # object-direction coefficient (proj)
                        fix = (1 - F.cosine_similarity(h[oo], h_clean[oo], dim=-1)).mean()
                        loss = steer + args.w_fix * fix
                        opt.zero_grad(); loss.backward(); opt.step()
                        with torch.no_grad():
                            delta.clamp_(-e01, e01)
                            delta.data = torch.clamp(x0 + delta, 0, 1) - x0

                    adv = torch.clamp(x0 + delta.detach(), 0, 1)
                    linf = round(delta.detach().abs().max().item() * 255, 1)
                    with torch.no_grad():
                        resid = vis(adv) - h_inp
                        proj = ((resid[si] * d[si]).sum(-1) / dn).mean().item()        # achieved proj (oracle target 0)
                        relL2 = ((resid[si] ** 2).sum(-1) / dn).mean().item()          # achieved relative L2 (target 0)
                    if seed == 0:
                        Image.fromarray((adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                            f"{imgdir}/{obj}_{arm}_eps{eps:g}_s0.png")
                    strict = False
                    for pk, pt, yn in pr:
                        a = gen(adv, pt); p = detect(a, obj, yn); strict = strict or p
                        ans_rows.append([obj, arm, eps, seed, pk, a.replace("\n", " ")[:200], p, linf])
                    diag_rows.append([obj, arm, eps, seed, round(proj, 4), round(relL2, 4), not strict, linf])
                    print(f"  [{arm:<4}] eps{eps:g} s{seed} Linf={linf:>4} | proj={proj:+.3f} relL2={relL2:.3f} "
                          f"(oracle target 0) | removed={not strict}")

    for name, rows in [("answers", ans_rows), ("diagnostics", diag_rows)]:
        with open(f"{args.outdir}/{name}.csv", "w", newline="") as fp:
            csv.writer(fp).writerows(rows)

    # summary: strict removal % + achieved proj/relL2 per (object, arm, eps)
    from collections import defaultdict
    grp = defaultdict(list)
    for r in diag_rows[1:]:
        obj, arm, eps, seed, proj, relL2, removed, linf = r
        grp[(obj, arm, float(eps))].append((removed, proj, relL2))
    summ = [["object", "arm", "eps", "strict_removal_rate", "mean_proj", "mean_relL2", "n"]]
    print("\n=== REBUILT ATTACK: bounded L2/proj on the 1.5x region -- does removal follow the oracle? ===")
    print(f"{'object':<5}{'arm':<5}{'eps':>4}{'removal%':>10}{'mean_proj':>11}{'mean_relL2':>12}")
    for key in sorted(grp.keys()):
        obj, arm, eps = key
        rows = grp[key]
        rem = 100 * np.mean([r[0] for r in rows]); mp = float(np.mean([r[1] for r in rows]))
        ml = float(np.mean([r[2] for r in rows])); n = len(rows)
        summ.append([obj, arm, eps, round(rem, 1), round(mp, 3), round(ml, 3), n])
        print(f"{obj:<5}{arm:<5}{eps:>4.0f}{rem:>9.0f}%{mp:>11.3f}{ml:>12.3f}")
    with open(f"{args.outdir}/summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)

    # plot: removal% and achieved proj vs eps, per arm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for arm in arms:
        for obj in args.objects.split(","):
            pts = sorted((eps, grp[(obj, arm, eps)]) for (o, ar, eps) in grp if o == obj and ar == arm)
            if not pts:
                continue
            xs = [e for e, _ in pts]
            a1.plot(xs, [100 * np.mean([r[0] for r in v]) for _, v in pts], "-o", label=f"{obj}:{arm}")
            a2.plot(xs, [np.mean([r[1] for r in v]) for _, v in pts], "-o", label=f"{obj}:{arm}")
    a1.set_xlabel("L_inf budget /255"); a1.set_ylabel("STRICT removal %"); a1.set_ylim(-5, 105); a1.grid(alpha=0.3); a1.legend()
    a2.axhline(0, ls="--", color="gray", label="oracle target (proj=0)")
    a2.set_xlabel("L_inf budget /255"); a2.set_ylabel("achieved mean proj over region (1=object intact)")
    a2.grid(alpha=0.3); a2.legend()
    fig.suptitle(f"Rebuilt attack @ {args.dilate}x region, read layer: does bounded L2/proj reach the oracle target?")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/removal_and_proj.png", dpi=130); plt.close(fig)

    print(f"\nsaved {args.outdir}/{{answers,diagnostics,summary}}.csv, removal_and_proj.png, adv/*.png")
    print("READ: removal% high AND mean_proj -> 0 => bounded attack reached the oracle target; concealment achieved. "
          "proj stuck near 1 (esp. l2 vs proj differ) => bounded budget can't drive the read-layer tokens to inpaint "
          "even with the right loss+region -> the ceiling is fidelity/budget, and the oracle target is unreachable.")


if __name__ == "__main__":
    main()
