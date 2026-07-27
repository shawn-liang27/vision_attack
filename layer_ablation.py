"""Stage 27 (v2): layer ablation + per-block propagation profile for the early-layer
concealment attack. Rewritten after code review -- fixes evaluation, propagation
metric, and design confounds.

THIS IS NOT A FAITHFUL AdvEDM-R REPRODUCTION (deviations, flagged):
  * D2: L_p steers toward the INPAINT image (--target inpaint) or the zeroed black
    box (--target zeroed), NOT solely the paper's zeroed image. inpaint-vs-zeroed is
    the variable most likely to explain "dog 100% here vs 0% in the AdvEDM run", so
    it is now an explicit ablation axis.
  * D1: L_fix uses the CLEAN attention as a static per-patch weight; the paper's Eq.7
    constrains the ADVERSARIAL attention A_adv (want_attn during optimization). We do
    NOT compute A_adv (too expensive per-iter); this is a documented deviation.

DESIGN (fixes C1/C2):
  Backbone (--backbone on, default) = L_cls + L_p@block-24 + L_fix, identical to the
  working run. Arm name = ADDITIONAL shallow layers where L_p is also applied. Since
  the block-24 backbone L_p is already a full attack, backbone-on CANNOT isolate the
  shallow layers -> use --backbone off for shallow-ONLY arms (loss = L_p at the listed
  layers only). 'base' = backbone only (control + eval-validity check); dropped when
  backbone off (nothing to optimize).

EVAL (fixes B1/B2/B3): presence is clause-level negation-aware; generic terms
(animal/pet) removed from SYN so "no animal" no longer scores as a detection; STRICT =
object present in ANY of {describe, direct, list, presup, detail} (weak prompt now
included). Reality-stripping ("dog statue/stuffed/painting-of") and image-destruction
("blur/noise/pattern") are flagged separately. Full captions logged; the inpaint
image's own captions are logged and the object is SKIPPED if the inpaint still shows
it (E4 -- unachievable removal target).

PROFILE (fixes A1/A2/A4): roiP_L = softmax-P(obj vs background set) from
visual_projection(post_layernorm(patch_L)), L=1..24 -- post-LN makes the projection
in-distribution; contrastive P is sign-meaningful. Verdict compares adv to CLEAN
(clean-adv drop), not adv to its own min. visual_projection on shallow layers is still
heuristic -> quantitative claims restricted to block 23 (LLaVA's read layer).

Seeds vary the delta init only (greedy decoding, fixed target/mask/prompts): error
bars are initialization variance, not evaluation variance (C4).

    # C2 -- the key test: does the inpaint target (not the shallow layers) explain it?
    uv run python layer_ablation.py --arms base --target inpaint,zeroed --eps 16 --seeds 5
    # C1+C3 -- shallow-only ablation, de-saturated eps, more seeds:
    uv run python layer_ablation.py --backbone off --arms all --target inpaint --eps 4,8,12 --seeds 15
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

# object-specific synonyms only -- generic terms (animal/pet) removed so a *negated*
# category answer to the presup prompt no longer scores as a detection (B1)
SYN = {"dog": ["dog", "puppy", "canine"], "cat": ["cat", "kitten", "feline"]}
CAT = {"dog": "animal", "cat": "animal"}
NEGS = ["background", "a wall", "a floor", "furniture", "the sky", "an empty room"]
NEG_CUES = ["no ", "not ", "n't", "without", "cannot", "can not", "none", "absent",
            "missing", "lack", "empty of", "free of", "devoid", "there is no",
            "there are no", "do not see", "don't see", "no sign"]
STRIP = ["statue", "stuffed", "figurine", "sculpture", " toy", "plush", "cartoon",
         "drawing", "painting", "poster", "mural", "cardboard", "plastic"]
DEGEN = ["blur", "blurr", "noise", "noisy", "static", "pixelat", "distort", "glitch",
         "abstract", "colorful pattern", "pattern of color", "random", "corrupt",
         "unclear", "difficult to", "hard to tell", "can't tell", "not clear"]
ARMS = {"base": [], "L2_only": [2], "L4_only": [4], "L6_only": [6],
        "L2_L4": [2, 4], "L2_L6": [2, 6], "L4_L6": [4, 6], "L2_L4_L6": [2, 4, 6]}


def _clauses(text):
    return re.split(r"[.,;:!?]| but | and | although | though | however ", text.lower())


def present(text, obj):
    """object asserted present -- clause-level, negation-aware (B1)."""
    syns = SYN.get(obj, [obj])
    for cl in _clauses(text):
        if any(sy in cl for sy in syns) and not any(neg in cl for neg in NEG_CUES):
            return True
    return False


def reality_stripped(text, obj):
    """object mentioned but as a non-real depiction ('dog statue', 'painting of a dog') (B3)."""
    syns = SYN.get(obj, [obj])
    return any(any(sy in cl for sy in syns) and any(st in cl for st in STRIP) for cl in _clauses(text))


def degenerate(text):
    """caption describes corrupted pixels rather than a coherent scene."""
    return any(d in text.lower() for d in DEGEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--eps", default="16", help="L2 epsilon(s), comma list (C3 de-saturation sweep)")
    ap.add_argument("--target", default="inpaint", help="removal target(s): comma list of {inpaint,zeroed} (C2)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--w2", type=float, default=2.0)
    ap.add_argument("--w3", type=float, default=0.2)
    ap.add_argument("--w-inter", type=float, default=1.0)
    ap.add_argument("--backbone", default="on", choices=["on", "off"],
                    help="on: every arm keeps L_cls+L_p@block24+L_fix; off: shallow-only (isolates the layers, C1)")
    ap.add_argument("--arms", default="all", help="comma list of arm names, or 'all'")
    ap.add_argument("--profile-arms", default="base,L2_L4_L6", help="arms to build the propagation profile for (A5)")
    ap.add_argument("--outdir", default="results/layer_ablation")
    args = ap.parse_args()
    RES = args.res

    clip = CLIPModel.from_pretrained(args.surrogate, attn_implementation="eager").to(DEVICE).eval()
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

    epslist = [float(e) for e in args.eps.split(",") if e.strip()]
    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    assert all(t in ("inpaint", "zeroed") for t in targets), targets
    arms = list(ARMS.keys()) if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.backbone == "off":
        arms = [a for a in arms if ARMS[a]]  # base (no layers) has nothing to optimize
    profile_arms = [a.strip() for a in args.profile_arms.split(",") if a.strip() and a.strip() in arms]
    inter_pool = sorted({l for a in arms for l in ARMS[a]})

    print(f"surrogate={args.surrogate} grid={GRID} blocks={NL}; victim={args.victim}")
    print(f"CONVENTION: 'layer L' = output of L-th transformer block (post attn+MLP+residual) = hidden_states[L], "
          f"1-indexed (hidden_states[0]=embeddings). shallow attacked = hidden_states[2],[4],[6]; "
          f"LLaVA reads hidden_states[-2]=hidden_states[{NL - 1}]; profile L=1..{NL}. post-block RESIDUAL (not V).")
    print(f"eps(L2)={epslist} targets={targets} backbone={args.backbone} w_inter={args.w_inter} "
          f"w1/w2/w3={args.w1}/{args.w2}/{args.w3} seeds={args.seeds} (init-variance only)")
    print(f"arms={arms}; shallow pool={inter_pool}; profile_arms={profile_arms}")
    init_l2 = (epslist[0] / (RES * 1.7)) / (3 ** 0.5) * (3 * RES * RES) ** 0.5
    print(f"delta init ~ U(-eps/(RES*1.7), +): init L2 ~= {init_l2:.1f} at eps={epslist[0]:.0f} "
          f"({100 * init_l2 / epslist[0]:.0f}% of budget) (E1)")

    def txt_emb(words):
        tk = cproc(text=words, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(clip.text_projection(clip.text_model(**tk).pooler_output), dim=-1)

    def vis(x01, want_attn=False):
        out = clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                output_hidden_states=True, output_attentions=want_attn)
        cls = F.normalize(clip.visual_projection(out.pooler_output), dim=-1)
        A = None
        if want_attn:
            A = torch.stack(out.attentions).mean(0).mean(1)[:, 0, 1:]
        return dict(cls=cls, last=out.last_hidden_state[:, 1:], attn=A, hs=out.hidden_states)

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

    def roiP_profile(x01, obj_idx, labels):
        """per-block calibrated P(obj vs bg) from visual_projection(post_layernorm(patch_L)) (A1/A4)."""
        with torch.no_grad():
            out = vis(x01)
            prof = []
            for L in range(1, NL + 1):
                tok = clip.vision_model.post_layernorm(out["hs"][L][:, 1:].squeeze(0))   # A1: post-LN first
                proj = F.normalize(clip.visual_projection(tok), dim=-1)
                prof.append(((proj[obj_idx] @ labels.T) * 100).softmax(-1)[:, 0].mean().item())
            return prof

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    objects = [o.strip() for o in args.objects.split(",")]
    samples = [s for s in samples if s["object"] in objects]
    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "adv"); os.makedirs(imgdir, exist_ok=True)

    ans_rows = [["arm", "object", "target", "eps", "seed", "criterion", "prompt", "answer_text",
                 "present", "reality_stripped", "degenerate", "linf"]]
    curve_rows = [["arm", "object", "target", "eps", "seed", "iter", "L_cls", "L_p", "L_fix", "L_p_inter"]]
    roiP_rows = [["object", "target", "eps", "arm", "seed", "layer", "kind", "roiP"]]
    adv_store = {}    # (target,eps,arm,object) -> list of adv tensors per seed
    clean_prof = {}   # object -> clean roiP list
    seen_obj = set()

    for s in samples:
        obj, sid = s["object"], s["id"]
        if obj in seen_obj:
            print(f"[{sid}] duplicate object '{obj}' -- keys collide; skipping (E3)"); continue
        seen_obj.add(obj)
        if not os.path.exists(s["mask"]):
            print(f"[{sid}] no mask; skip"); continue
        x0 = load01(s["image"])
        E_t = txt_emb([f"a photo of a {obj}"])[0]
        labels = txt_emb([f"a photo of a {obj}"] + [f"a photo of {d}" for d in NEGS])  # consistent template (D3)

        with torch.no_grad():
            clean = vis(x0, want_attn=True)
        clean_last_n = F.normalize(clean["last"].squeeze(0), dim=-1)
        A_clean = clean["attn"][0]

        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        obj_idx = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)
        bg_idx = ~obj_idx
        print(f"\n=== {sid} '{obj}' ({int(obj_idx.sum())}/{GRID * GRID} obj tokens) ===")

        prompts = {"describe": "Describe this image.",
                   "direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
                   "list": "List all objects you see in this image.",
                   "presup": f"What {CAT.get(obj, 'object')} is in this image?",
                   "detail": "Describe this image in detail."}
        clean_prof[obj] = roiP_profile(x0, obj_idx, labels)
        for L in range(1, NL + 1):
            roiP_rows.append([obj, "clean", 0, "clean", -1, L, "clean", round(clean_prof[obj][L - 1], 5)])

        # precompute removal-target references per requested target
        target_refs = {}
        mpix = torch.from_numpy((np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.NEAREST)) > 127)
                                .astype(np.float32)).view(1, 1, RES, RES).to(DEVICE)
        for tgt in targets:
            if tgt == "inpaint":
                if not (s.get("target") and os.path.exists(s["target"])):
                    print(f"  [{obj}] target=inpaint but no inpaint file; skipping this target"); continue
                xR = load01(s["target"])
                # E4 validity gate: does the inpaint image itself still show the object?
                ref_dir = gen(xR, prompts["direct"]); ref_desc = gen(xR, "Describe this image.")
                ans_rows.append(["_inpaint_ref", obj, "inpaint", 0, 0, "ref", "direct", ref_dir.replace("\n", " ")[:200],
                                 present(ref_dir, obj), reality_stripped(ref_dir, obj), degenerate(ref_dir), 0])
                ans_rows.append(["_inpaint_ref", obj, "inpaint", 0, 0, "ref", "describe", ref_desc.replace("\n", " ")[:200],
                                 present(ref_desc, obj), reality_stripped(ref_desc, obj), degenerate(ref_desc), 0])
                if present(ref_dir, obj) or present(ref_desc, obj):
                    print(f"  [{obj}] INPAINT STILL SHOWS OBJECT (dir={ref_dir[:40]!r}) -> unachievable target; SKIP inpaint (E4)")
                    continue
                print(f"  [{obj}] inpaint-ref ok: {ref_desc[:70]!r}")
            else:
                xR = x0 * (1 - mpix)
            with torch.no_grad():
                fR = vis(xR)
            target_refs[tgt] = (F.normalize(fR["last"].squeeze(0), dim=-1),
                                {l: F.normalize(fR["hs"][l][:, 1:].squeeze(0), dim=-1) for l in inter_pool})

        for tgt in [t for t in targets if t in target_refs]:
            featR_last_n, featR_layer = target_refs[tgt]
            for eps in epslist:
                for arm in arms:
                    arm_layers = ARMS[arm]
                    for seed in range(args.seeds):
                        g = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
                        delta = ((torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * (eps / (RES * 1.7)))
                        delta = delta.detach().requires_grad_(True)
                        opt = torch.optim.Adam([delta], lr=args.lr)
                        for it in range(args.iters):
                            x = torch.clamp(x0 + delta, 0, 1)
                            f = vis(x, want_attn=False)
                            Lp_inter = x0.new_zeros(())
                            if arm_layers:
                                for l in arm_layers:
                                    pil = F.normalize(f["hs"][l][:, 1:].squeeze(0), dim=-1)
                                    Lp_inter = Lp_inter - (pil[obj_idx] * featR_layer[l][obj_idx]).sum(-1).mean()
                                Lp_inter = Lp_inter / len(arm_layers)
                            if args.backbone == "on":
                                L_cls = (f["cls"].squeeze(0) @ E_t)
                                pl = F.normalize(f["last"].squeeze(0), dim=-1)
                                Lp = -(pl[obj_idx] * featR_last_n[obj_idx]).sum(-1).mean()
                                w = A_clean[bg_idx]; w = w / w.sum().clamp_min(1e-9)
                                Lfix = -(w * (pl[bg_idx] * clean_last_n[bg_idx]).sum(-1)).sum()
                                L = args.w1 * L_cls + args.w2 * Lp + args.w3 * Lfix + args.w_inter * Lp_inter
                                cl, lp, lf = round(L_cls.item(), 4), round(Lp.item(), 4), round(Lfix.item(), 4)
                            else:
                                L = args.w_inter * Lp_inter
                                cl = lp = lf = ""
                            opt.zero_grad(); L.backward(); opt.step()
                            with torch.no_grad():
                                n = delta.flatten().norm(2)
                                if n > eps:
                                    delta.mul_(eps / n)
                                delta.data = torch.clamp(x0 + delta, 0, 1) - x0
                            if it % 10 == 0 or it == args.iters - 1:
                                curve_rows.append([arm, obj, tgt, eps, seed, it, cl, lp, lf, round(float(Lp_inter), 4)])

                        adv = torch.clamp(x0 + delta.detach(), 0, 1)
                        linf = round(delta.detach().abs().max().item() * 255, 1)
                        Image.fromarray((adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                            f"{imgdir}/{tgt}_{arm}_{obj}_eps{eps:g}_s{seed}.png")
                        if arm in profile_arms:
                            adv_store.setdefault((tgt, eps, arm, obj), []).append(adv.detach())

                        strict_present = False
                        for pk, pt in prompts.items():
                            a = gen(adv, pt); p = present(a, obj); strict_present = strict_present or p
                            crit = "weak" if pk == "describe" else "strict"
                            ans_rows.append([arm, obj, tgt, eps, seed, crit, pk, a.replace("\n", " ")[:200],
                                             p, reality_stripped(a, obj), degenerate(a), linf])
                        print(f"  [{tgt[:3]}|{arm:<9}|eps{eps:g}] {obj} s{seed} Linf={linf:>5} | strict_absent={not strict_present}")

    for name, rows in [("answers", ans_rows), ("curves", curve_rows)]:
        with open(f"{args.outdir}/{name}.csv", "w", newline="") as fp:
            csv.writer(fp).writerows(rows)

    # ================= PART 1: ablation summary (fixes B1/B2/B3) =================
    from collections import defaultdict
    grp = defaultdict(lambda: defaultdict(lambda: {"weak": False, "strict": False, "strip": False,
                                                    "degen": False, "linf": 0.0}))
    for r in ans_rows[1:]:
        arm, obj, tgt, eps, seed, crit, prompt, txt, pres, strip, degen, linf = r
        if arm == "_inpaint_ref":
            continue
        pres, strip, degen = (pres is True), (strip is True), (degen is True)
        cell = grp[(arm, obj, tgt, eps)][seed]
        if crit == "weak":
            cell["weak"] = pres; cell["linf"] = float(linf)
        cell["strict"] = cell["strict"] or pres          # STRICT = present in ANY probe (incl. weak) -> B2
        cell["strip"] = cell["strip"] or strip
        cell["degen"] = cell["degen"] or degen

    print("\n=== PART 1: layer ablation ===")
    print(f"{'arm':<10}{'obj':<5}{'tgt':<8}{'eps':>4}{'weak_rem%':>10}{'strict_rem%':>12}"
          f"{'Linf':>7}{'strip%':>8}{'degen_of_rem%':>14}")
    summ = [["arm", "object", "target", "eps", "weak_removal_rate", "strict_removal_rate",
             "mean_linf", "reality_stripped_rate", "degen_of_removed_rate", "n"]]
    for key in sorted(grp.keys()):
        arm, obj, tgt, eps = key
        cells = grp[key].values()
        weak_rem = 100 * np.mean([not c["weak"] for c in cells])
        strict_rem = 100 * np.mean([not c["strict"] for c in cells])
        strip_rate = 100 * np.mean([c["strip"] for c in cells])
        removed = [c for c in cells if not c["strict"]]
        degen_rem = 100 * np.mean([c["degen"] for c in removed]) if removed else 0.0
        mean_linf = float(np.mean([c["linf"] for c in cells]))
        n = len(grp[key])
        summ.append([arm, obj, tgt, eps, round(weak_rem, 1), round(strict_rem, 1),
                     round(mean_linf, 1), round(strip_rate, 1), round(degen_rem, 1), n])
        print(f"{arm:<10}{obj:<5}{tgt:<8}{eps:>4.0f}{weak_rem:>9.0f}%{strict_rem:>11.0f}%"
              f"{mean_linf:>7.0f}{strip_rate:>7.0f}%{degen_rem:>13.0f}%")
    with open(f"{args.outdir}/ablation_summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)
    print("VALIDITY: base(inpaint) vs base(zeroed) isolates the TARGET (C2). If base(inpaint) already removes "
          "and base(zeroed) does not, the inpaint target -- not the shallow layers -- explains it. "
          "degen_of_rem%>0 => some 'removals' are image-destruction, not concealment (check adv/*.png + captions).")

    # ---- loss curves: one figure per (object,target,eps) with subplots per arm ----
    curve_keys = sorted({(r[1], r[2], r[3]) for r in curve_rows[1:]})
    for (obj, tgt, eps) in curve_keys:
        rows = [r for r in curve_rows[1:] if r[1] == obj and r[2] == tgt and r[3] == eps]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
        for ax, arm in zip(axes.ravel(), ARMS.keys()):
            ar = [r for r in rows if r[0] == arm]
            if not ar:
                ax.set_visible(False); continue
            its = sorted({r[5] for r in ar})
            for j, lab in [(6, "L_cls"), (7, "L_p"), (8, "L_fix"), (9, "L_p_inter")]:
                vals = [[r[j] for r in ar if r[5] == it and r[j] != ""] for it in its]
                if any(vals):
                    ax.plot(its, [np.mean(v) if v else np.nan for v in vals], label=lab)
            ax.set_title(arm); ax.set_xlabel("iter"); ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f"Loss curves ({obj}, target={tgt}, eps={eps:g}, mean over seeds)")
        fig.tight_layout(); fig.savefig(f"{args.outdir}/loss_{obj}_{tgt}_eps{eps:g}.png", dpi=100); plt.close(fig)

    # ================= PART 2: propagation profile (fixes A2 -- vs clean) =================
    layers = list(range(1, NL + 1))
    print("\n=== PART 2: propagation profile (roiP=P(obj), post-LN; verdict vs CLEAN) ===")
    labels_by = {obj: txt_emb([f"a photo of a {obj}"] + [f"a photo of {d}" for d in NEGS]) for obj in clean_prof}
    obj_idx_by = {}
    for s in samples:
        if s["object"] in clean_prof:
            mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
            obj_idx_by[s["object"]] = torch.from_numpy(
                (mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)

    for (tgt, eps, arm) in sorted({(k[0], k[1], k[2]) for k in adv_store}):
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = {"dog": "tab:blue", "cat": "tab:orange"}
        for obj in [o for o in objects if (tgt, eps, arm, o) in adv_store]:
            oi, lb = obj_idx_by[obj], labels_by[obj]
            adv_curves = [roiP_profile(adv, oi, lb) for adv in adv_store[(tgt, eps, arm, obj)]]
            for seed, cur in enumerate(adv_curves):
                for L in range(1, NL + 1):
                    roiP_rows.append([obj, tgt, eps, arm, seed, L, "adv", round(cur[L - 1], 5)])
            adv_arr, clean_arr = np.array(adv_curves), np.array(clean_prof[obj])
            m, sd = adv_arr.mean(0), adv_arr.std(0)
            c = colors.get(obj, "tab:green")
            ax.plot(layers, clean_arr, "-", color=c, label=f"{obj} clean")
            ax.plot(layers, m, "--", color=c, label=f"{obj} adv")
            ax.fill_between(layers, m - sd, m + sd, color=c, alpha=0.18)
            drop = clean_arr - m                      # A2: adv vs CLEAN
            d_att = float(np.mean([drop[k - 1] for k in [2, 4, 6]]))
            d_read = float(drop[NL - 2])              # block 23 = LLaVA's read layer
            verdict = ("PROPAGATES to block 23" if d_read >= 0.6 * max(d_att, 1e-6)
                       else "RESIDUAL RESTORES by block 23" if d_read < 0.3 * max(d_att, 1e-6) else "PARTIAL")
            print(f"  [{tgt}|{arm}|eps{eps:g}] {obj}: clean@23={clean_arr[NL-2]:.3f} adv@23={m[NL-2]:.3f} "
                  f"drop@[2,4,6]avg={d_att:+.3f} drop@23={d_read:+.3f} => {verdict}")
        for L in [2, 4, 6]:
            ax.axvline(L, ls=":", color="gray", lw=1)
        ax.axvline(NL - 1, ls="-.", color="red", lw=1.2, label=f"LLaVA reads (block {NL - 1})")
        ax.set_xlabel("transformer block L (post-block residual)")
        ax.set_ylabel("roiP = P(obj vs background), post-LN")
        ax.set_title(f"Propagation: clean vs adv ({tgt}, {arm}, eps={eps:g}, {args.seeds} seeds)\n"
                     "dotted=attacked layers; block<23 projection is heuristic (trust block 23)")
        ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.outdir}/propagation_{tgt}_{arm}_eps{eps:g}.png", dpi=130); plt.close(fig)

    with open(f"{args.outdir}/roiP.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(roiP_rows)
    print(f"\nsaved {args.outdir}/{{answers,curves,roiP,ablation_summary}}.csv, "
          f"propagation_<tgt>_<arm>_eps*.png, loss_*.png, adv/*.png")
    print("DEVIATIONS from AdvEDM-R: L_p target=inpaint/zeroed (not paper's zero-only, D2); "
          "L_fix uses CLEAN attention as static weights, A_adv NOT constrained (D1). Not a faithful reproduction.")


if __name__ == "__main__":
    main()
