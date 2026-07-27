"""Stage 27: layer ablation + per-layer propagation profile for the early-layer
concealment attack (AdvEDM-R with shallow-layer deep supervision toward inpaint).

Context: the {2,4,6}+inpaint run reportedly concealed dog at eps=16 L2. This
characterizes it: (1) which of {2,4,6} do the work, (2) does the early corruption
propagate to block 23 (LLaVA's input) or does the residual stream restore it.

CONVENTION (printed at startup, verifiable): 'layer L' = the OUTPUT of the L-th
transformer block (post attn+MLP+residual add) = hidden_states[L], 1-indexed
(hidden_states[0] = patch+pos embeddings). CLIP ViT-L/14-336 has 24 blocks. The
attacked shallow layers index hidden_states[2],[4],[6]. LLaVA reads
hidden_states[-2] = hidden_states[23] (block 23). Profile covers L=1..24. This is
the post-block RESIDUAL STREAM x (what determines Q,K downstream), NOT the value V.

PART 1 -- ablation (8 arms: base + 3 singles + 3 pairs + triple), dog & cat, eps=16,
5 seeds. All arms keep the AdvEDM-R backbone (L_cls + L_p@block-24 + L_fix); the arm
name lists the ADDITIONAL shallow layers where L_p is also applied toward the inpaint
target. 'base' = backbone only (control). --backbone off => pure (only the listed
shallow layers, no backbone). Graded WEAK (object omitted from open caption) and
STRICT (absent under ALL of direct/list/presup/detail). Full captions + a degeneracy
flag + the inpaint-image's own caption (ground-truth "removed" reference) are logged
so a "removal" that is really image-destruction is visible, not hidden.

PART 2 -- propagation profile on the L2_L4_L6 adv images (dog & cat, all seeds):
roiP_L = mean_{i in obj} cos( visual_projection(token_i^L), phi_T(obj) ) for L=1..24,
clean vs adv. Plot mean+/-std over seeds, dashed lines at attacked 2,4,6 and block 23.

    uv run python layer_ablation.py --dataset dataset.jsonl --objects dog,cat --eps 16 --seeds 5
"""

import argparse
import csv
import json
import os

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

SYN = {"dog": ["dog", "puppy", "canine", "pet", "animal"],
       "cat": ["cat", "kitten", "feline", "pet", "animal"]}
CAT = {"dog": "animal", "cat": "animal"}
# markers that a caption describes corrupted pixels rather than a coherent dog-free scene
DEGEN = ["blur", "blurr", "noise", "noisy", "static", "pixelat", "distort", "glitch",
         "abstract", "colorful pattern", "pattern of color", "random", "corrupt",
         "unclear", "difficult to", "hard to tell", "cannot", "can't tell", "not clear"]

ARMS = {"base": [], "L2_only": [2], "L4_only": [4], "L6_only": [6],
        "L2_L4": [2, 4], "L2_L6": [2, 6], "L4_L6": [4, 6], "L2_L4_L6": [2, 4, 6]}


def present(text, obj):
    return any(s in text.lower() for s in SYN.get(obj, [obj]))


def degenerate(text):
    return any(d in text.lower() for d in DEGEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--victim", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--objects", default="dog,cat")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--eps", type=float, default=16.0, help="L2 epsilon")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--w2", type=float, default=2.0)
    ap.add_argument("--w3", type=float, default=0.2)
    ap.add_argument("--w-inter", type=float, default=1.0)
    ap.add_argument("--backbone", default="on", choices=["on", "off"],
                    help="on: every arm keeps L_cls+L_p@block24+L_fix (working baseline); off: only the listed shallow layers")
    ap.add_argument("--arms", default="all", help="comma list of arm names, or 'all'")
    ap.add_argument("--profile-arm", default="L2_L4_L6")
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

    print(f"surrogate={args.surrogate} grid={GRID} blocks={NL}; victim={args.victim}")
    print(f"CONVENTION: 'layer L' = output of L-th transformer block (post attn+MLP+residual) = hidden_states[L], "
          f"1-indexed (hidden_states[0]=embeddings). attacked shallow = hidden_states[2],[4],[6]; "
          f"LLaVA reads hidden_states[-2]=hidden_states[{NL - 1}] (block {NL - 1}); profile L=1..{NL}. "
          f"post-block RESIDUAL STREAM (not V).")
    print(f"eps={args.eps} L2, seeds={args.seeds}, backbone={args.backbone}, w_inter={args.w_inter}, "
          f"weights w1/w2/w3={args.w1}/{args.w2}/{args.w3}")

    arms = list(ARMS.keys()) if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.backbone == "off":
        arms = [a for a in arms if ARMS[a]]  # base is empty -> nothing to optimize with backbone off
    inter_pool = sorted({l for a in arms for l in ARMS[a]})
    print(f"arms={arms}; shallow-layer pool={inter_pool}")

    def txt_emb(words):
        tk = cproc(text=words, return_tensors="pt", padding=True).to(DEVICE)
        return F.normalize(clip.text_projection(clip.text_model(**tk).pooler_output), dim=-1)

    def vis(x01, want_attn=False):
        out = clip.vision_model((x01 - MEAN) / STD, interpolate_pos_encoding=True,
                                output_hidden_states=True, output_attentions=want_attn)
        cls = F.normalize(clip.visual_projection(out.pooler_output), dim=-1)
        A = None
        if want_attn:
            att = torch.stack(out.attentions).mean(0).mean(1)
            A = att[:, 0, 1:]
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

    def roiP_profile(x01, obj_idx, phi):
        """per-layer object-region roiP = mean cos(visual_projection(patch_L), phi), L=1..NL."""
        with torch.no_grad():
            out = vis(x01)
            return [(F.normalize(clip.visual_projection(out["hs"][L][:, 1:].squeeze(0)), dim=-1)[obj_idx] @ phi).mean().item()
                    for L in range(1, NL + 1)]

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    objects = [o.strip() for o in args.objects.split(",")]
    samples = [s for s in samples if s["object"] in objects]
    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "adv")
    os.makedirs(imgdir, exist_ok=True)

    ans_rows = [["arm", "object", "eps", "seed", "criterion", "prompt", "answer_text", "object_present", "degenerate", "linf"]]
    curve_rows = [["arm", "object", "seed", "iter", "L_cls", "L_p", "L_fix", "L_p_inter"]]
    roiP_rows = [["object", "seed", "layer", "kind", "roiP"]]
    adv_store = {}   # object -> list of adv tensors from profile arm, per seed
    clean_prof = {}  # object -> clean roiP list (len NL)
    obj_meta = {}    # object -> (obj_idx, phi) for the propagation profile
    printed_shapes = False

    for s in samples:
        obj, sid = s["object"], s["id"]
        if not os.path.exists(s["mask"]):
            print(f"[{sid}] no mask; skip"); continue
        if not (s.get("target") and os.path.exists(s["target"])):
            print(f"[{sid}] no inpaint target; skip (this experiment requires the inpaint counterfactual)"); continue
        x0 = load01(s["image"])
        xR = load01(s["target"])                       # inpaint removal target
        E_t = txt_emb([f"a photo of a {obj}"])[0]
        phi = txt_emb([obj])[0]                        # bare-word text emb for the propagation profile (per spec)

        clean = vis(x0, want_attn=True)
        featR = vis(xR)
        if not printed_shapes:
            print(f"SHAPES: patch_last={tuple(clean['last'].shape)} attn A(CLS->patch)={tuple(clean['attn'].shape)} "
                  f"#hidden_states={len(clean['hs'])}")
            printed_shapes = True
        clean_last_n = F.normalize(clean["last"].squeeze(0), dim=-1).detach()
        A_clean = clean["attn"][0].detach()
        featR_last_n = F.normalize(featR["last"].squeeze(0), dim=-1).detach()               # block-24 target (backbone L_p)
        featR_layer = {l: F.normalize(featR["hs"][l][:, 1:].squeeze(0), dim=-1).detach()    # shallow-layer targets (frozen)
                       for l in inter_pool}

        # segmentation -> token-grid object region
        mg = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.BILINEAR), np.float32) / 255
        obj_idx = torch.from_numpy((mg.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3)) > 0.3).reshape(-1)).to(DEVICE)
        bg_idx = ~obj_idx
        obj_meta[obj] = (obj_idx, phi)
        print(f"\n=== {sid} '{obj}' ({int(obj_idx.sum())}/{GRID * GRID} obj tokens) ===")

        prompts = {"direct": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
                   "list": "List all objects you see in this image.",
                   "presup": f"What {CAT.get(obj, 'object')} is in this image?",
                   "detail": "Describe this image in detail."}
        weak_prompt = "Describe this image."

        # ground-truth "removed" reference: what LLaVA says about the inpaint image itself
        ref_cap = gen(xR, weak_prompt)
        ans_rows.append(["_inpaint_ref", obj, 0, 0, "weak", "describe", ref_cap.replace("\n", " ")[:200],
                         present(ref_cap, obj), degenerate(ref_cap), 0])
        print(f"  inpaint-ref caption: {ref_cap[:80]!r} (object_present={present(ref_cap, obj)})")

        # clean propagation profile (seed-independent)
        clean_prof[obj] = roiP_profile(x0, obj_idx, phi)
        for L in range(1, NL + 1):
            roiP_rows.append([obj, -1, L, "clean", round(clean_prof[obj][L - 1], 5)])

        for arm in arms:
            arm_layers = ARMS[arm]
            for seed in range(args.seeds):
                g = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
                delta = (torch.rand(x0.shape, generator=g, device=DEVICE) * 2 - 1) * (args.eps / (RES * 1.7))
                delta = delta.detach().requires_grad_(True)
                opt = torch.optim.Adam([delta], lr=args.lr)
                for it in range(args.iters):
                    x = torch.clamp(x0 + delta, 0, 1)
                    f = vis(x, want_attn=False)
                    L_cls = (f["cls"].squeeze(0) @ E_t)
                    pl = F.normalize(f["last"].squeeze(0), dim=-1)
                    Lp = -(pl[obj_idx] * featR_last_n[obj_idx]).sum(-1).mean()
                    w = A_clean[bg_idx]; w = w / w.sum().clamp_min(1e-9)
                    Lfix = -(w * (pl[bg_idx] * clean_last_n[bg_idx]).sum(-1)).sum()
                    Lp_inter = x0.new_zeros(())
                    if arm_layers:
                        for l in arm_layers:
                            pil = F.normalize(f["hs"][l][:, 1:].squeeze(0), dim=-1)
                            Lp_inter = Lp_inter - (pil[obj_idx] * featR_layer[l][obj_idx]).sum(-1).mean()
                        Lp_inter = Lp_inter / len(arm_layers)
                    if args.backbone == "on":
                        L = args.w1 * L_cls + args.w2 * Lp + args.w3 * Lfix + args.w_inter * Lp_inter
                    else:
                        L = args.w_inter * Lp_inter
                    opt.zero_grad(); L.backward(); opt.step()
                    with torch.no_grad():
                        n = delta.flatten().norm(2)
                        if n > args.eps:
                            delta.mul_(args.eps / n)
                        delta.data = torch.clamp(x0 + delta, 0, 1) - x0
                    if it % 10 == 0 or it == args.iters - 1:
                        curve_rows.append([arm, obj, seed, it, round(L_cls.item(), 4), round(Lp.item(), 4),
                                           round(Lfix.item(), 4), round(float(Lp_inter), 4)])

                adv = torch.clamp(x0 + delta.detach(), 0, 1)
                linf = round(delta.detach().abs().max().item() * 255, 1)
                Image.fromarray((adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                    f"{imgdir}/{arm}_{obj}_s{seed}.png")
                if arm == args.profile_arm:
                    adv_store.setdefault(obj, []).append(adv.detach())

                # WEAK + STRICT eval, full captions + degeneracy flag
                wa = gen(adv, weak_prompt)
                ans_rows.append([arm, obj, args.eps, seed, "weak", "describe", wa.replace("\n", " ")[:200],
                                 present(wa, obj), degenerate(wa), linf])
                strict_present = False
                any_degen = degenerate(wa)
                for pk, pt in prompts.items():
                    a = gen(adv, pt); p = present(a, obj); strict_present = strict_present or p
                    any_degen = any_degen or degenerate(a)
                    ans_rows.append([arm, obj, args.eps, seed, "strict", pk, a.replace("\n", " ")[:200], p, degenerate(a), linf])
                print(f"  [{arm:<9}] {obj} seed={seed} Linf={linf:>5}/255 | "
                      f"weak_absent={not present(wa, obj)} strict_absent={not strict_present} degenerate={any_degen}")

    # ---- write raw logs (roiP written after Part 2 appends adv rows) ----
    for name, rows in [("answers", ans_rows), ("curves", curve_rows)]:
        with open(f"{args.outdir}/{name}.csv", "w", newline="") as fp:
            csv.writer(fp).writerows(rows)

    # ================= PART 1: ablation summary =================
    from collections import defaultdict
    grp = defaultdict(lambda: {"weak": defaultdict(list), "strict": defaultdict(list),
                               "degen_removed": [], "linf": []})
    for r in ans_rows[1:]:
        arm, obj, eps, seed, crit, prompt, txt, pres, degen, linf = r
        if arm == "_inpaint_ref":
            continue
        pres = (pres is True or pres == "True"); degen = (degen is True or degen == "True")
        grp[(arm, obj)][crit][seed].append(pres)
        if crit == "weak":
            grp[(arm, obj)]["linf"].append(float(linf))
    # degeneracy among "removed" strict cases: did concealment coincide with corrupted output?
    strict_by = defaultdict(lambda: defaultdict(lambda: {"present": False, "degen": False}))
    for r in ans_rows[1:]:
        arm, obj, eps, seed, crit, prompt, txt, pres, degen, linf = r
        if arm == "_inpaint_ref" or crit != "strict":
            continue
        pres = (pres is True or pres == "True"); degen = (degen is True or degen == "True")
        strict_by[(arm, obj)][seed]["present"] |= pres
        strict_by[(arm, obj)][seed]["degen"] |= degen

    print("\n=== PART 1: layer ablation (eps={:.0f} L2, {} seeds) ==="
          .format(args.eps, args.seeds))
    print(f"{'arm':<10}{'object':<6}{'weak_rem%':>10}{'strict_rem%':>12}{'mean_Linf':>10}{'degen_of_removed':>18}")
    summ = [["arm", "object", "eps", "weak_removal_rate", "strict_removal_rate", "mean_linf", "degen_removed_frac", "n"]]
    for (arm, obj) in sorted(grp.keys()):
        d = grp[(arm, obj)]
        weak_rem = 100 * np.mean([not any(v) for v in d["weak"].values()])
        strict_rem = 100 * np.mean([not any(v) for v in d["strict"].values()])
        removed = [sd for sd, v in strict_by[(arm, obj)].items() if not v["present"]]
        degen_removed = np.mean([strict_by[(arm, obj)][sd]["degen"] for sd in removed]) if removed else 0.0
        mean_linf = float(np.mean(d["linf"])) if d["linf"] else 0.0
        n = len(d["strict"])
        summ.append([arm, obj, args.eps, round(weak_rem, 1), round(strict_rem, 1),
                     round(mean_linf, 1), round(float(degen_removed), 2), n])
        print(f"{arm:<10}{obj:<6}{weak_rem:>9.0f}%{strict_rem:>11.0f}%{mean_linf:>10.1f}{degen_removed:>17.0%}")
    with open(f"{args.outdir}/ablation_summary.csv", "w", newline="") as fp:
        csv.writer(fp).writerows(summ)
    print("VALIDITY: 'degen_of_removed'>0 means some 'removals' had corrupted/degenerate captions (destruction, "
          "not concealment). If 'base' shows high strict_rem too, the eval is not discriminating. Compare adv PNGs "
          f"in {imgdir} and the _inpaint_ref caption in answers.csv against the 'removed' captions.")

    # ---- loss-curve plots (mean over seeds), one figure per object ----
    for obj in objects:
        rows = [r for r in curve_rows[1:] if r[1] == obj]
        if not rows:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
        for ax, arm in zip(axes.ravel(), ARMS.keys()):
            ar = [r for r in rows if r[0] == arm]
            if not ar:
                ax.set_visible(False); continue
            its = sorted({r[3] for r in ar})
            for j, lab in [(4, "L_cls"), (5, "L_p"), (6, "L_fix"), (7, "L_p_inter")]:
                ys = [np.mean([r[j] for r in ar if r[3] == it]) for it in its]
                ax.plot(its, ys, label=lab)
            ax.set_title(f"{arm}"); ax.set_xlabel("iter"); ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f"Loss curves ({obj}, eps={args.eps:.0f} L2, mean over {args.seeds} seeds)")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/loss_{obj}.png", dpi=110); plt.close(fig)

    # ================= PART 2: propagation profile =================
    layers = list(range(1, NL + 1))
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"dog": "tab:blue", "cat": "tab:orange"}
    print("\n=== PART 2: propagation profile (roiP, mean over seeds) ===")
    for obj in objects:
        if obj not in adv_store or obj not in clean_prof:
            print(f"  [{obj}] no profile-arm adv images; skip"); continue
        obj_idx, phi = obj_meta[obj]
        adv_curves = [roiP_profile(adv, obj_idx, phi) for adv in adv_store[obj]]
        for seed, cur in enumerate(adv_curves):
            for L in range(1, NL + 1):
                roiP_rows.append([obj, seed, L, "adv", round(cur[L - 1], 5)])
        adv_arr = np.array(adv_curves); clean_arr = np.array(clean_prof[obj])
        m, sd = adv_arr.mean(0), adv_arr.std(0)
        c = colors.get(obj, "tab:green")
        ax.plot(layers, clean_arr, "-", color=c, label=f"{obj} clean")
        ax.plot(layers, m, "--", color=c, label=f"{obj} adv")
        ax.fill_between(layers, m - sd, m + sd, color=c, alpha=0.18)
        recov = m[NL - 2] - m.min()   # block-23 vs the minimum after attack
        print(f"  {obj}: clean roiP@[2,4,6,23]={[round(clean_arr[k - 1], 3) for k in [2, 4, 6, 23]]} "
              f"adv roiP@[2,4,6,23]={[round(m[k - 1], 3) for k in [2, 4, 6, 23]]} "
              f"min={m.min():.3f}@L{int(m.argmin()) + 1} recovery(adv23-min)={recov:+.3f} "
              f"=> {'RESIDUAL RESTORES signal' if recov > 0.03 else 'corruption PROPAGATES'}")
    for L in [2, 4, 6]:
        ax.axvline(L, ls=":", color="gray", lw=1)
    ax.axvline(NL - 1, ls="-.", color="red", lw=1.2, label=f"LLaVA reads (block {NL - 1})")
    ax.set_xlabel("transformer block L (post-block residual)"); ax.set_ylabel("roiP = mean cos(proj(patch), phi_T(obj))")
    ax.set_title(f"Per-layer object-region roiP: clean vs adv ({args.profile_arm}, eps={args.eps:.0f} L2, {args.seeds} seeds)\n"
                 "dotted = attacked layers (2,4,6); visual_projection on shallow layers is heuristic (relative drop is valid)")
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/propagation_profile.png", dpi=130); plt.close(fig)

    with open(f"{args.outdir}/roiP.csv", "w", newline="") as fp:  # rewrite with adv rows appended
        csv.writer(fp).writerows(roiP_rows)
    print(f"\nsaved {args.outdir}/{{answers,curves,roiP,ablation_summary}}.csv, "
          f"propagation_profile.png, loss_<obj>.png, adv/*.png")


if __name__ == "__main__":
    main()
