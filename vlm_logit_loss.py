"""Stage 22: end-to-end white-box attack on LLaVA's ANSWER LOGIT (no CLIP proxy).

Every prior attack optimized a CLIP-encoder proxy (CLS / dense / value / penult /
crop-embedding) and hoped it propagated to the decoder. This optimizes the exact
quantity we're graded on: LLaVA-1.5's next-token logit for "yes" vs "no" in
response to "Is there a {object}?". Minimize P(yes) - P(no) by PGD/MI-FGSM on the
image pixels, backprop through encoder -> projector -> LLM.

    loss = logit["Yes"] - logit["No"]   (minimize -> flip the answer to "No")

Runs per-object over dataset.jsonl and reports, per (object, budget): P(yes)
before/after, the greedy answer before/after, and a describe cross-check -- so
we see the predicted soft/hard split (dog/cat may flip, sign/car may not).

Two clean outcomes:
  * still fails on hard objects at high budget -> the wall is fundamental; no
    attack in this family removes them.
  * succeeds where CLIP-cosine failed -> the proxy decoupling was the problem
    and this is a working white-box concealment method.

Perturbation region via --pad (fraction of RES around the mask bbox; 1.0 = whole
image, the max-capacity default). Attacks + evaluates in one loaded model.

Example:
    uv run python vlm_logit_loss.py --dataset dataset.jsonl --budgets 8,16,32 --steps 150
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--source", default=None)
    ap.add_argument("--object", default="dog")
    ap.add_argument("--mask", default=None)
    ap.add_argument("--input-res", type=int, default=336)
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--pad", type=float, default=1.0, help="perturb region: frac of RES around bbox; 1.0=whole image")
    ap.add_argument("--outdir", default="results/vlm_logit")
    args = ap.parse_args()
    RES = args.input_res

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map=DEVICE).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    ip = processor.image_processor
    MEAN = torch.tensor(ip.image_mean, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)
    STD = torch.tensor(ip.image_std, device=DEVICE, dtype=DTYPE).view(1, 3, 1, 1)

    def ids_for(words):
        s = set()
        for w in words:
            t = tok(w, add_special_tokens=False).input_ids
            if t:
                s.add(t[0])
        return sorted(s)
    YES = ids_for(["Yes", "yes", " Yes", " yes"])
    NO = ids_for(["No", "no", " No", " no"])
    print(f"yes ids={YES}  no ids={NO}")

    os.makedirs(args.outdir, exist_ok=True)
    sqdir = os.path.join(args.outdir, "square")
    os.makedirs(sqdir, exist_ok=True)

    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset) as f:
            samples = [json.loads(l) for l in f if l.strip()]
    else:
        sid = os.path.splitext(os.path.basename(args.source))[0]
        samples = [{"id": sid, "image": args.source, "object": args.object, "mask": args.mask}]

    def build_inputs(img336, obj):
        q = f"Is there a {obj} in this image? Answer with only 'yes' or 'no'."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(images=img336, text=prompt, return_tensors="pt").to(DEVICE)

    @torch.no_grad()
    def answer(pixel_values, input_ids, attn, prompt_len, max_new=24):
        gen = model.generate(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values,
                             max_new_tokens=max_new, do_sample=False)
        return tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()

    rows = ["id,object,budget,p_yes_before,p_yes_after,ans_before,ans_after,detected_after"]
    for s in samples:
        obj, sid = s["object"], s["id"]
        image = Image.open(s["image"]).convert("RGB")
        W0, H0 = image.size
        img336 = image.resize((RES, RES), Image.BICUBIC)
        x0 = torch.from_numpy(np.asarray(img336, np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)

        # perturbation region (in 336 space)
        region = torch.ones((1, 1, RES, RES), device=DEVICE, dtype=DTYPE)
        if args.pad < 1.0 and s.get("mask") and os.path.exists(s["mask"]):
            M = np.array(Image.open(s["mask"]).convert("L").resize((RES, RES), Image.NEAREST)) > 127
            if M.sum() > 0:
                ys, xs = np.nonzero(M); pad = int(args.pad * RES)
                region = torch.zeros((1, 1, RES, RES), device=DEVICE, dtype=DTYPE)
                region[:, :, max(0, ys.min() - pad):min(RES, ys.max() + pad + 1),
                       max(0, xs.min() - pad):min(RES, xs.max() + pad + 1)] = 1.0

        inp = build_inputs(img336, obj)
        input_ids, attn = inp["input_ids"], inp["attention_mask"]
        prompt_len = input_ids.shape[1]
        pv_shape = inp["pixel_values"].shape
        assert pv_shape[-2:] == (RES, RES), f"unexpected pixel_values shape {pv_shape} (LLaVA-1.6 AnyRes?)"

        def logits_next(x01):
            pv = ((x01 - MEAN) / STD).to(DTYPE)
            out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv)
            return out.logits[0, -1]

        with torch.no_grad():
            lg0 = logits_next(x0).float()
            p = lg0.softmax(-1)
            p_yes0 = p[YES].sum().item()
            ans0 = answer(((x0 - MEAN) / STD).to(DTYPE), input_ids, attn, prompt_len)
        print(f"\n=== {sid} '{obj}' === baseline P(yes)={p_yes0:.3f}  answer={ans0[:40]!r}")

        for b in [float(x) for x in args.budgets.split(",")]:
            eps = b / 255.0
            alpha = max(1.0 / 255, 2.5 * eps / args.steps)
            delta = torch.zeros_like(x0, requires_grad=True)
            momentum = torch.zeros_like(x0)
            for _ in range(args.steps):
                x = torch.clamp(x0 + delta * region, 0, 1)
                lg = logits_next(x)
                loss = (lg[YES].max() - lg[NO].max())            # minimize -> flip to "no"
                g, = torch.autograd.grad(loss, delta)
                with torch.no_grad():
                    momentum.mul_(0.9).add_(g / g.abs().mean().clamp_min(1e-12))
                    delta.add_(-alpha * momentum.sign() * region).clamp_(-eps, eps)
                    delta.data = torch.clamp(x0 + delta * region, 0, 1) - x0
                delta.requires_grad_(True)

            adv = torch.clamp(x0 + delta.detach() * region, 0, 1)
            with torch.no_grad():
                p_yes = logits_next(adv).float().softmax(-1)[YES].sum().item()
                ans = answer(((adv - MEAN) / STD).to(DTYPE), input_ids, attn, prompt_len)
            detected = ("yes" in ans.lower()) or (obj.lower() in ans.lower())
            # save square (what the model saw) + full-res composite
            sq = (adv.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(sq).save(f"{sqdir}/{sid}_eps{int(b)}.png")
            up = F.interpolate((adv - x0).float(), size=(H0, W0), mode="bicubic", align_corners=False)
            full = (torch.from_numpy(np.asarray(image, np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
                    + up.cpu()).clamp(0, 1)
            Image.fromarray((full.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                f"{args.outdir}/{sid}_eps{int(b)}.png")
            rows.append(f"{sid},{obj},{b:g},{p_yes0:.4f},{p_yes:.4f},{ans0[:30].replace(chr(10),' ')},"
                        f"{ans[:30].replace(chr(10),' ')},{detected}")
            print(f"  eps={b:>4.0f}/255  P(yes) {p_yes0:.3f}->{p_yes:.3f}  answer={ans[:40]!r}  detected={detected}")

    with open(f"{args.outdir}/metrics.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    # per-object flip summary
    print("\n=== end-to-end logit attack: did the answer flip? ===")
    print(f"{'object':<18}{'budgets flipped (answer=no)':>30}")
    import csv
    from collections import defaultdict
    by = defaultdict(list)
    for r in csv.DictReader(open(f"{args.outdir}/metrics.csv")):
        by[r["object"]].append((r["budget"], r["detected_after"] == "False"))
    for obj, lst in by.items():
        flipped = [b for b, f in lst if f]
        print(f"{obj:<18}{(','.join(flipped) if flipped else 'none'):>30}")
    print(f"\nsaved {args.outdir}/metrics.csv + adv images")


if __name__ == "__main__":
    main()
