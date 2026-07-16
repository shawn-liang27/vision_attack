"""Stage 9: does an actual VLM still see the dog? (transfer test)

The PGD attacks are white-box on CLIP ViT-B/16. This feeds the saved adversarial
images to real open ~7B VLMs and asks whether they still detect the dog -- the
non-circular judge the CLIP probes could not provide. Gemini already recognized
the dog in every image; this quantifies it on open models, including LLaVA
(whose CLIP ViT-L/14-336 encoder is the closest architecture to what we
attacked, so the most favorable case for transfer).

For every image x prompt x model it records the answer, flags whether the object
was detected, and writes a markdown table + JSON.

Examples:
    uv run python vlm_eval.py                       # default 7B models
    uv run python vlm_eval.py --models Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import glob
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

DEFAULT_MODELS = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf",  # CLIP ViT-L/14-336 encoder (closest to attacked)
]


def ask(model, processor, image, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="results/adv_images")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--object", default="dog")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    obj = args.object

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")
    print(f"{len(paths)} images, {len(args.models.split(','))} model(s)")

    prompts = {
        "binary": f"Is there a {obj} in this image? Answer with only 'yes' or 'no'.",
        "describe": "Describe this image in one sentence.",
        "animals": "What animals, if any, are in this image?",
    }

    results = {}  # model -> image -> {prompt: answer, detected: bool}
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== loading {model_id} ===")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=DTYPE, device_map="auto").eval()
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"  failed to load {model_id}: {type(e).__name__}: {e}")
            continue

        results[model_id] = {}
        for p in paths:
            name = os.path.basename(p)
            img = Image.open(p).convert("RGB")
            ans = {k: ask(model, processor, img, q, args.max_new_tokens) for k, q in prompts.items()}
            text_blob = " ".join(ans.values()).lower()
            detected = ("yes" in ans["binary"].lower()) or (obj in text_blob)
            ans["detected"] = detected
            results[model_id][name] = ans
            print(f"  {name:<28} detected={detected}  binary={ans['binary'][:20]!r}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/vlm_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # markdown: rows = images, columns = per-model detected flag; then details
    with open(f"{args.outdir}/vlm_eval.md", "w") as f:
        f.write(f"# VLM transfer test: does the VLM still see the {obj}?\n\n")
        models = list(results.keys())
        f.write("| image | " + " | ".join(m.split("/")[-1] for m in models) + " |\n")
        f.write("|" + "---|" * (len(models) + 1) + "\n")
        for name in sorted({n for m in models for n in results[m]}):
            cells = []
            for m in models:
                d = results[m].get(name, {}).get("detected")
                cells.append("🐕 seen" if d else ("— missed" if d is False else "n/a"))
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
        f.write("\n## Detection rate per model\n\n")
        for m in models:
            det = sum(v["detected"] for v in results[m].values())
            f.write(f"- **{m}**: {det}/{len(results[m])} images still show the {obj}\n")
        f.write("\n## Full answers\n\n")
        for m in models:
            f.write(f"### {m}\n\n")
            for name, ans in results[m].items():
                f.write(f"**{name}** — detected={ans['detected']}\n")
                for k in ("binary", "describe", "animals"):
                    f.write(f"  - *{k}*: {ans[k]}\n")
                f.write("\n")

    print("\nsaved results/vlm_eval.{json,md}")
    for m in results:
        det = sum(v["detected"] for v in results[m].values())
        print(f"  {m}: {det}/{len(results[m])} still show the {obj}")


if __name__ == "__main__":
    main()
