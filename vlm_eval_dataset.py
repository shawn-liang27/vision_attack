"""Dataset-aware VLM eval: per image, ask about ITS object (from the id prefix).

edge_gen images are named <id>__<arm>_seed<s>.png / <id>__baseline.png. This maps
each image to its object via dataset.jsonl (matching the id prefix) and asks
"Is there a {object}?" + a free-form describe, flagging detection per the image's
own object -- needed because the dataset has different objects per sample.

    uv run python vlm_eval_dataset.py --images-dir results/edge_gen \
        --dataset dataset.jsonl --models llava-hf/llava-1.5-7b-hf
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


def ask(model, processor, image, prompt, max_new_tokens=64):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="results/edge_gen")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--models", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.images_dir

    with open(args.dataset) as f:
        obj_of = {json.loads(l)["id"]: json.loads(l)["object"] for l in f if l.strip()}
    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")

    def object_for(name):  # <id>__... ; id may itself contain no "__"
        sid = name.split("__")[0]
        return sid, obj_of.get(sid)

    results = {}
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== loading {model_id} ===")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=DTYPE, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(model_id)
        results[model_id] = {}
        for p in paths:
            name = os.path.basename(p)
            sid, obj = object_for(name)
            if obj is None:
                print(f"  {name}: no object mapping, skipping"); continue
            img = Image.open(p).convert("RGB")
            binary = ask(model, processor, img, f"Is there a {obj} in this image? Answer only 'yes' or 'no'.")
            desc = ask(model, processor, img, "Describe this image in one sentence.")
            detected = ("yes" in binary.lower()) or (obj.lower() in desc.lower())
            results[model_id][name] = {"object": obj, "binary": binary, "describe": desc, "detected": detected}
            print(f"  {name:<34} obj={obj:<14} detected={detected}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/vlm_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {outdir}/vlm_eval.json")


if __name__ == "__main__":
    main()
