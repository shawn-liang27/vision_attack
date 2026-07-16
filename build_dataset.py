"""Scan dataset/ and write dataset.jsonl (one row per sample).

Object label is inferred from the filename (e.g. 'fire_hydrant_2.jpg' -> 'fire
hydrant'); VERIFY/EDIT the printed labels in dataset.jsonl before prep_dataset,
since the SAM3 mask prompt and the VLM detection word both come from it.

    uv run python build_dataset.py --dir dataset
"""

import argparse
import glob
import json
import os
import re


def infer_object(stem):
    s = re.sub(r"[_\-]+", " ", stem)          # separators -> space
    s = re.sub(r"\d+", " ", s)                # drop digits
    return " ".join(s.split()).strip().lower() or stem.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset")
    ap.add_argument("--out", default="dataset.jsonl")
    args = ap.parse_args()

    paths = sorted(sum([glob.glob(os.path.join(args.dir, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")], []))
    if not paths:
        raise SystemExit(f"no images in {args.dir}/")

    rows = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        obj = infer_object(stem)
        rows.append({
            "id": stem,
            "image": p,
            "object": obj,
            "mask": f"masks/{stem}_mask.png",
            "target": f"targets/{stem}_removed.png",
        })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {args.out} ({len(rows)} samples). VERIFY the inferred objects:\n")
    print(f"  {'id':<24}{'object (edit if wrong)':<24}image")
    for r in rows:
        print(f"  {r['id']:<24}{r['object']:<24}{r['image']}")
    print("\nedit the 'object' fields in dataset.jsonl if any are wrong, then run prep_dataset.py")


if __name__ == "__main__":
    main()
