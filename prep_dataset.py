"""Per-sample prep for the generalization run: SAM3 mask + inpainted target.

For each dataset.jsonl row, generates (if missing):
  * masks/<id>_mask.png    -- SAM3 text-prompted segmentation of the object
  * targets/<id>_removed.png -- SDXL inpaint removing the object (M-Attack target)

Reuses generate_mask.py and generate_removed.py as subprocesses (they own the
SAM3 / SDXL model loading). Skips a sample's stage if the output already exists.

    uv run python prep_dataset.py --dataset dataset.jsonl
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--dilate", type=int, default=24)
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = ap.parse_args()

    with open(args.dataset) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for r in rows:
        img, obj, mask, tgt = r["image"], r["object"], r["mask"], r["target"]
        print(f"\n=== {r['id']}  object='{obj}' ===")

        if args.force or not os.path.exists(mask):
            print(f"  SAM3 mask -> {mask}")
            subprocess.run([sys.executable, "generate_mask.py",
                            "--image", img, "--prompt", obj, "--out", mask], check=True)
        else:
            print(f"  mask exists ({mask})")

        if args.force or not os.path.exists(tgt):
            print(f"  SDXL inpaint target -> {tgt}")
            subprocess.run([sys.executable, "generate_removed.py",
                            "--image", img, "--mask", mask, "--out", tgt,
                            "--prompt", f"empty background where the {obj} was, natural scene continuation",
                            "--negative", f"{obj}, animal, person, object",
                            "--dilate", str(args.dilate)], check=True)
        else:
            print(f"  target exists ({tgt})")

    print(f"\nprep done for {len(rows)} samples. Next: m_attack_edge.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
