"""Download CORD v2 (CC BY 4.0) from Hugging Face into data/raw/cord-v2.

Usage: python scripts/download_data.py
"""

import json
from pathlib import Path

from datasets import load_dataset, load_from_disk

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cord-v2"


def main() -> None:
    if OUT.exists():
        print(f"CORD already present at {OUT} (delete the folder to re-download).")
    else:
        ds = load_dataset("naver-clova-ix/cord-v2")  # train 800 / validation 100 / test 100
        ds.save_to_disk(str(OUT))
        print({split: len(d) for split, d in ds.items()})

    # Sanity check: print the structure of one training sample.
    ds = load_from_disk(str(OUT))
    sample = ds["train"][0]
    gt = json.loads(sample["ground_truth"])
    print("columns:", ds["train"].column_names)
    print("image size:", sample["image"].size)
    print("ground_truth keys:", list(gt.keys()))
    print("first valid_line:", json.dumps(gt["valid_line"][0], ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()