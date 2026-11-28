"""Build OCR-noisy evaluation sets (low / medium / high) from clean CORD splits.

Output: data/processed/noisy_eval/<split>_<level>.jsonl (git-ignored, deterministic, can be regenerated).
Distractor words are drawn from the *train* split only.

By default only the validation split is built. The test variants carry no model information and are
deterministic, but follow the test protocol: build them with ``--splits test`` only on the day the TEST run
is prepared (user#4), at the frozen commit.

Usage:
    python scripts/make_noisy_eval.py                    # validation, all levels
    python scripts/make_noisy_eval.py --splits validation test
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvision.data.augment import (
    LEVELS,
    add_ocr_noise,
    build_distractor_pool,
    write_samples_jsonl,
)
from docvision.data.cord import load_cord_split

SEED = 13
OUT = ROOT / "data" / "processed" / "noisy_eval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["validation"], choices=["validation", "test"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pool = build_distractor_pool(list(load_cord_split("train", ROOT / "data" / "raw" / "cord-v2")))
    print(f"distractor pool: {len(pool)} words from train")
    for split in args.splits:
        clean = list(load_cord_split(split, ROOT / "data" / "raw" / "cord-v2"))
        for name, cfg in LEVELS.items():
            noisy = [add_ocr_noise(s, cfg, args.seed, pool) for s in clean]
            path = OUT / f"{split}_{name}.jsonl"
            write_samples_jsonl(noisy, path)
            print(f"{path.name}: {len(noisy)} samples, mean words {sum(len(s.words) for s in noisy) / len(noisy):.1f}")
    cfg_out = {"seed": args.seed, "levels": {k: v.__dict__ for k, v in LEVELS.items()}}
    (OUT / "noise_config.json").write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()