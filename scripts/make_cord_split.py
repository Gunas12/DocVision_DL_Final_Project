"""Write the frozen CORD split manifest to data/splits/cord_split_v1.json.

The split is the official CORD v2 split (800/100/100). It is NOT re-sampled;
the manifest only records sample ids and content fingerprints so that the split
is committed, reviewable and checkable for leakage.

Usage: python scripts/make_cord_split.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_from_disk

from docvision.data.cord import DEFAULT_ROOT, SPLITS, make_manifest

OUT = ROOT / "data" / "splits" / "cord_split_v1.json"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists. The split is frozen; do not regenerate it.")
    ds = load_from_disk(str(ROOT / DEFAULT_ROOT))
    raw = {split: list(ds[split]["ground_truth"]) for split in SPLITS}
    manifest = make_manifest(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT, manifest["sizes"])


if __name__ == "__main__":
    main()