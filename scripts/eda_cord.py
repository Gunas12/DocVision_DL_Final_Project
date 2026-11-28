"""Print a small EDA of CORD (label counts, lengths, group checks) as Markdown.

Usage: python scripts/eda_cord.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvision.data.cord import SPLITS, load_cord_split
from docvision.data.schema import FIELDS


def main() -> None:
    entities = {s: Counter() for s in SPLITS}
    lengths: dict[str, list[int]] = {s: [] for s in SPLITS}
    items: dict[str, list[int]] = {s: [] for s in SPLITS}
    no_total = Counter()
    bad_groups = Counter()
    for split in SPLITS:
        for sample in load_cord_split(split):
            lengths[split].append(len(sample.words))
            for label in sample.labels:
                if label.startswith("B-"):
                    entities[split][label[2:]] += 1
            groups = {g for g, lab in zip(sample.group_ids, sample.labels) if lab == "B-ITEM_NAME"}
            n_names = sum(lab == "B-ITEM_NAME" for lab in sample.labels)
            items[split].append(n_names)
            if len(groups) != n_names:
                bad_groups[split] += 1
            if "B-TOTAL" not in sample.labels:
                no_total[split] += 1

    print("| Field | " + " | ".join(SPLITS) + " |")
    print("|---|" + "---|" * len(SPLITS))
    for field in FIELDS:
        print(f"| {field} | " + " | ".join(str(entities[s][field]) for s in SPLITS) + " |")
    print()
    print("| Split | Receipts | Words/receipt (mean / max) | Items/receipt (mean) | No TOTAL | Item/group mismatch |")
    print("|---|---|---|---|---|---|")
    for s in SPLITS:
        n = len(lengths[s])
        print(
            f"| {s} | {n} | {sum(lengths[s]) / n:.1f} / {max(lengths[s])} | "
            f"{sum(items[s]) / n:.1f} | {no_total[s]} | {bad_groups[s]} |"
        )


if __name__ == "__main__":
    main()