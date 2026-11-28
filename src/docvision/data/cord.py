"""CORD v2 loader: words, boxes, BIO labels and line-item groups.

Raw data lives in ``data/raw/cord-v2`` (see ``scripts/download_data.py``).
Each example has an ``image`` and a ``ground_truth`` JSON string whose
``valid_line`` entries look like::

    {"words": [{"quad": {...}, "is_key": 0, "row_id": 1, "text": "1"}, ...],
     "category": "menu.cnt", "group_id": 3, "sub_group_id": 0}

Conventions:
* Words are put in reading order: rows sorted top to bottom, words in a row
  left to right (``row_id`` identifies a visual row).
* Boxes are ``[x0, y0, x1, y1]`` normalised to 0-1000 (LayoutLM convention).
* Only value words (``is_key == 0``) get a field label; key words such as the
  printed text "TOTAL" are labelled ``O``.
* Within one ``valid_line`` the first word is ``B-<FIELD>``, the rest ``I-<FIELD>``.
* ``group_ids`` is the CORD ``group_id`` for line-item words, ``-1`` otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docvision.data.schema import LINE_ITEM_FIELDS, OUTSIDE, cord_category_to_field

DEFAULT_ROOT = Path("data/raw/cord-v2")
SPLITS = ("train", "validation", "test")
NO_GROUP = -1


@dataclass
class CordSample:
    sample_id: str
    split: str
    index: int  # row index inside the split (use it to fetch the image)
    width: int
    height: int
    words: list[str]
    boxes: list[list[int]]
    labels: list[str]
    group_ids: list[int]


def quad_to_box(quad: dict[str, int], width: int, height: int) -> list[int]:
    """Convert a CORD quad to a [x0, y0, x1, y1] box normalised to 0-1000."""
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]

    def norm(v: int, size: int) -> int:
        return max(0, min(1000, round(1000 * v / size)))

    return [norm(min(xs), width), norm(min(ys), height), norm(max(xs), width), norm(max(ys), height)]


def parse_ground_truth(gt: dict[str, Any], width: int, height: int) -> dict[str, list]:
    """Turn a CORD ``ground_truth`` dict into aligned word/box/label/group lists."""
    records: list[dict[str, Any]] = []
    for line_idx, line in enumerate(gt["valid_line"]):
        field = cord_category_to_field(line["category"])
        for word in line["words"]:
            text = word["text"].strip()
            if not text:
                continue
            quad = word["quad"]
            records.append(
                {
                    "text": text,
                    "box": quad_to_box(quad, width, height),
                    "row_id": word["row_id"],
                    "field": None if word.get("is_key", 0) else field,
                    "line_idx": line_idx,
                    "group_id": line.get("group_id", NO_GROUP),
                }
            )

    # Reading order: rows by mean vertical centre, words left to right inside a row.
    row_y: dict[Any, list[float]] = {}
    for r in records:
        row_y.setdefault(r["row_id"], []).append((r["box"][1] + r["box"][3]) / 2)
    row_pos = {rid: sum(v) / len(v) for rid, v in row_y.items()}
    records.sort(key=lambda r: (row_pos[r["row_id"]], r["box"][0]))

    words, boxes, labels, group_ids = [], [], [], []
    started: set[int] = set()
    for r in records:
        field = r["field"]
        if field is None:
            label = OUTSIDE
        else:
            label = f"{'I' if r['line_idx'] in started else 'B'}-{field}"
            started.add(r["line_idx"])
        words.append(r["text"])
        boxes.append(r["box"])
        labels.append(label)
        group_ids.append(r["group_id"] if field in LINE_ITEM_FIELDS else NO_GROUP)
    return {"words": words, "boxes": boxes, "labels": labels, "group_ids": group_ids}


def load_cord_split(split: str, root: str | Path = DEFAULT_ROOT) -> Iterator[CordSample]:
    """Yield every sample of one official CORD split (train 800 / validation 100 / test 100)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    from datasets import load_from_disk  # imported lazily so tests need no heavy deps

    ds = load_from_disk(str(root))[split]
    for i, example in enumerate(ds):
        width, height = example["image"].size
        parsed = parse_ground_truth(json.loads(example["ground_truth"]), width, height)
        yield CordSample(
            sample_id=f"cord-{split}-{i:04d}", split=split, index=i, width=width, height=height, **parsed
        )