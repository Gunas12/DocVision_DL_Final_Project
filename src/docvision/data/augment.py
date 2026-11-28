"""OCR-noise simulation and bounding-box jitter for CORD samples.

CORD ground truth is "clean OCR": only annotated words, perfect text, tight boxes. A real OCR engine
makes character errors, splits and merges words, misses words and also returns unrelated text
(shop name, address). ``add_ocr_noise`` simulates this on a ``CordSample`` while keeping the word-level
BIO labels consistent, so noisy samples can be scored with the same metrics.

Noise is applied to *evaluation* data (``scripts/make_noisy_eval.py``) so that "OCR-noisy" scores measure
robustness of a model trained on clean data. Everything is seeded and order-independent: the random
generator of a sample depends only on (seed, level, sample id).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

from docvision.data.cord import CordSample
from docvision.data.schema import OUTSIDE

CONFUSIONS = {
    "0": "O", "O": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5",
    "8": "B", "B": "8", "2": "Z", ",": ".", ".": ",",
}  # fmt: skip
SAME_LINE_TOLERANCE = 15  # max vertical-centre distance (0-1000 units) for merging two words


@dataclass(frozen=True)
class NoiseConfig:
    name: str
    char_error_rate: float = 0.0  # per character
    split_rate: float = 0.0  # per word
    merge_rate: float = 0.0  # per word
    drop_rate: float = 0.0  # per word
    jitter: float = 0.0  # std of box coordinate noise, in 0-1000 units
    distractor_rate: float = 0.0  # extra O words, as a fraction of the word count


LEVELS: dict[str, NoiseConfig] = {
    "low": NoiseConfig("low", 0.01, 0.02, 0.02, 0.01, 2.0, 0.2),
    "medium": NoiseConfig("medium", 0.03, 0.05, 0.05, 0.03, 5.0, 0.5),
    "high": NoiseConfig("high", 0.06, 0.10, 0.10, 0.06, 10.0, 1.0),
}
TRAIN_JITTER = NoiseConfig("train_jitter", jitter=3.0)  # optional light training augmentation


def field_of(label: str) -> str | None:
    return None if label == OUTSIDE else label[2:]


def repair_bio(labels: list[str]) -> list[str]:
    """Turn any ``I-X`` that does not continue an ``X`` span into ``B-X``."""
    out, prev = [], OUTSIDE
    for label in labels:
        if label.startswith("I-") and field_of(prev) != field_of(label):
            label = "B-" + label[2:]
        out.append(label)
        prev = label
    return out


def _corrupt_text(text: str, rate: float, rng: random.Random) -> str:
    chars = []
    for ch in text:
        if rate and rng.random() < rate:
            if ch in CONFUSIONS:
                chars.append(CONFUSIONS[ch])
            # other characters are dropped
        else:
            chars.append(ch)
    return "".join(chars) or text


def _jitter_box(box: list[int], std: float, rng: random.Random) -> list[int]:
    if not std:
        return list(box)
    x0, y0, x1, y1 = (min(1000, max(0, round(v + rng.gauss(0, std)))) for v in box)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _split_word(text: str, box: list[int], label: str, rng: random.Random):
    cut = rng.randint(1, len(text) - 1)
    mid = box[0] + round((box[2] - box[0]) * cut / len(text))
    first, second = text[:cut], text[cut:]
    field = field_of(label)
    second_label = OUTSIDE if field is None else f"I-{field}"
    return (first, [box[0], box[1], mid, box[3]], label), (second, [mid, box[1], box[2], box[3]], second_label)


def _can_merge(a_label: str, b_label: str, a_box: list[int], b_box: list[int]) -> bool:
    same_line = abs((a_box[1] + a_box[3]) / 2 - (b_box[1] + b_box[3]) / 2) <= SAME_LINE_TOLERANCE
    if not same_line:
        return False
    if a_label == OUTSIDE and b_label == OUTSIDE:
        return True
    field = field_of(a_label)
    return field is not None and b_label == f"I-{field}"


def build_distractor_pool(samples: list[CordSample]) -> list[tuple[str, list[int]]]:
    """Collect (word, box) pairs labelled ``O`` (for example "Cash", "TOTAL") from other receipts."""
    return [
        (w, b)
        for s in samples
        for w, b, label in zip(s.words, s.boxes, s.labels)
        if label == OUTSIDE
    ]


def _insert_distractors(items: list[dict], pool: list[tuple[str, list[int]]], n: int, rng: random.Random) -> None:
    for _ in range(n):
        word, box = rng.choice(pool)
        y = (box[1] + box[3]) / 2
        idx = next((i for i, it in enumerate(items) if (it["box"][1] + it["box"][3]) / 2 > y), len(items))
        while idx < len(items) and items[idx]["label"].startswith("I-"):
            idx += 1  # never insert inside an entity span
        items.insert(idx, {"text": word, "box": list(box), "label": OUTSIDE, "group": -1})


def add_ocr_noise(
    sample: CordSample,
    cfg: NoiseConfig,
    seed: int,
    distractor_pool: list[tuple[str, list[int]]] | None = None,
) -> CordSample:
    """Return a noisy copy of ``sample``. Deterministic for a given (seed, cfg.name, sample id)."""
    rng = random.Random(f"{seed}-{cfg.name}-{sample.sample_id}")
    src = list(zip(sample.words, sample.boxes, sample.labels, sample.group_ids))

    items: list[dict] = []
    i = 0
    while i < len(src):
        text, box, label, group = src[i]
        if rng.random() < cfg.drop_rate:
            i += 1
            continue
        if (
            i + 1 < len(src)
            and rng.random() < cfg.merge_rate
            and _can_merge(label, src[i + 1][2], box, src[i + 1][1])
        ):
            nxt = src[i + 1]
            text += nxt[0]
            box = [min(box[0], nxt[1][0]), min(box[1], nxt[1][1]), max(box[2], nxt[1][2]), max(box[3], nxt[1][3])]
            i += 1
        text = _corrupt_text(text, cfg.char_error_rate, rng)
        if len(text) >= 4 and rng.random() < cfg.split_rate:
            (t1, b1, l1), (t2, b2, l2) = _split_word(text, box, label, rng)
            items.append({"text": t1, "box": b1, "label": l1, "group": group})
            items.append({"text": t2, "box": b2, "label": l2, "group": group})
        else:
            items.append({"text": text, "box": box, "label": label, "group": group})
        i += 1

    if distractor_pool and cfg.distractor_rate:
        _insert_distractors(items, distractor_pool, round(cfg.distractor_rate * len(src)), rng)

    labels = repair_bio([it["label"] for it in items])
    return replace(
        sample,
        sample_id=f"{sample.sample_id}-noisy-{cfg.name}",
        words=[it["text"] for it in items],
        boxes=[_jitter_box(it["box"], cfg.jitter, rng) for it in items],
        labels=labels,
        group_ids=[it["group"] for it in items],
    )


def write_samples_jsonl(samples: list[CordSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")


def read_samples_jsonl(path: str | Path) -> list[CordSample]:
    with Path(path).open(encoding="utf-8") as f:
        return [CordSample(**json.loads(line)) for line in f if line.strip()]