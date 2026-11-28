import random

from docvision.data.augment import (
    LEVELS,
    NoiseConfig,
    add_ocr_noise,
    build_distractor_pool,
    read_samples_jsonl,
    repair_bio,
    write_samples_jsonl,
)
from docvision.data.cord import CordSample
from docvision.data.schema import LABELS


def _sample(sid="cord-val-0000"):
    words = ["Nasi", "Goreng", "Subtotal", "52,000", "TOTAL", "68,200", "Terima", "kasih"]
    labels = ["B-ITEM_NAME", "I-ITEM_NAME", "O", "B-SUBTOTAL", "O", "B-TOTAL", "O", "O"]
    boxes = [[100, 100 + 40 * i, 200, 120 + 40 * i] for i in range(len(words))]
    return CordSample(sid, "validation", 0, 864, 1296, words, boxes, labels, [3, 3, -1, -1, -1, -1, -1, -1])


def _valid(s):
    assert len(s.words) == len(s.boxes) == len(s.labels) == len(s.group_ids)
    assert all(label in LABELS for label in s.labels)
    assert s.labels == repair_bio(s.labels)
    assert all(0 <= v <= 1000 and b[0] <= b[2] and b[1] <= b[3] for b in s.boxes for v in b)
    assert all(w for w in s.words)


def test_zero_noise_is_identity_except_id():
    s = _sample()
    out = add_ocr_noise(s, NoiseConfig("none"), 13)
    assert (out.words, out.boxes, out.labels, out.group_ids) == (s.words, s.boxes, s.labels, s.group_ids)
    assert out.sample_id.endswith("-noisy-none")


def test_noise_is_deterministic_and_seed_dependent():
    s, pool = _sample(), [("Cash", [10, 500, 90, 520]), ("Jl.", [10, 50, 40, 70])]
    a = add_ocr_noise(s, LEVELS["high"], 13, pool)
    b = add_ocr_noise(s, LEVELS["high"], 13, pool)
    assert a == b
    assert any(add_ocr_noise(s, LEVELS["high"], seed, pool) != a for seed in range(14, 30))


def test_noisy_samples_stay_valid_across_many_seeds():
    pool = [("Cash", [10, 500, 90, 520]), ("Jl.", [10, 50, 40, 70])]
    for seed in range(200):
        for cfg in LEVELS.values():
            _valid(add_ocr_noise(_sample(), cfg, seed, pool))


def test_distractors_are_outside_and_never_split_an_entity():
    cfg = NoiseConfig("d", distractor_rate=3.0)
    pool = [("Cash", [10, 100 + 20, 90, 100 + 40])]  # lands right between the item-name words
    for seed in range(50):
        out = add_ocr_noise(_sample(), cfg, seed, pool)
        _valid(out)
        assert out.labels.count("B-ITEM_NAME") == 1 and out.labels.count("I-ITEM_NAME") == 1
        i = out.labels.index("B-ITEM_NAME")
        assert out.labels[i + 1] == "I-ITEM_NAME"
        assert len(out.words) == 8 + 24


def test_split_keeps_entity_labels():
    cfg = NoiseConfig("s", split_rate=1.0)
    out = add_ocr_noise(_sample(), cfg, 1)
    assert out.labels[0] == "B-ITEM_NAME" and out.labels[1] == "I-ITEM_NAME"
    assert out.labels.count("B-TOTAL") == 1 and out.labels.count("I-TOTAL") == 1
    _valid(out)


def test_repair_bio_and_pool_and_jsonl_roundtrip(tmp_path):
    assert repair_bio(["O", "I-TOTAL", "I-TOTAL", "B-TAX", "I-TOTAL"]) == ["O", "B-TOTAL", "I-TOTAL", "B-TAX", "B-TOTAL"]
    assert [w for w, _ in build_distractor_pool([_sample()])] == ["Subtotal", "TOTAL", "Terima", "kasih"]
    noisy = [add_ocr_noise(_sample(), LEVELS["medium"], 13)]
    write_samples_jsonl(noisy, tmp_path / "x" / "n.jsonl")
    assert read_samples_jsonl(tmp_path / "x" / "n.jsonl") == noisy
    assert random.Random("a-b").random() == random.Random("a-b").random()