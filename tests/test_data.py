"""Split leakage tests. They read the committed manifest, so they run in CI without the data."""

import json
from pathlib import Path

import pytest

from docvision.data.cord import SPLITS, load_cord_split, make_manifest

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "splits" / "cord_split_v1.json"
ROOT = Path("data/raw/cord-v2")


def test_make_manifest_structure():
    raw = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    m = make_manifest(raw)
    assert m["sizes"] == {"train": 2, "validation": 1, "test": 1}
    assert m["splits"]["train"] == ["cord-train-0000", "cord-train-0001"]
    assert len(set(m["fingerprints"].values())) == 4


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.exists(), "run scripts/make_cord_split.py and commit data/splits/cord_split_v1.json"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_split_sizes_are_official(manifest):
    assert manifest["sizes"] == {"train": 800, "validation": 100, "test": 100}


def test_splits_do_not_overlap(manifest):
    sets = {s: set(manifest["splits"][s]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            assert sets[a].isdisjoint(sets[b]), f"{a} and {b} share sample ids"


def test_no_duplicate_receipts_across_splits(manifest):
    fp = manifest["fingerprints"]
    by_split = {s: {fp[i] for i in manifest["splits"][s]} for s in SPLITS}
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])


@pytest.mark.skipif(not ROOT.exists(), reason="CORD not downloaded")
def test_manifest_matches_data_on_disk(manifest):
    from datasets import load_from_disk

    from docvision.data.cord import fingerprint

    ds = load_from_disk(str(ROOT))
    for split in SPLITS:
        raw = ds[split]["ground_truth"]
        ids = manifest["splits"][split]
        assert len(raw) == len(ids)
        for sample_id, gt in zip(ids, raw):
            assert manifest["fingerprints"][sample_id] == fingerprint(gt)


def test_loader_rejects_unknown_split():
    with pytest.raises(ValueError):
        list(load_cord_split("dev"))