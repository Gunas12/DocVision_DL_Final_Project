import pytest

from docvision.data.cord import CordSample
from docvision.data.dataset import CordDataset, make_layoutlmv3_collate
from docvision.data.schema import ID2LABEL


def _sample(i):
    return CordSample(
        sample_id=f"cord-train-{i:04d}", split="train", index=i, width=100, height=100,
        words=["1", "x", "Tea", "5,000"],
        boxes=[[0, 0, 10, 10], [11, 0, 20, 10], [30, 0, 60, 10], [700, 0, 800, 10]],
        labels=["B-QTY", "I-QTY", "B-ITEM_NAME", "B-PRICE"],
        group_ids=[1, 1, 1, 1],
    )


def test_items_have_label_ids_and_aligned_lengths():
    ds = CordDataset([_sample(0), _sample(1)])
    assert len(ds) == 2
    item = ds[1]
    assert item["id"] == "cord-train-0001" and item["image"] is None
    assert [ID2LABEL[x] for x in item["labels"]] == ["B-QTY", "I-QTY", "B-ITEM_NAME", "B-PRICE"]
    assert len(item["words"]) == len(item["boxes"]) == len(item["labels"]) == len(item["group_ids"])
    assert all(0 <= v <= 1000 for box in item["boxes"] for v in box)


def test_image_loader_is_called_with_sample_index():
    ds = CordDataset([_sample(7)], image_loader=lambda idx: f"img-{idx}")
    assert ds[0]["image"] == "img-7"


def test_collate_calls_processor_with_word_level_inputs():
    seen = {}

    def fake_processor(images, words, **kwargs):
        seen.update(images=images, words=words, **kwargs)
        return "batch"

    ds = CordDataset([_sample(0), _sample(1)], image_loader=lambda idx: f"img-{idx}")
    out = make_layoutlmv3_collate(fake_processor, max_length=128)([ds[0], ds[1]])
    assert out == "batch"
    assert seen["images"] == ["img-0", "img-1"]
    assert seen["max_length"] == 128 and seen["truncation"] and seen["padding"]
    assert len(seen["word_labels"]) == 2 and len(seen["boxes"][0]) == 4


def test_collate_requires_images():
    ds = CordDataset([_sample(0)])
    with pytest.raises(ValueError):
        make_layoutlmv3_collate(lambda *a, **k: None)([ds[0]])