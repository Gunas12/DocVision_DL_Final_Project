from pathlib import Path

import pytest

from docvision.data.cord import SPLITS, load_cord_split, parse_ground_truth, quad_to_box
from docvision.data.schema import LABEL2ID


def _quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def _word(text, x0, y0, x1, y1, row, is_key=0):
    return {"quad": _quad(x0, y0, x1, y1), "is_key": is_key, "row_id": row, "text": text}


GT = {
    "valid_line": [
        {"words": [_word("Nasi", 100, 200, 150, 220, 1), _word("Goreng", 160, 200, 230, 220, 1)],
         "category": "menu.nm", "group_id": 3},
        {"words": [_word("15,000", 700, 200, 800, 220, 1)], "category": "menu.price", "group_id": 3},
        {"words": [_word("TOTAL", 100, 400, 180, 420, 2, is_key=1)], "category": "total.total_price", "group_id": 9},
        {"words": [_word("30,000", 700, 400, 800, 420, 2)], "category": "total.total_price", "group_id": 9},
        {"words": [_word("cash", 100, 600, 180, 620, 3)], "category": "total.cashprice", "group_id": 10},
    ]
}


def test_quad_to_box_is_normalised_and_clipped():
    assert quad_to_box(_quad(0, 0, 864, 1296), 864, 1296) == [0, 0, 1000, 1000]
    assert quad_to_box(_quad(432, 648, 900, 700), 864, 1296)[2] == 1000


def test_parse_ground_truth_labels_and_groups():
    out = parse_ground_truth(GT, 1000, 1000)
    assert out["words"] == ["Nasi", "Goreng", "15,000", "TOTAL", "30,000", "cash"]
    assert out["labels"] == ["B-ITEM_NAME", "I-ITEM_NAME", "B-PRICE", "O", "B-TOTAL", "O"]
    assert out["group_ids"] == [3, 3, 3, -1, -1, -1]
    assert all(label in LABEL2ID for label in out["labels"])
    assert len(out["boxes"]) == len(out["words"])


def test_reading_order_sorts_rows_then_x():
    out = parse_ground_truth(GT, 1000, 1000)
    assert out["boxes"][0][0] < out["boxes"][1][0] < out["boxes"][2][0]


ROOT = Path("data/raw/cord-v2")


@pytest.mark.skipif(not ROOT.exists(), reason="CORD not downloaded")
def test_real_split_sizes_and_label_validity():
    expected = {"train": 800, "validation": 100, "test": 100}
    for split in SPLITS:
        samples = list(load_cord_split(split))
        assert len(samples) == expected[split]
        for s in samples[:20]:
            assert len(s.words) == len(s.boxes) == len(s.labels) == len(s.group_ids)
            assert all(label in LABEL2ID for label in s.labels)