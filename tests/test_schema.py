from docvision.data import schema as s


def test_label_maps_are_consistent():
    assert s.LABELS[0] == "O"
    assert len(s.LABELS) == 1 + 2 * len(s.FIELDS)
    assert all(s.ID2LABEL[s.LABEL2ID[l]] == l for l in s.LABELS)


def test_fields_cover_mapping():
    assert set(s.FIELDS) == set(s.CORD_TO_FIELD.values())
    assert set(s.LINE_ITEM_FIELDS) | set(s.DOCUMENT_FIELDS) == set(s.FIELDS)


def test_unknown_category_is_outside():
    assert s.cord_category_to_field("menu.nm") == "ITEM_NAME"
    assert s.cord_category_to_field("total.cashprice") is None


def test_document_to_dict_and_review_flag():
    doc = s.Document(menu=[s.LineItem(name=s.ExtractedField(text="Tea"))])
    assert doc.needs_review is False
    doc.review_flags.append(s.ReviewFlag("arithmetic", "SUBTOTAL", "sum mismatch"))
    d = doc.to_dict()
    assert d["needs_review"] is True and d["menu"][0]["name"]["text"] == "Tea"