"""Label set and output JSON schema for DocVision.

This module is the contract between the data loaders, the models, the
structure/validation code and the UI. Change it only through a pull request
that all consumers review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Core fields the product extracts: CORD category -> DocVision field name.
# Every CORD category not listed here is mapped to the "O" (outside) label.
CORD_TO_FIELD: dict[str, str] = {
    "menu.nm": "ITEM_NAME",
    "menu.cnt": "QTY",
    "menu.unitprice": "UNIT_PRICE",
    "menu.price": "PRICE",
    "sub_total.subtotal_price": "SUBTOTAL",
    "sub_total.tax_price": "TAX",
    "sub_total.service_price": "SERVICE",
    "sub_total.discount_price": "DISCOUNT",
    "total.total_price": "TOTAL",
}

FIELDS: list[str] = sorted(set(CORD_TO_FIELD.values()), key=list(CORD_TO_FIELD.values()).index)

# Fields that live inside one line item (used by grouping / relation head).
LINE_ITEM_FIELDS: tuple[str, ...] = ("ITEM_NAME", "QTY", "UNIT_PRICE", "PRICE")
# Fields that appear once per document.
DOCUMENT_FIELDS: tuple[str, ...] = ("SUBTOTAL", "TAX", "SERVICE", "DISCOUNT", "TOTAL")

OUTSIDE = "O"
LABELS: list[str] = [OUTSIDE] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100  # used for non-first subword tokens and padding


def cord_category_to_field(category: str) -> str | None:
    """Return the DocVision field for a CORD category, or None if it is not tracked."""
    return CORD_TO_FIELD.get(category)


# ---------------------------------------------------------------------------
# Output document schema (what the pipeline returns and the API serialises)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedField:
    """One extracted value with its provenance and confidence."""

    text: str
    value: float | None = None  # parsed number for amount fields, else None
    confidence: float = 0.0
    box: list[int] | None = None  # [x0, y0, x1, y1] normalised to 0-1000
    token_indices: list[int] = field(default_factory=list)


@dataclass
class LineItem:
    name: ExtractedField | None = None
    qty: ExtractedField | None = None
    unit_price: ExtractedField | None = None
    price: ExtractedField | None = None


@dataclass
class ReviewFlag:
    """A reason why a field or the whole document needs human review."""

    kind: str  # "low_confidence" | "arithmetic" | "format" | "missing_field"
    field: str  # e.g. "TOTAL" or "menu[2].price"
    message: str


@dataclass
class Document:
    menu: list[LineItem] = field(default_factory=list)
    subtotal: ExtractedField | None = None
    tax: ExtractedField | None = None
    service: ExtractedField | None = None
    discount: ExtractedField | None = None
    total: ExtractedField | None = None
    review_flags: list[ReviewFlag] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_flags)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["needs_review"] = self.needs_review
        return out