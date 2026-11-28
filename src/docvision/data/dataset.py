"""Map-style dataset and collate helpers on top of the CORD loader.

``CordDataset`` is a plain map-style dataset (``__len__`` / ``__getitem__``), so it works with
``torch.utils.data.DataLoader`` without importing torch here.

Each item is a dict::

    {"id": str, "words": list[str], "boxes": list[[x0, y0, x1, y1]] (0-1000),
     "labels": list[int] (word-level label ids), "group_ids": list[int],
     "image": PIL.Image or None}

Label ids come from ``schema.LABEL2ID``. Subword alignment (label on the first subword,
``IGNORE_INDEX`` on the rest) is done by the Hugging Face processor in ``make_layoutlmv3_collate``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from docvision.data.cord import DEFAULT_ROOT, CordSample, load_cord_split
from docvision.data.schema import LABEL2ID


class CordDataset:
    def __init__(
        self,
        samples: Sequence[CordSample],
        image_loader: Callable[[int], Any] | None = None,
    ) -> None:
        """``image_loader(index)`` returns a PIL image for ``sample.index``; None disables images."""
        self.samples = list(samples)
        self.image_loader = image_loader

    @classmethod
    def from_split(
        cls, split: str, root: str | Path = DEFAULT_ROOT, load_images: bool = True
    ) -> CordDataset:
        samples = list(load_cord_split(split, root))
        loader = None
        if load_images:
            # lazy import: only needed when images are requested
            from datasets import load_from_disk

            hf_split = load_from_disk(str(root))[split]

            def loader(index: int) -> Any:
                return hf_split[index]["image"].convert("RGB")

        return cls(samples, loader)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        return {
            "id": s.sample_id,
            "words": s.words,
            "boxes": s.boxes,
            "labels": [LABEL2ID[label] for label in s.labels],
            "group_ids": s.group_ids,
            "image": self.image_loader(s.index) if self.image_loader else None,
        }


def make_layoutlmv3_collate(processor: Any, max_length: int = 512) -> Callable[[list[dict]], Any]:
    """Collate function for LayoutLMv3 (``LayoutLMv3Processor`` built with ``apply_ocr=False``).

    The processor tokenises the words, aligns word labels to the first subword (others become
    -100), resizes the images and pads the batch. Returns a ``BatchEncoding`` of tensors with
    the keys ``input_ids``, ``attention_mask``, ``bbox``, ``pixel_values`` and ``labels``.
    """

    def collate(batch: list[dict]) -> Any:
        if any(item["image"] is None for item in batch):
            raise ValueError("LayoutLMv3 needs images: build the dataset with load_images=True")
        return processor(
            [item["image"] for item in batch],
            [item["words"] for item in batch],
            boxes=[item["boxes"] for item in batch],
            word_labels=[item["labels"] for item in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate