"""LayoutLMv3 tagger; pretrained weights use CC BY-NC-SA 4.0 (non-commercial)."""

import copy
import json
import re
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)


def _merge(parent, child):
    result = copy.deepcopy(parent)
    for key, value in child.items():
        result[key] = _merge(result[key], value) if (
            isinstance(value, dict) and isinstance(result.get(key), dict)
        ) else copy.deepcopy(value)
    return result


def _reject_test_split(config):
    for key, value in config.items():
        if isinstance(value, dict):
            _reject_test_split(value)
        elif (key == "split" or key.endswith("_split")) and value == "test":
            raise ValueError("The test split is reserved for user#4")


def load_config(path, overrides=(), _parents=()):
    """Deep-merge YAML mappings using `inherit: path/relative/to/child.yaml`.

    Child values win; lists replace lists. Dotted overrides such as
    `training.lr=3e-5` use YAML values, including scientific-notation floats.
    Supply labels as a YAML list or constructor argument until schema.py exists.
    """
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("Config inheritance cycle")
    child = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(child, dict):
        raise TypeError("Config must be a mapping")
    parent = child.pop("inherit", None)
    config = _merge(load_config(path.parent / parent, _parents=(*_parents, path))
                    if parent else {}, child)
    for override in overrides:
        key, value = override.split("=", 1)
        target = config
        parts = key.split(".")
        if not all(parts):
            raise ValueError("Override keys must be nonempty")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise TypeError("Cannot override a child of a scalar")
        parsed = yaml.safe_load(value)
        # PyYAML treats exponent-only notation (3e-5) as a string.
        if isinstance(parsed, str) and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+", value):
            parsed = float(value)
        target[parts[-1]] = parsed
    _reject_test_split(config)
    return config


def validate_boxes(words, boxes):
    """Reject malformed/reversed integer boxes; clip ordered coordinates."""
    if len(words) != len(boxes):
        raise ValueError("One box is required per word")
    result = []
    for box in boxes:
        if len(box) != 4 or any(not isinstance(v, Integral) or isinstance(v, bool) for v in box):
            raise ValueError("Boxes must contain four integers")
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError("Box coordinates must be ordered")
        result.append([max(0, min(1000, int(v))) for v in box])
    return result


class LayoutLMv3Tagger(torch.nn.Module):
    """Word tagger with injectable model/processor for offline smoke tests.

    Missing images use a white RGB canvas. Unrepresented words get "O" (or the
    first label if absent) and zero probabilities/vectors; inspect `truncated`.
    freeze_lower_layers=0 leaves all parameters trainable.
    """

    def __init__(self, config, labels=None, *, model=None, processor=None, device="cpu"):
        super().__init__()
        _reject_test_split(config)
        self.config = copy.deepcopy(config)
        labels = config.get("labels") if labels is None else labels
        if not isinstance(labels, list) or not labels or any(
            not isinstance(label, str) or not label for label in labels
        ) or len(set(labels)) != len(labels):
            raise ValueError("Provide a nonempty list of unique label strings")
        self.id2label = dict(enumerate(labels))
        self.label2id = {label: i for i, label in self.id2label.items()}
        self.config["labels"] = labels.copy()
        ablation = config.get("ablation", {})
        self.use_layout = ablation.get("use_layout", True)
        self.use_image = ablation.get("use_image", True)
        source = config["model"]["pretrained"]
        if model is None:
            model_config = LayoutLMv3Config.from_pretrained(
                source, id2label=self.id2label, label2id=self.label2id,
                visual_embed=self.use_image, input_size=config.get("image_size", 224),
            )
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                source, config=model_config, ignore_mismatched_sizes=True,
            )
        if model.config.visual_embed != self.use_image or model.config.num_labels != len(labels):
            raise ValueError("Injected model must match image ablation and label count")
        self.model = model
        model.config.id2label, model.config.label2id = self.id2label, self.label2id
        self.processor = processor or LayoutLMv3Processor.from_pretrained(source, apply_ocr=False)
        self.processor.image_processor.apply_ocr = False
        size = config.get("image_size", 224)
        self.processor.image_processor.size = {"height": size, "width": size}
        count = ablation.get("freeze_lower_layers", 0)
        if not isinstance(count, int) or not 0 <= count <= model.config.num_hidden_layers:
            raise ValueError("freeze_lower_layers must be within the encoder depth")
        if count:
            for name, parameter in model.layoutlmv3.named_parameters():
                if not name.startswith("encoder.") or any(
                    name.startswith(f"encoder.layer.{i}.") for i in range(count)
                ):
                    parameter.requires_grad_(False)
        if config.get("gradient_checkpointing", False):
            # Transformers 5.24.1 omits this flag despite checkpoint-capable layers.
            self.model.supports_gradient_checkpointing = True
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.to(device)

    def encode(self, words, boxes, image=None, word_labels=None, *, padding="max_length"):
        """Align a document; fixed padding remains the training default."""
        boxes = validate_boxes(words, boxes)
        if word_labels is not None and len(word_labels) != len(words):
            raise ValueError("One label is required per word")
        kwargs = {"boxes": boxes, "truncation": True, "padding": padding,
                  "max_length": self.config.get("max_length", 512), "return_tensors": "pt"}
        if self.use_image:
            image = image if image is not None else Image.new(
                "RGB", (self.config.get("image_size", 224),) * 2, "white"
            )
            encoded = self.processor(images=image.convert("RGB"), text=words, **kwargs)
        else:
            encoded = self.processor.tokenizer(text=words, **kwargs)
        if not self.use_layout:
            encoded["bbox"].zero_()
        if word_labels is not None:
            label_ids = [self.label2id[label] for label in word_labels]
            aligned, seen = [], set()
            for word in encoded.word_ids():
                aligned.append(label_ids[word] if word is not None and word not in seen else -100)
                seen.add(word)
            encoded["labels"] = torch.tensor([aligned], dtype=torch.long)
        return encoded

    def forward(self, **inputs):
        if not self.use_layout and "bbox" in inputs:
            inputs["bbox"] = torch.zeros_like(inputs["bbox"])
        if not self.use_image:
            inputs.pop("pixel_values", None)
        return self.model(**inputs)

    def predict(self, words, boxes, image=None, return_hidden=False):
        return self.predict_batch([{"words": words, "boxes": boxes, "image": image}],
                                  batch_size=1, return_hidden=return_hidden)[0]

    @torch.inference_mode()
    def predict_batch(self, docs, batch_size=8, return_hidden=False):
        """Predict with dynamic padding, then restore every module's prior mode."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._predict_batch(docs, batch_size, return_hidden)
        finally:
            for module, training in modes:
                module.training = training

    def _predict_batch(self, docs, batch_size, return_hidden):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        results = []
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            if any(doc.get("split") == "test" for doc in batch):
                raise ValueError("The test split is reserved for user#4")
            encodings = [self.encode(d["words"], d["boxes"], d.get("image"), padding=False) for d in batch]
            inputs = self.processor.tokenizer.pad(
                [{k: v[0] for k, v in e.items() if k != "pixel_values"} for e in encodings],
                padding=True, return_tensors="pt",
            )
            if self.use_image:
                inputs["pixel_values"] = torch.cat([e["pixel_values"] for e in encodings])
            inputs = inputs.to(next(self.parameters()).device)
            output = self(**inputs, output_hidden_states=return_hidden)
            probabilities = output.logits.softmax(-1).float().cpu().numpy()
            hidden = output.hidden_states[-1][:, :inputs["input_ids"].shape[1]] if return_hidden else None
            for i, (doc, encoding) in enumerate(zip(batch, encodings)):
                first = {}
                for token, word in enumerate(encoding.word_ids()):
                    if word is not None:
                        first.setdefault(word, token)
                word_ids, tokens = list(first), list(first.values())
                probs = np.zeros((len(doc["words"]), len(self.id2label)), dtype=np.float32)
                probs[word_ids] = probabilities[i, tokens]
                labels = [self.id2label[self.label2id.get("O", 0)]] * len(probs)
                for word in first:
                    labels[word] = self.id2label[int(probs[word].argmax())]
                result = {"labels": labels,
                          "probs": probs.max(-1).tolist(), "label_probs": probs,
                          "truncated": bool(encoding.encodings[0].overflowing) or len(first) < len(probs)}
                if return_hidden:
                    vectors = np.zeros((len(probs), self.model.config.hidden_size), dtype=np.float32)
                    vectors[word_ids] = hidden[i, tokens].float().cpu().numpy()
                    result["hidden"] = vectors
                results.append(result)
        return results

    def save_pretrained(self, directory):
        """Save model assets; the future trainer adds trainer_state.pt."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        (directory / "labels.json").write_text(json.dumps(list(self.id2label.values())), encoding="utf-8")
        (directory / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")


def load_tagger(checkpoint_dir, device="cpu"):
    """Load model assets on CPU by default; other tagger families register later."""
    directory = Path(checkpoint_dir).resolve()
    config = load_config(directory / "config.yaml")
    if config["model"]["name"] != "layoutlmv3":
        raise ValueError("Unsupported tagger family")
    config["model"]["pretrained"] = str(directory)
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    return LayoutLMv3Tagger(config, labels, device=device).eval()
