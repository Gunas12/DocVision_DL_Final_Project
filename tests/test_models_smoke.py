"""Offline model checks; pretrained weights are explicitly opt-in."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from docvision.models.layoutlmv3_tagger import (
    LayoutLMv3Tagger,
    load_config,
    load_tagger,
    validate_boxes,
)


@pytest.fixture
def make_tagger():
    def make(**ablation):
        torch.manual_seed(13)
        torch.set_num_threads(2)
        vocab = {word: i for i, word in enumerate(
            ["<s>", "<pad>", "</s>", "<unk>", "Ġ", "t", "e", "a", "p", "o", "Ġt", "Ġte", "Ġtea", "po", "pot"]
        )}
        backend = Tokenizer(BPE(vocab, [("Ġ", "t"), ("Ġt", "e"), ("Ġte", "a"), ("p", "o"), ("po", "t")]))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=True)
        processor = LayoutLMv3Processor(LayoutLMv3ImageProcessor(apply_ocr=False),
                                       LayoutLMv3TokenizerFast(tokenizer_object=backend))
        config = {"model": {"name": "layoutlmv3", "pretrained": "offline"}, "max_length": 8,
                  "image_size": 16, "ablation": ablation, "gradient_checkpointing": True}
        model = LayoutLMv3ForTokenClassification(LayoutLMv3Config(
            vocab_size=len(vocab), hidden_size=32, coordinate_size=4, shape_size=8,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
            max_position_embeddings=16, input_size=16, patch_size=8, num_labels=3,
            visual_embed=ablation.get("use_image", True),
        ))
        return LayoutLMv3Tagger(config, ["O", "B-item", "I-item"], model=model, processor=processor)
    return make


@pytest.mark.parametrize("ablation", [{}, {"use_layout": False}, {"use_image": False}, {"freeze_lower_layers": 1}])
def test_forward_alignment_and_ablations(make_tagger, ablation):
    tagger = make_tagger(**ablation)
    inputs = tagger.encode(["teapot", "tea"], [[-5, 0, 1200, 20], [0, 0, 3, 4]], word_labels=["B-item", "O"])
    assert inputs["labels"].tolist() == [[-100, 1, -100, 0, -100, -100, -100, -100]]
    assert ("pixel_values" in inputs) == tagger.use_image
    if not tagger.use_layout:
        assert inputs["bbox"].count_nonzero() == 0
    rng = torch.get_rng_state()
    tagger.config["gradient_checkpointing"] = False
    reference = tagger(**inputs)
    reference.loss.backward()
    gradient = tagger.model.classifier.weight.grad.clone()
    tagger.zero_grad(set_to_none=True)
    torch.set_rng_state(rng)
    tagger.config["gradient_checkpointing"] = True
    output = tagger(**inputs)
    assert output.logits.shape == (1, 8, 3) and torch.isfinite(output.loss)
    output.loss.backward()
    torch.testing.assert_close(output.loss, reference.loss)
    torch.testing.assert_close(tagger.model.classifier.weight.grad, gradient)
    assert tagger.model.classifier.weight.grad is not None
    if ablation.get("freeze_lower_layers"):
        for name, parameter in tagger.model.layoutlmv3.named_parameters():
            if not name.startswith("encoder.") or name.startswith("encoder.layer.0."):
                assert not parameter.requires_grad and parameter.grad is None
        assert any(p.requires_grad for p in tagger.model.layoutlmv3.encoder.layer[1].parameters())


@pytest.mark.parametrize("use_image", [False, True])
def test_predict_batch_hidden_truncation_and_reload(make_tagger, tmp_path, use_image):
    tagger = make_tagger(use_image=use_image)
    docs = [{"words": words, "boxes": [[0, 0, 10, 10]] * len(words)}
            for words in (["teapot", "tea"], ["teapot"] * 9, [], ["tea"] * 5 + ["teapot"])]
    results = tagger.predict_batch(docs, batch_size=2, return_hidden=True)
    for doc, result in zip(docs, results):
        assert len(result["labels"]) == len(result["probs"]) == len(doc["words"])
        assert result["label_probs"].shape == (len(doc["words"]), 3)
        assert result["hidden"].shape == (len(doc["words"]), 32)
        assert result["label_probs"].dtype == result["hidden"].dtype == np.float32
    assert [r["truncated"] for r in results] == [False, True, False, True]
    assert np.all(results[1]["label_probs"][3:] == 0) and np.all(results[1]["hidden"][3:] == 0)
    encoded = tagger.encode(**docs[0])
    with torch.inference_mode():
        output = tagger(**encoded, output_hidden_states=True)
    np.testing.assert_allclose(results[0]["hidden"], output.hidden_states[-1][0, [1, 3]].numpy(), atol=1e-6)
    assert not tagger.training and not tagger.model.training
    tagger.save_pretrained(tmp_path)
    loaded = load_tagger(tmp_path)
    assert loaded.id2label == tagger.id2label
    prediction = loaded.predict(**docs[0])
    np.testing.assert_allclose(prediction["label_probs"], results[0]["label_probs"], atol=1e-6)
    assert "hidden" not in prediction
    with pytest.raises(ValueError, match="test split"):
        tagger.predict_batch([{**docs[0], "split": "test"}])
    with pytest.raises(ValueError, match="batch_size"):
        tagger.predict_batch(docs, batch_size=0)


@pytest.mark.parametrize("boxes", [[], [[0, 0, 1]], [[2, 0, 1, 3]], [[0, 4, 1, 3]], [[0, 0, 1.5, 2]]])
def test_invalid_boxes(boxes):
    with pytest.raises(ValueError):
        validate_boxes(["tea"], boxes)


def test_config_and_box_clipping(tmp_path):
    config = load_config("configs/layoutlmv3.yaml", ["training.lr=3e-5", "ablation.use_image=false"])
    assert config["training"]["lr"] == 3e-5 and config["training"]["batch_size"] == 2
    assert not config["ablation"]["use_image"] and config["ablation"]["use_layout"]
    with pytest.raises(ValueError, match="test split"):
        load_config("configs/layoutlmv3.yaml", ["data.split=test"])
    assert validate_boxes(["tea"], [[-10, -20, 1100, 1200]]) == [[0, 0, 1000, 1000]]
    parent, child = tmp_path / "parent.yaml", tmp_path / "child.yaml"
    parent.write_text("labels: [O, B-item]\ntraining: {lr: 0.01, batch_size: 4}\n", encoding="utf-8")
    child.write_text("inherit: parent.yaml\nlabels: [O]\ntraining: {lr: 0.02}\n", encoding="utf-8")
    assert load_config(child) == {"labels": ["O"], "training": {"lr": 0.02, "batch_size": 4}}
    parent.write_text("inherit: child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(child)


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to download pretrained weights")
def test_pretrained_layoutlmv3_forward():
    tagger = LayoutLMv3Tagger(load_config("configs/layoutlmv3.yaml"), ["O", "B-item"])
    inputs = tagger.encode(["tea", "milk", "tax", "total", "5"], [[0, 0, 10, 10]] * 5)
    with torch.inference_mode():
        assert tagger(**inputs).logits.shape == (1, 512, 2)
