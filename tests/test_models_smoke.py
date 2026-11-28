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
    def make(*, labels=None, gradient_checkpointing=True, **ablation):
        labels = ["O", "B-item", "I-item"] if labels is None else labels
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
                  "image_size": 16, "ablation": ablation, "gradient_checkpointing": gradient_checkpointing}
        model = LayoutLMv3ForTokenClassification(LayoutLMv3Config(
            vocab_size=len(vocab), hidden_size=32, coordinate_size=4, shape_size=8,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
            max_position_embeddings=16, input_size=16, patch_size=8, num_labels=len(labels),
            visual_embed=ablation.get("use_image", True),
        ))
        return LayoutLMv3Tagger(config, labels, model=model, processor=processor)
    return make


@pytest.mark.parametrize("ablation", [{}, {"use_layout": False}, {"use_image": False}, {"freeze_lower_layers": 1}])
def test_forward_alignment_and_ablations(make_tagger, ablation):
    reference_tagger = make_tagger(gradient_checkpointing=False, **ablation)
    tagger = make_tagger(**ablation)
    encoder = tagger.model.layoutlmv3.encoder
    assert encoder.gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in encoder.layer)
    assert all(layer._gradient_checkpointing_func.keywords["use_reentrant"] is False for layer in encoder.layer)
    assert not reference_tagger.model.layoutlmv3.encoder.gradient_checkpointing
    inputs = tagger.encode(["teapot", "tea"], [[-5, 0, 1200, 20], [0, 0, 3, 4]], word_labels=["B-item", "O"])
    assert inputs["labels"].tolist() == [[-100, 1, -100, 0, -100, -100, -100, -100]]
    assert ("pixel_values" in inputs) == tagger.use_image
    if not tagger.use_layout:
        assert inputs["bbox"].count_nonzero() == 0
    rng = torch.get_rng_state()
    reference = reference_tagger(**inputs)
    reference.loss.backward()
    torch.set_rng_state(rng)
    output = tagger(**inputs)
    assert output.logits.shape == (1, 8, 3) and torch.isfinite(output.loss)
    output.loss.backward()
    torch.testing.assert_close(output.loss, reference.loss)
    reference_parameters = dict(reference_tagger.named_parameters())
    for name, parameter in tagger.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)
    assert tagger.model.classifier.weight.grad is not None
    if ablation.get("freeze_lower_layers"):
        for name, parameter in tagger.model.layoutlmv3.named_parameters():
            if not name.startswith("encoder.") or name.startswith("encoder.layer.0."):
                assert not parameter.requires_grad and parameter.grad is None
        assert any(p.requires_grad for p in tagger.model.layoutlmv3.encoder.layer[1].parameters())


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("api", ["predict", "predict_batch"])
@pytest.mark.parametrize("fail", [False, True])
def test_prediction_restores_modes(make_tagger, training, api, fail):
    tagger = make_tagger().train(training)
    tagger.model.dropout.eval()  # Preserve intentional mixed submodule modes too.
    modes = {name: module.training for name, module in tagger.named_modules()}
    observed = []

    def inspect_mode(module, args):
        assert torch.is_inference_mode_enabled()
        assert all(not child.training for child in tagger.modules())
        observed.append(True)
        if fail:
            raise RuntimeError("inference failed")

    doc = {"words": ["tea"], "boxes": [[0, 0, 10, 10]]}
    handle = tagger.model.register_forward_pre_hook(inspect_mode)
    try:
        kwargs = doc if api == "predict" else {"docs": [doc]}
        if fail:
            with pytest.raises(RuntimeError, match="inference failed"):
                getattr(tagger, api)(**kwargs)
        else:
            getattr(tagger, api)(**kwargs)
        assert observed == [True]
    finally:
        handle.remove()
    assert tagger.predict_batch([]) == []
    assert modes == {name: module.training for name, module in tagger.named_modules()}


@pytest.mark.parametrize("labels, fallback", [(["B-item", "O", "I-item"], "O"), (["B-item", "I-item"], "B-item")])
@pytest.mark.parametrize("use_image", [False, True])
def test_truncated_words_use_outside_label(make_tagger, labels, fallback, use_image):
    tagger = make_tagger(labels=labels, use_image=use_image)
    result = tagger.predict(["teapot"] * 9, [[0, 0, 10, 10]] * 9, return_hidden=True)
    assert result["truncated"] and result["labels"][3:] == [fallback] * 6
    assert all(prob > 0 for prob in result["probs"][:3])
    assert result["probs"][3:] == [0.0] * 6
    assert np.all(result["label_probs"][3:] == 0) and np.all(result["hidden"][3:] == 0)


@pytest.mark.parametrize("use_image", [False, True])
def test_inference_pads_to_batch_longest(make_tagger, use_image):
    tagger = make_tagger(use_image=use_image).eval()
    docs = [{"words": words, "boxes": [[0, 0, 10, 10]] * len(words)}
            for words in (["tea"], ["teapot", "tea"])]
    lengths = []
    handle = tagger.model.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs["input_ids"].shape[1]), with_kwargs=True,
    )
    try:
        single = tagger.predict(**docs[0])
        batched = tagger.predict_batch(docs, batch_size=2)
    finally:
        handle.remove()
    assert lengths == [3, 5] and max(lengths) < tagger.config["max_length"]
    for doc, result in zip(docs, batched):
        padded = tagger.encode(**doc)
        assert padded["input_ids"].shape[1] == tagger.config["max_length"]
        with torch.inference_mode():
            probabilities = tagger(**padded).logits.softmax(-1)[0].numpy()
        first = [i for i, word in enumerate(padded.word_ids())
                 if word is not None and (i == 0 or padded.word_ids()[i - 1] != word)]
        np.testing.assert_allclose(result["label_probs"], probabilities[first], atol=1e-4)
    np.testing.assert_allclose(single["label_probs"], batched[0]["label_probs"], atol=1e-4)


@pytest.mark.parametrize("use_image", [False, True])
def test_predict_batch_hidden_truncation_and_reload(make_tagger, tmp_path, use_image):
    tagger = make_tagger(use_image=use_image).eval()
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


@pytest.mark.parametrize("boxes", [
    [], [None], [4], [[0, 0, 1]], [[0, 0, 1, 2, 3]],
    [[2, 0, 1, 3]], [[0, 4, 1, 3]], [[1.49, 0, 1.4, 2]], [[1001.2, 0, 1000.8, 2]],
    [[0, 0, float("nan"), 2]], [[0, -float("inf"), 1, 2]], [[0, 0, np.float32("inf"), 2]],
    [[0, 0, "1", 2]], [[0, 0, True, 2]], [[0, 0, 1 + 2j, 2]],
])
def test_invalid_boxes(boxes):
    with pytest.raises(ValueError):
        validate_boxes(["tea"], boxes)


@pytest.mark.parametrize("box, expected", [
    ([0.0, 0.0, 1.5, 2.5], [0, 0, 2, 2]),
    ([-10.6, -0.49, 1001.3, 2.51], [0, 0, 1000, 3]),
    ([np.int64(-3), np.float32(1.6), np.float64(2.5), np.int32(4)], [0, 2, 2, 4]),
])
def test_real_boxes_round_and_encode(make_tagger, box, expected):
    assert validate_boxes(["tea"], [box]) == [expected]
    encoded = make_tagger(use_image=False).encode(["tea"], [box])
    assert encoded["bbox"].dtype == torch.long
    assert encoded["bbox"][0, 1].tolist() == expected


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
