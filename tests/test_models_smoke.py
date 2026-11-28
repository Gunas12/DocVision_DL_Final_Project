"""Offline model checks; pretrained weights are explicitly opt-in."""
import json
import os
import random
from unittest.mock import Mock

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
from docvision.training.trainer import Trainer, resolve_precision


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


@pytest.mark.parametrize("modern", [False, True])
def test_checkpointing_api_selection(make_tagger, monkeypatch, modern):
    tagger = make_tagger(gradient_checkpointing=False)
    calls = []
    if modern:
        monkeypatch.setattr(tagger.model, "activation_checkpointing_enable", lambda **kwargs: calls.append(kwargs))
        argument = "activation_checkpointing_kwargs"
    else:
        monkeypatch.setattr(tagger.model, "activation_checkpointing_enable", None)
        monkeypatch.setattr(tagger.model, "gradient_checkpointing_enable", lambda **kwargs: calls.append(kwargs))
        argument = "gradient_checkpointing_kwargs"
    config = {**tagger.config, "gradient_checkpointing": True}
    LayoutLMv3Tagger(config, model=tagger.model, processor=tagger.processor)
    assert calls == [{argument: {"use_reentrant": False}}]


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


@pytest.fixture
def training_case(make_tagger, tmp_path):
    records = [{"words": words, "boxes": [[0, 0, 10, 10]] * len(words),
                "labels": ["B-item"] + ["I-item"] * (len(words) - 1)}
               for words in (["tea"], ["teapot", "tea"], ["tea", "teapot"], ["teapot"])]

    def make(name="run", *, use_image=False, tracker=None, metric_fn=None, seed=13, overwrite=False, **options):
        training = {"batch_size": 1, "grad_accum_steps": 1, "lr": 0.01, "weight_decay": 0.01,
                    "warmup_ratio": 0.25, "max_epochs": 2, "max_steps": None,
                    "early_stopping_patience": 20, "eval_every_steps": 3, "save_every_steps": 1, **options}
        config = {"seed": seed, "precision": "bf16", "output_dir": str(tmp_path / name), "training": training}
        return Trainer(make_tagger(use_image=use_image), records, records, config,
                       tracker=tracker, metric_fn=metric_fn, overwrite=overwrite)
    return make


@pytest.mark.parametrize("device, requested, capability, expected", [
    ("cpu", "bf16", 0, "fp32"), ("cpu", "fp16", 0, "fp32"), ("cpu", "fp32", 0, "fp32"),
    ("cuda", "bf16", 7, "fp16"), ("cuda", "bf16", 8, "bf16"), ("cuda", "fp16", 7, "fp16"),
])
def test_precision_policy(monkeypatch, caplog, device, requested, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (capability, 0))
    assert resolve_precision(device, requested) == expected
    assert ("falling back" in caplog.text) == (device == "cuda" and requested == "bf16" and capability < 8)
    with pytest.raises(ValueError):
        resolve_precision(device, "invalid")


@pytest.mark.parametrize("use_image", [False, True])
def test_training_collate_and_update(training_case, use_image, monkeypatch):
    trainer = training_case(use_image=use_image)
    inputs = trainer.collate(trainer.train_records[:2])
    assert inputs["input_ids"].shape == (2, 5)
    assert inputs["labels"][0, 3:].tolist() == [-100, -100]
    assert inputs["bbox"][0, 3:].count_nonzero() == 0
    assert ("pixel_values" in inputs) == use_image
    assert trainer.precision == "fp32" and trainer.scaler is None
    clip = Mock(wraps=torch.nn.utils.clip_grad_norm_)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    assert len(trainer.fit(stop_after=1)) == 1 and clip.call_args.args[1] == 1.0
    assert trainer.scheduler.last_epoch == 1 and trainer.scheduler.get_last_lr()[0] == 0.005


@pytest.mark.parametrize("accumulation", [1, 2, 3])
def test_resume_reproduces_losses_and_rng(training_case, accumulation):
    full = training_case("full", grad_accum_steps=accumulation)
    losses = full.fit()
    assert len(losses) == ((4 + accumulation - 1) // accumulation) * 2
    assert np.mean(losses[-2:]) < np.mean(losses[:2]) and full.scheduler.get_last_lr() == [0.0]
    rng = (random.random(), np.random.rand(), torch.rand(2))
    partial = training_case("resume", grad_accum_steps=accumulation)
    partial.fit(stop_after=1)
    assert partial.state["step_in_epoch"] == accumulation
    resumed = training_case("resume", grad_accum_steps=accumulation, overwrite=accumulation == 2)
    assert resumed.fit(resume="auto") == losses
    assert resumed.state == full.state and resumed.scheduler.state_dict() == full.scheduler.state_dict()
    for actual, expected in zip(resumed.tagger.parameters(), full.tagger.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (random.random(), np.random.rand()) == rng[:2]
    torch.testing.assert_close(torch.rand(2), rng[2], atol=0, rtol=0)
    state = torch.load(resumed.run_dir / "last/trainer_state.pt", weights_only=False)
    assert {"optimizer", "scheduler", "scaler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"} <= state.keys()


def test_early_stopping_tracking_and_init_from(training_case, tmp_path, monkeypatch):
    tracker = Mock()
    trainer = training_case(tracker=tracker, metric_fn=lambda gold, pred: 0.5,
                            eval_every_steps=1, early_stopping_patience=1)
    assert len(trainer.fit()) == 2 and trainer.state["patience_counter"] == 1
    assert tracker.log_params.call_args.args[0]["seed"] == 13
    assert tracker.log_metrics.call_count == 2
    rows = [json.loads(line) for line in (trainer.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["config"] and rows[0]["git_commit"] and rows[-1]["val_f1"] == 0.5
    checkpoint = trainer.run_dir / "best"
    assert all((checkpoint / name).is_file() for name in ("labels.json", "config.yaml", "trainer_state.pt", "tokenizer_config.json"))
    fresh = Trainer(None, trainer.train_records, trainer.val_records,
                    {"output_dir": str(tmp_path / "fresh")}, init_from=checkpoint)
    assert not fresh.optimizer.state and fresh.state["global_step"] == 0
    for actual, expected in zip(fresh.tagger.parameters(), load_tagger(checkpoint).parameters()):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fresh.fit(resume="auto")
    trainer.metric_fn = None
    monkeypatch.setattr(trainer.tagger, "predict_batch", lambda records, **kwargs:
                        [{"labels": ["O"]}] + [{"labels": r["labels"]} for r in records[1:]])
    assert trainer.evaluate() == pytest.approx(6 / 7)  # Three correct entities, one missed.


def test_checkpoint_failure_recovery_and_split_guards(training_case, monkeypatch):
    trainer = training_case()
    trainer.fit(stop_after=1)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", Mock(side_effect=OSError("interrupted write")))
        with pytest.raises(OSError, match="interrupted write"):
            trainer.save_checkpoint()
    last = trainer.run_dir / "last"
    assert (last / "trainer_state.pt").is_file()
    with (trainer.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"kind": "train", "global_step": 999}\n')  # Unsaved work before a crash.
    last.replace(trainer.run_dir / ".last.previous")  # Simulate a killed publication swap.
    restored = training_case()
    restored.resume()
    assert restored.state == trainer.state and last.is_dir()
    assert all(json.loads(line)["global_step"] != 999
               for line in (trainer.run_dir / "metrics.jsonl").read_text().splitlines())
    bad = [{**trainer.train_records[0], "split": "test"}]
    held_out = type("HeldOut", (list,), {"split": "test"})(trainer.train_records)
    for action in (lambda: trainer.collate(bad), lambda: trainer.evaluate(bad),
                   lambda: trainer.evaluate(split="test"), lambda: trainer.evaluate(held_out),
                   lambda: trainer.collate(held_out),
                   lambda: Trainer(trainer.tagger, bad, bad, trainer.config)):
        with pytest.raises(ValueError, match="test split"):
            action()


@pytest.mark.parametrize("artifact", ["last", "best", "metrics.jsonl"])
def test_fresh_run_refuses_each_existing_artifact(training_case, artifact):
    trainer = training_case()
    path = trainer.run_dir / artifact
    path.write_text("existing log", encoding="utf-8") if artifact.endswith("jsonl") else path.mkdir()
    with pytest.raises(FileExistsError, match="resume.*overwrite=True"):
        trainer.fit(stop_after=1)
    assert path.exists()
    if artifact.endswith("jsonl"):
        assert path.read_text(encoding="utf-8") == "existing log"
    else:
        assert not (trainer.run_dir / "metrics.jsonl").exists()


def test_fresh_seed_preserves_prior_run_and_overwrite_replaces_it(training_case, tmp_path):
    original = training_case(eval_every_steps=1)
    original.fit(stop_after=1)
    snapshot = {p.relative_to(original.run_dir): p.read_bytes() for p in original.run_dir.rglob("*") if p.is_file()}
    fresh = training_case(seed=42, eval_every_steps=1)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        fresh.fit(stop_after=1)
    assert snapshot == {p.relative_to(fresh.run_dir): p.read_bytes() for p in fresh.run_dir.rglob("*") if p.is_file()}
    (fresh.run_dir / "old-only.txt").write_text("old artifact", encoding="utf-8")
    sibling = tmp_path / "outside.txt"
    sibling.write_text("keep", encoding="utf-8")
    replacement = training_case(seed=42, overwrite=True, eval_every_steps=1)
    replacement.fit(stop_after=1)
    replacement.fit(stop_after=2)  # Continuing the same Trainer must not clear its run.
    rows = [json.loads(line) for line in (replacement.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows if row["kind"] == "run"] == [42]
    assert replacement.state["global_step"] == 2 and not (replacement.run_dir / "old-only.txt").exists()
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert load_tagger(replacement.run_dir / "best").config["seed"] == 42
