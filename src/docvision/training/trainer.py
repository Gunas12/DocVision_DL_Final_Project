"""Resumable training on in-memory records: words, boxes, labels, optional PIL image.

Callers seed model initialization; Trainer seeds runtime RNG. Samplers use
seed + epoch. Checkpoints occur at optimizer boundaries, replaying unsaved work.
Optional user#4 adapters: metrics.entity_f1(gold, predicted), and
tracking.create_tracker(config, run_dir) returning log_params/log_metrics methods.
"""

import importlib
import importlib.util
import json
import logging
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import get_linear_schedule_with_warmup

from docvision.models.layoutlmv3_tagger import _merge, _reject_test_split, load_tagger


def resolve_precision(device, precision):
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16 or fp32")
    if torch.device(device).type != "cuda":
        return "fp32"
    if precision == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        logging.getLogger(__name__).warning("GPU compute capability < 8: falling back from bf16 to fp16")
        return "fp16"
    return precision


class Trainer:
    def __init__(self, tagger, train_records, val_records, config=None, *, device="cpu",
                 init_from=None, tracker=None, metric_fn=None):
        self.device, self.init_from = torch.device(device), init_from
        self.tagger = load_tagger(init_from, device) if init_from else tagger.to(device)
        self.config = _merge(self.tagger.config, config or {})
        _reject_test_split(self.config)
        for records in (train_records, val_records):
            _reject_test_split({"split": getattr(records, "split", None)})
            if not len(records):
                raise ValueError("Training and validation records must be nonempty")
            for record in records:
                _reject_test_split(record)
        self.train_records, self.val_records = train_records, val_records
        self.options = self.config["training"]
        for key in ("batch_size", "grad_accum_steps", "max_epochs", "eval_every_steps",
                    "save_every_steps", "early_stopping_patience", "max_steps"):
            if key == "max_steps" and self.options.get(key) is None:
                continue
            if not isinstance(self.options[key], int) or self.options[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.seed = self.config.get("seed", 13)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.precision = resolve_precision(device, self.config.get("precision", "fp32"))
        self.scaler = torch.amp.GradScaler("cuda") if self.precision == "fp16" else None
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]
        self.optimizer = torch.optim.AdamW(self.tagger.parameters(), lr=self.options["lr"],
                                           weight_decay=self.options.get("weight_decay", 0.01))
        batches = math.ceil(len(train_records) / self.options["batch_size"])
        self.total_steps = math.ceil(batches / self.options["grad_accum_steps"]) * self.options["max_epochs"]
        self.total_steps = min(self.total_steps, self.options.get("max_steps") or self.total_steps)
        if self.total_steps < 1 or not 0 <= self.options.get("warmup_ratio", 0.1) < 1:
            raise ValueError("Require positive max_steps and warmup_ratio in [0, 1)")
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(self.total_steps * self.options.get("warmup_ratio", 0.1)), self.total_steps,
        )
        self.run_dir = Path(self.config.get("output_dir", "checkpoints/layoutlmv3")).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = {"epoch": 0, "global_step": 0, "step_in_epoch": 0, "best_metric": -math.inf,
                      "patience_counter": 0, "last_eval_step": -1, "losses": []}
        metrics_module, tracking_module = "docvision.evaluation.metrics", "docvision.utils.tracking"
        if metric_fn is None and importlib.util.find_spec(metrics_module):
            metric_fn = importlib.import_module(metrics_module).entity_f1
        if tracker is None and importlib.util.find_spec(tracking_module):
            tracker = importlib.import_module(tracking_module).create_tracker(self.config, self.run_dir)
        self.metric_fn, self.tracker = metric_fn, tracker
        try:
            self.commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                         capture_output=True, text=True, check=False).stdout.strip() or "unknown"
        except OSError:
            self.commit = "unknown"
        if tracker:
            tracker.log_params({"config": json.dumps(self.config), "seed": self.seed, "git_commit": self.commit})
        self._log("run", config=self.config, seed=self.seed, git_commit=self.commit, precision=self.precision)

    def _log(self, kind, **values):
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "global_step": self.state["global_step"], **values}) + "\n")
        if self.tracker and kind == "eval":
            self.tracker.log_metrics(values, step=self.state["global_step"])

    def collate(self, records):
        _reject_test_split({"split": getattr(records, "split", None)})
        for record in records:
            _reject_test_split(record)
        encodings = [self.tagger.encode(r["words"], r["boxes"], r.get("image"), r["labels"], padding=False)
                     for r in records]
        inputs = self.tagger.processor.tokenizer.pad(
            [{k: v[0] for k, v in e.items() if k != "pixel_values"} for e in encodings],
            padding=True, return_tensors="pt",
        )
        padding = ~inputs["attention_mask"].bool()
        inputs["labels"][padding], inputs["bbox"][padding] = -100, 0
        if self.tagger.use_image:
            inputs["pixel_values"] = torch.cat([e["pixel_values"] for e in encodings])
        return inputs.to(self.device)

    def evaluate(self, records=None, *, split="validation"):
        if split not in ("val", "validation"):
            raise ValueError("Only validation evaluation is allowed; the test split is reserved")
        records = self.val_records if records is None else records
        _reject_test_split({"split": getattr(records, "split", None)})
        for record in records:
            _reject_test_split(record)
        predicted = self.tagger.predict_batch(records, batch_size=self.options["batch_size"])
        gold, labels = [r["labels"] for r in records], [r["labels"] for r in predicted]
        if self.metric_fn:
            score = float(self.metric_fn(gold, labels))
        else:
            from seqeval.metrics import f1_score
            from seqeval.scheme import IOB2
            score = float(f1_score(gold, labels, mode="strict", scheme=IOB2, zero_division=0))
        if not math.isfinite(score):
            raise ValueError("Validation F1 must be finite")
        self.state["last_eval_step"] = self.state["global_step"]
        improved = score > self.state["best_metric"]
        self.state["patience_counter"] = 0 if improved else self.state["patience_counter"] + 1
        self._log("eval", val_f1=score)
        if improved:
            self.state["best_metric"] = score
            self.save_checkpoint("best")
        return score

    def _recover(self, name):
        path, previous = self.run_dir / name, self.run_dir / f".{name}.previous"
        if previous.exists() and previous.resolve().parent != self.run_dir:
            raise ValueError("Checkpoint backup must stay inside the run directory")
        if not path.exists() and previous.exists():
            previous.replace(path)
        return path

    def save_checkpoint(self, name="last"):
        """Publish a complete directory by rename; recover the backup after a killed swap."""
        if name not in ("last", "best"):
            raise ValueError("Checkpoint name must be last or best")
        path, previous = self._recover(name), self.run_dir / f".{name}.previous"
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=self.run_dir) as temporary:
            directory = Path(temporary)
            self.tagger.save_pretrained(directory)
            (directory / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
            state = {**self.state, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                     "log_offset": (self.run_dir / "metrics.jsonl").stat().st_size,
                     "scaler": self.scaler.state_dict() if self.scaler else None, "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            torch.save(state, directory / "trainer_state.pt")
            if previous.exists():
                shutil.rmtree(previous)
            if path.exists():
                path.replace(previous)
            try:
                directory.replace(path)
            finally:
                self._recover(name)
            if previous.exists():
                shutil.rmtree(previous)

    def resume(self, checkpoint="auto"):
        if self.init_from:
            raise ValueError("init_from and resume are mutually exclusive")
        path = self._recover("last") if checkpoint == "auto" else Path(checkpoint)
        if checkpoint == "auto" and not path.exists():
            return
        restored = load_tagger(path, self.device)
        if any(restored.config.get(k) != self.config.get(k) for k in
               ("training", "seed", "max_length", "ablation", "precision")) or restored.id2label != self.tagger.id2label:
            raise ValueError("Resume requires the original training schedule and label map")
        self.tagger.model.load_state_dict(restored.model.state_dict())
        self.tagger.processor = restored.processor
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state.pop("optimizer"))
        self.scheduler.load_state_dict(state.pop("scheduler"))
        scaler = state.pop("scaler")
        if self.scaler and scaler:
            self.scaler.load_state_dict(scaler)
        random.setstate(state.pop("python_rng"))
        np.random.set_state(state.pop("numpy_rng"))
        torch.set_rng_state(state.pop("torch_rng"))
        cuda_rng = state.pop("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        offset = state.pop("log_offset")
        if path.resolve().parent == self.run_dir:
            with (self.run_dir / "metrics.jsonl").open("r+b") as stream:
                stream.truncate(offset)
        self.state = state

    def fit(self, *, resume=None, stop_after=None):
        """Return update losses; stop_after interrupts at an absolute optimizer step."""
        if resume:
            self.resume(resume)
        if stop_after and self.state["global_step"] >= stop_after:
            return self.state["losses"]
        batch_size, accumulation = self.options["batch_size"], self.options["grad_accum_steps"]
        while self.state["epoch"] < self.options["max_epochs"] and self.state["global_step"] < self.total_steps:
            if self.state["patience_counter"] >= self.options["early_stopping_patience"]:
                break
            order = torch.randperm(len(self.train_records), generator=torch.Generator().manual_seed(
                self.seed + self.state["epoch"])).tolist()
            batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            self.tagger.train()
            for start in range(self.state["step_in_epoch"], len(batches), accumulation):
                group = batches[start:start + accumulation]
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for indices in group:
                    inputs = self.collate([self.train_records[i] for i in indices])
                    with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.precision != "fp32"):
                        loss = self.tagger(**inputs).loss / len(group)
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss must be finite")
                    loss_sum += float(loss.detach())
                    (self.scaler.scale(loss) if self.scaler else loss).backward()
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.tagger.parameters(), 1.0)
                scale = self.scaler.get_scale() if self.scaler else None
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.state["step_in_epoch"] = start + len(group)
                if not self.scaler or self.scaler.get_scale() >= scale:
                    self.scheduler.step()
                    self.state["global_step"] += 1
                    self.state["losses"].append(loss_sum)
                    self._log("train", loss=loss_sum, lr=self.scheduler.get_last_lr()[0])
                step = self.state["global_step"]
                if step % self.options["eval_every_steps"] == 0 and step != self.state["last_eval_step"]:
                    self.evaluate()
                if step % self.options["save_every_steps"] == 0 or (stop_after and step >= stop_after):
                    self.save_checkpoint()
                if stop_after and step >= stop_after:
                    return self.state["losses"]
                if step >= self.total_steps or self.state["patience_counter"] >= self.options["early_stopping_patience"]:
                    break
            if self.state["step_in_epoch"] == len(batches):
                self.state["epoch"] += 1
                self.state["step_in_epoch"] = 0
            if self.state["last_eval_step"] != self.state["global_step"]:
                self.evaluate()
            self.save_checkpoint()
        return self.state["losses"]
