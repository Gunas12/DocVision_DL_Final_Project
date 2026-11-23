# DocVision

Structure-aware document intelligence for receipts and invoices.

> **Status: work in progress.** This README is temporary. It will be replaced by a complete,
> professional README in the final week of the project.

## What it will do

DocVision reads a receipt or invoice image and produces structured data:

1. **OCR** extracts words and bounding boxes.
2. A **layout-aware token tagger** (LayoutLMv3 / LiLT) labels fields such as item name, quantity, price, subtotal, tax, total.
3. A **relation head** groups fields into line items.
4. **Rule-based validation** (for example, sum of items must match the subtotal) and confidence scores flag uncertain fields.
5. A **review UI** lets a person correct flagged fields; corrections are stored and used to fine-tune the model again.

## Repository layout

```
src/docvision/   Python package (data, models, training, postprocess, validation, evaluation, ocr, pipeline, feedback, serve)
configs/         Experiment configuration files (YAML)
scripts/         Data download, reproduction and benchmarking scripts
tests/           Unit and smoke tests
runs/            Exported experiment runs and the results table
report/          IEEE-format paper (one file per section)
presentation/    Beamer slides (one file per presenter)
docs/            UI specification and mockups, compute plan, GPU queue
```

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest ruff
pytest -q
```

Full environment (training, evaluation, demo) is added to `requirements.txt` as modules are implemented.

## Team and ownership

See [OWNERS.md](OWNERS.md) for who owns which folders and files, and
[CONTRIBUTING.md](CONTRIBUTING.md) for branching, commit and code review rules.

## Data and licences

Datasets: CORD and a subset of DocBank. See [data/README.md](data/README.md) for sources and licences.
Pretrained model weights keep their own licences (for example, LayoutLMv3 weights are non-commercial).

## License

Code: MIT (see [LICENSE](LICENSE)).
