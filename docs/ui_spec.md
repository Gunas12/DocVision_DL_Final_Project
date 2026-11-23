# UI specification

Owner: user#5. Mockups: `docs/ui_mockups/01_upload_queue.png`, `02_review_screen.png`, `03_dashboard.png` (illustrative data).

| Screen | Component | API / module | Notes |
|---|---|---|---|
| Upload and queue | Dropzone, queue table | `POST /extract`, `pipeline/infer.py` | Batch upload; status: processing, needs review, auto-approved |
| Upload and queue | Sidebar: model, OCR engine, threshold, validation rules | pipeline configuration | Threshold is calibrated on the validation split |
| Review | Image with coloured boxes | `ocr/engine.py` | Click a row to highlight its box (and vice versa) |
| Review | Editable fields with confidence chips | structure output of `postprocess/` | Flag: confidence below threshold or a rule failed |
| Review | Validation banner | `validation/rules.py` | For example, sum of items does not match subtotal |
| Review | Approve, Save corrections, Skip | `POST /correct`, `feedback/store.py` | JSONL: image hash, prediction, correction, time |
| Dashboard | KPI cards, review effort before/after | `evaluation/review_effort.py` | |
| Dashboard | Retrain, Export corrections | `feedback/finetune.py` | Suggested after 50 corrections |

Design rules: colour by field type (item blue, quantity purple, price green, subtotal/tax orange, total red);
flag yellow, error red, approved green; every screen fits 1366x768; saving a correction takes one click.
