# Ownership

Every folder and file has exactly one owner. **You edit only the files you own.**
If you need a change in someone else's file, open an issue or send a pull request; the owner reviews it
(this is enforced by `.github/CODEOWNERS`).

| Member | GitHub handle | Name                  | Role                                                                 |
| ------ | ------------- | --------------------- | -------------------------------------------------------------------- |
| user#1 | @user1-github | _Surname, Given name_ | Lead, data, infrastructure/Docker, report assembly                   |
| user#2 | @user2-github | _Surname, Given name_ | Modelling: XLM-R baseline, LayoutLMv3, LiLT, training infrastructure |
| user#3 | @user3-github | _Surname, Given name_ | Structure and validation, slide assembly                             |
| user#4 | @user4-github | _Surname, Given name_ | Experiments, BiLSTM-CRF, evaluation, MLOps                           |
| user#5 | @user5-github | _Surname, Given name_ | Product: OCR, pipeline, API, UI, review loop                         |

Default reviewer (cycle): user#1 is reviewed by user#2, user#2 by user#3, user#3 by user#4, user#4 by user#5, user#5 by user#1.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## user#1 - Lead, data, infrastructure/Docker, report assembly

- `README.md` - Project overview (temporary until the final week)
- `OWNERS.md` - Who owns which folders and files
- `CONTRIBUTING.md` - Branching, commits, pull requests and code review rules
- `LICENSE` - MIT license
- `.gitignore` - Ignore data, checkpoints and experiment logs
- `Makefile` - Shortcuts: setup, data, train, reproduce, test, lint
- `requirements.txt` - Python dependencies (pin with pip freeze after the first working env)
- `contribution_report.pdf` - Contribution report required at repo root (placeholder)
- `Dockerfile` - CPU container for API and UI
- `docker-compose.yml` - Run API and UI together
- `.github/workflows/ci.yml` - CI: lint and tests on every push
- `.github/PULL_REQUEST_TEMPLATE.md` - Pull request template with author and reviewer checklists
- `.github/CODEOWNERS` - Automatic review requests per path
- `data/README.md` - How to obtain the datasets, licences
- `data/splits/cord_split_v1.json` - Fixed CORD split (committed)
- `data/splits/docbank_subset_v1.json` - DocBank subset page ids
- `scripts/download_data.sh` - Download CORD and the DocBank subset
- `scripts/prepare_docbank_subset.py` - Seeded DocBank subset selection
- `scripts/make_noisy_eval.py` - Build OCR-noisy evaluation variants
- `scripts/check_template_leftovers.sh` - Detect leftover template text in the report
- `src/docvision/utils/seed.py` - Global seeding
- `src/docvision/data/schema.py` - Label set and output JSON schema
- `src/docvision/data/cord.py` - CORD loader (words, boxes, BIO labels, line-item groups)
- `src/docvision/data/docbank.py` - DocBank loader
- `src/docvision/data/dataset.py` - Torch dataset, box normalisation, collate
- `src/docvision/data/augment.py` - Box jitter and OCR-noise augmentation
- `tests/test_data.py` - Split leakage and loader tests
- `report/report.tex` - IEEE report entry point (assembly only)
- `report/refs.bib` - Bibliography
- `report/sections/abstract.tex` - Abstract
- `report/sections/experimental_setup.tex` - Datasets, splits, licences
- `report/sections/conclusion.tex` - Conclusion
- `report/sections/tools_ack.tex` - Tools and acknowledgements (AI disclosure)
- `presentation/slides/u1_title_data_conclusion.tex` - Slides S1, S2, S3, S14
- `docs/contribution_report_draft.md` - Draft of the contribution report

## user#2 - Modelling: XLM-R baseline, LayoutLMv3, LiLT, training infrastructure

- `configs/base.yaml` - Shared settings: seeds, precision, accumulation, checkpoints
- `configs/text_baseline.yaml` - Text-only XLM-R baseline
- `configs/layoutlmv3.yaml` - LayoutLMv3 on CORD
- `configs/lilt.yaml` - LiLT (second model family)
- `configs/docbank.yaml` - DocBank subset
- `src/docvision/models/text_baseline.py` - B2 text-only XLM-R tagger
- `src/docvision/models/layoutlmv3_tagger.py` - M1 LayoutLMv3 token tagger
- `src/docvision/models/lilt_tagger.py` - M2 LiLT token tagger
- `src/docvision/training/trainer.py` - Training loop with checkpoint and resume
- `src/docvision/training/train_tagger.py` - CLI: train taggers
- `tests/test_models_smoke.py` - Model smoke tests
- `report/sections/method_model.tex` - Models and training
- `report/sections/discussion.tex` - Discussion and limitations
- `presentation/slides/u2_models_training.tex` - Slides S5, S6
- `docs/compute_plan.md` - Measured memory and time budget

## user#3 - Structure and validation, slide assembly

- `configs/relation.yaml` - Relation head
- `src/docvision/models/rule_baseline.py` - B0 rule-based baseline
- `src/docvision/models/relation_head.py` - Pairwise relation head for line items
- `src/docvision/training/train_relation.py` - CLI: train relation head
- `src/docvision/postprocess/decode.py` - Decode BIO tags into fields
- `src/docvision/postprocess/grouping.py` - Heuristic line-item grouping
- `src/docvision/validation/rules.py` - Arithmetic and format validation rules
- `src/docvision/validation/confidence.py` - Confidence scores and review routing
- `tests/test_decode_grouping.py` - Decode and grouping tests
- `tests/test_rules.py` - Validation rule tests
- `report/sections/related_work.tex` - Related work
- `report/sections/method_structure.tex` - Relation extraction and validation
- `presentation/main.tex` - Beamer entry point (assembly only)
- `presentation/slides/u3_structure_validation.tex` - Slides S7, S8

## user#4 - Experiments, BiLSTM-CRF, evaluation, MLOps

- `configs/baseline_crf.yaml` - BiLSTM-CRF baseline
- `configs/ablations/no_layout.yaml` - Ablation: no layout
- `configs/ablations/no_image.yaml` - Ablation: no image
- `configs/ablations/frozen_lower.yaml` - Ablation: frozen lower layers
- `scripts/reproduce.sh` - One command to reproduce headline results
- `scripts/run_ablations.sh` - Run ablations and seeds
- `scripts/export_runs.py` - Export tracked runs to runs/
- `scripts/benchmark_latency.py` - Latency, memory and model size benchmark
- `src/docvision/utils/tracking.py` - Experiment tracking wrapper
- `src/docvision/models/baseline_crf.py` - B1 BiLSTM-CRF baseline
- `src/docvision/training/train_baseline_crf.py` - CLI: train BiLSTM-CRF
- `src/docvision/evaluation/metrics.py` - Field F1, exact match, relation F1
- `src/docvision/evaluation/review_effort.py` - Review effort and flag precision/recall
- `src/docvision/evaluation/error_analysis.py` - Error analysis
- `src/docvision/evaluation/plots.py` - Plots
- `src/docvision/evaluation/evaluate.py` - CLI: evaluate on val or test
- `tests/test_metrics.py` - Metric correctness tests
- `runs/results.md` - Results table linked to run ids
- `report/sections/evaluation_setup.tex` - Metrics, baselines, protocol
- `report/sections/results.tex` - Results
- `presentation/slides/u4_results_latency.tex` - Slides S9, S10, S13
- `docs/gpu_queue.md` - Team GPU queue

## user#5 - Product: OCR, pipeline, API, UI, review loop

- `src/docvision/ocr/engine.py` - OCR wrapper returning words and boxes
- `src/docvision/pipeline/infer.py` - End-to-end inference pipeline
- `src/docvision/feedback/store.py` - Store reviewer corrections as JSONL
- `src/docvision/feedback/finetune.py` - Fine-tune from stored corrections
- `src/docvision/serve/api.py` - FastAPI service
- `src/docvision/serve/ui.py` - Streamlit review UI
- `tests/test_api.py` - API tests
- `report/sections/introduction.tex` - Introduction
- `report/sections/method_system.tex` - System: OCR, API, UI, review loop
- `report/sections/deployment_note.tex` - Deployment note
- `presentation/slides/u5_product_demo.tex` - Slides S4, S11, S12
- `docs/ui_spec.md` - UI components mapped to API and modules
- `docs/ui_mockups/01_upload_queue.png` - UI mockup: upload and queue
- `docs/ui_mockups/02_review_screen.png` - UI mockup: review screen
- `docs/ui_mockups/03_dashboard.png` - UI mockup: dashboard

Files marked above that do not exist yet are created by their owner when the work starts, in the owner's own folder.
