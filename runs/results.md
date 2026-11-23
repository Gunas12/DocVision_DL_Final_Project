# Results

Owner: user#4. Validation numbers until the agreed test day; every row links to a run id exported in this folder.

## Main table

| Model | Field F1 | Exact match | Relation F1 | CPU latency (ms) | Review effort | Run id |
|---|---|---|---|---|---|---|
| B0 Rule baseline | | | - | | | |
| B1 BiLSTM-CRF | | | - | | | |
| B2 XLM-R (text only) | | | - | | | |
| M1 LayoutLMv3-base | | | - | | | |
| M2 LiLT | | | - | | | |
| M1 + relation + validation | | | | | | |
| M1 + relation + validation + retrain on corrections | | | | | | |

## Ablation (3 seeds, mean +- std)

| Variant | Field F1 | Delta vs M1 |
|---|---|---|
| no layout | | |
| no image | | |
| frozen lower layers | | |
| no relation head (heuristic grouping) | | |

## Robustness (OCR noise)

| Model | Clean F1 | Noisy F1 | Delta |
|---|---|---|---|
