# Contributing

This document defines how we work. Follow it for code, experiments, the report and the slides.

## 1. Language

All code, comments, docstrings, configuration keys, commit messages, issue and pull request titles and descriptions,
the README, the report and the slides are written in **English**.

## 2. Ownership

Each file has one owner (see [OWNERS.md](OWNERS.md) and `.github/CODEOWNERS`).
Edit only your own files. For changes in someone else's file, open an issue or a pull request and tag the owner.
Each member writes their own report sections and their own slides.

## 3. Branching

- `main` is protected. Nobody pushes to `main` directly.
- Branch from `main`: `<type>/u<N>-<short-topic>`, for example `feat/u2-layoutlmv3-tagger`.
- Types: `feat`, `fix`, `exp` (experiment), `docs`, `test`, `chore`, `refactor`.
- Keep branches short-lived (ideally under 2 days). Rebase on `main` before opening the pull request.

## 4. Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary`, at most 72 characters.

```
feat(models): add LayoutLMv3 token tagger
fix(data): align subword labels with first-token rule
exp(layoutlmv3): lr 5e-5, seed 13, run_id=lmv3-cord-s13
docs(report): draft method_model section
```

The body explains why, not what. Commit small and often. Never commit data, checkpoints, secrets or large files.

## 5. Issues

One task per issue. Title in imperative form. Add the label of the owner (`u1`..`u5`) and the phase (`phase-a`, `phase-b`, `phase-c`).
Each issue has a clear done-criterion and a due date.

## 6. Pull requests

1. Open a **draft** pull request early for anything larger than a few hours of work; mark it ready when done.
2. One concern per pull request, preferably under 400 changed lines (excluding generated files).
3. Title follows Conventional Commits. Fill in the pull request template completely. Link the issue (`Closes #N`).
4. CI must be green (lint and tests). Do not ask for review while CI is red.
5. The author never merges without approval and never approves their own pull request.
6. Merge with **squash and merge**; delete the branch afterwards.

## 7. Code review

**Who reviews.** The default reviewer is the next member in the cycle (user#1 is reviewed by user#2, user#2 by user#3, user#3 by user#4,
user#4 by user#5, user#5 by user#1). If the pull request touches files owned by someone else, that owner is requested automatically
(CODEOWNERS) and must also approve. If the reviewer is unavailable for 24 hours, the next member in the cycle takes over.

**Required to merge.** At least one approval from a non-author, approvals from all requested code owners, green CI,
no unresolved blocking comments.

**Response time.** First review response within 24 hours; author answers within 24 hours.

**How to comment.** Be specific, kind and technical. Prefix comments so the author knows what is expected:

| Prefix | Meaning |
|---|---|
| `blocking:` | Must be fixed before merge |
| `suggestion:` | Recommended change; author may decline with a reason |
| `question:` | Reviewer needs an explanation |
| `nit:` | Minor style point, optional |
| `praise:` | Something done well |

Review outcomes: **Approve**, **Request changes** (at least one `blocking:` comment), or **Comment**.
Disagreements that are not resolved in two rounds are decided by a third member within 24 hours.

**Reviewer checklist.**

- Correctness and edge cases; the change does what the description says.
- Tests exist for new logic and pass; no unrelated changes.
- Readable code, clear names, English comments and docstrings; no dead code or debug prints.
- No secrets, no large files, no absolute local paths.
- Reproducibility: fixed seed, configuration in `configs/`, no hidden state.
- Experiments: the pull request lists config, seed, run id and validation metrics compared with the baseline; the test split is **not** used.
- Report or slides: every number matches `runs/results.md`; figures are readable; citations are present; no template or TODO text.
- You can explain the change in your own words (every member must be able to defend the whole system).

## 8. Experiments

- Every run reads a YAML config from `configs/` and logs config, seed, git commit and metrics (MLflow, exported to `runs/`).
- Tune on the validation split only. The test split is touched once, on the agreed day, by the evaluation owner.
- Training must be resumable (checkpoint every epoch), because GPU time on the shared slice can end at any moment.

## 9. Definition of done for a pull request

Code and tests merged to `main`, CI green, documentation or config updated, issue closed, and (for experiments) the run recorded in `runs/results.md`.
