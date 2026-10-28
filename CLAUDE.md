# Project conventions

## Writing style
- In prose, never join or end clauses with dashes or semicolons. Use full stops and short sentences instead. Hyphens inside compound words are fine.
- This applies to the README, docs, UI text, commit messages, release notes, issue and PR text, and replies to the owner.
- Write the way a person would. Avoid filler, hype and stacked qualifiers.

## Code
- Do not write code comments. Names, types and tests carry the meaning.
- Python 3.12, ruff for lint and format, pytest for tests. Run `make check` before committing.
- Tests never touch the network and never need the real dataset or an LLM server.

## Analysis rules
- All statistics live in `core/`. The LLM layer in `app/` never computes a number.
- Every result carries a `data_source` field. Never present synthetic data as real.
- Keep the default sample at 500,000 rows and the default seed at 42. Do not change the seed to get a nicer result.
- Report estimates with their intervals, and state the limits next to the claim.

## Language model
- `SYSTEM_PROMPT` in `app/agent.py` and the tool descriptions determine the published evaluation results. After changing either, rerun `make eval` and update the README table and VALIDATION.md.
- The weekly workflow in `.github/workflows/llm-eval.yml` repeats the evaluation against the real model.

## Commands
- `make install`, `make check`, `make run`, `make validate`, `make export-demo`, `make eval`, `make docker-up`.
