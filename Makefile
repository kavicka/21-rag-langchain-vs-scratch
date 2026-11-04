.PHONY: sync data index eval-retrieval eval-generation engineering judge trace plot api web test lint

# The Hugging Face Xet store is blocked on some networks (it was in the cloud run): plain HTTPS downloads work.
export HF_HUB_DISABLE_XET := 1
PIPELINES := scratch lc_tuned lc_default lg_tuned lg_default

sync:
	uv sync --all-groups
	cd frontend && npm ci

data:             ## re-download exactly the 396 trials of step 15 (eval/study_ids.txt)
	uv run python -m rag.ingest --from-ids

index:            ## step 15 indexes (fixed + section) and the two LangChain indexes (default + tuned)
	uv run python -m rag.index
	uv run python -m lc.index

eval-retrieval:   ## no LLM needed
	uv run python eval/run_eval.py --retrieval

eval-generation:  ## one process per pipeline so only one copy of the LLM is in memory
	for p in $(PIPELINES); do uv run python eval/run_eval.py --generation --pipeline $$p; done

engineering:      ## lines of code, dependencies, import time, framework overhead
	uv run python eval/engineering.py

judge:            ## RAGAS LLM-as-judge vs the rule scorer on lc_default's answers
	uv run --group judge python eval/judge.py --pipeline lc_default --balanced 24

trace:            ## self-hosted Arize Phoenix on :6006, traces of lc_tuned and lg_tuned
	uv run --group trace python eval/trace.py

plot:
	uv run python plot.py

api:
	uv run uvicorn api.main:app --port 8000

web:
	cd frontend && npm run dev

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
