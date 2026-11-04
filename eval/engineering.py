"""The engineering side of the comparison: code size, dependencies, import time and framework overhead.

None of this needs the LLM. Numbers go to results.json["engineering"].

  loc        logical lines (no blank lines, comments or docstrings) per file, grouped by job
  deps       packages and disk size of a venv with step 15's dependencies vs the same + the LangChain stack
  import_ms  fresh interpreter, median of 7: time to import each pipeline's modules
  overhead   per-call cost of the orchestration alone, with an in-memory retriever and a fake model,
             so retrieval and generation cost ~0 and only the framework is left

Run:  uv run python eval/engineering.py [--skip-deps]
"""

import argparse
import io
import json
import statistics
import subprocess
import sys
import tempfile
import time
import timeit
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import merge_results  # noqa: E402
from rag.config import ROOT  # noqa: E402

# What each pipeline needs for the same job. Shared helpers (ingest, scoring, eval) are not counted.
LOC_GROUPS = {
    "scratch": {
        "chunking + indexing": ["rag/chunking.py", "rag/index.py"],
        "retrieval (bm25, dense, rrf, rerank)": ["rag/retrieve.py"],
        "prompt + answer + citations": ["rag/answer.py"],
        "LLM providers": ["rag/llm.py"],
    },
    "langchain": {
        "chunking + indexing": ["lc/index.py"],
        "retrieval (bm25, dense, rrf, rerank)": ["lc/retrievers.py"],
        "prompt + answer + citations": ["lc/common.py", "lc/chain.py"],
        "LLM providers": ["lc/llm.py"],
        "corrective loop (LangGraph)": ["lc/graph.py"],
    },
}

BASE_DEPS = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.28",
    "pydantic-settings>=2.6",
    "qdrant-client>=1.12",
    "sentence-transformers>=3.3",
    "transformers>=4.51",
    "numpy>=2.1",
]
LC_DEPS = [
    "langchain>=1.0",
    "langchain-core>=1.0",
    "langchain-classic>=1.0",
    "langchain-community>=0.4",
    "langchain-text-splitters>=1.0",
    "langchain-huggingface>=1.0",
    "langchain-qdrant>=1.0",
    "langgraph>=1.0",
    "rank-bm25>=0.2.2",
]
IMPORTS = {
    "scratch": "import rag.retrieve, rag.answer, rag.llm",
    "langchain (LCEL)": "import lc.retrievers, lc.chain, lc.llm",
    "langchain + langgraph": "import lc.retrievers, lc.chain, lc.llm, lc.graph",
}


def logical_lines(path: Path) -> int:
    """Count lines that hold code: tokens other than comments, strings standing alone (docstrings) and NL."""
    lines: set[int] = set()
    prev = tokenize.NEWLINE
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
        if tok.type in skip:
            prev = tok.type if tok.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT) else prev
            continue
        is_docstring = tok.type == tokenize.STRING and prev in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
        prev = tok.type
        if not is_docstring:
            lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines)


def loc() -> dict:
    out: dict = {}
    for side, groups in LOC_GROUPS.items():
        out[side] = {g: sum(logical_lines(ROOT / f) for f in files) for g, files in groups.items()}
        out[side]["total"] = sum(out[side].values())
    return out


def _venv(reqs: list[str], where: Path) -> dict:
    subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(where)], check=True)
    py = where / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), *reqs], check=True)
    frozen = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True).stdout
    site = next(where.glob("lib/python*/site-packages"))
    mb = int(subprocess.run(["du", "-sm", str(site)], capture_output=True, text=True).stdout.split()[0])
    return {"packages": sorted(line.split("==")[0].lower() for line in frozen.splitlines() if "==" in line), "mb": mb}


def deps() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        base = _venv(BASE_DEPS, Path(tmp) / "base")
        full = _venv(BASE_DEPS + LC_DEPS, Path(tmp) / "full")
    added = sorted(set(full["packages"]) - set(base["packages"]))
    return {
        "base_packages": len(base["packages"]),
        "base_mb": base["mb"],
        "with_langchain_packages": len(full["packages"]),
        "with_langchain_mb": full["mb"],
        "added_packages": len(added),
        "added_mb": full["mb"] - base["mb"],
        "added": added,
    }


def import_ms(stmt: str, repeat: int = 7) -> float:
    code = f"import time; t = time.perf_counter(); {stmt}; print((time.perf_counter() - t) * 1000)"
    times = []
    for _ in range(repeat):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
        times.append(float(r.stdout.strip().splitlines()[-1]))
    return round(statistics.median(times), 1)


def overhead() -> dict:
    """µs per call of the orchestration alone: in-memory documents, fake model, so only the framework is timed."""
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    from lc.llm import FakeChatModel
    from pipelines import GraphPipeline, LCPipeline, run
    from rag.answer import build_messages, finalize
    from rag.llm import FakeLLM
    from rag.retrieve import Hit
    from tests.test_lc import ListRetriever

    f = lambda x: x  # noqa: E731
    plain = lambda x: f(f(f(x)))  # noqa: E731
    lcel = RunnableLambda(f) | RunnableLambda(f) | RunnableLambda(f)

    chunk = {"nct_id": "NCT00000001", "text": "[NCT00000001] T | Design\nPhase: Phase 2"}
    docs = [Document(chunk["text"], metadata={"nct_id": chunk["nct_id"]})] * 5
    hits, llm = [Hit(chunk, 1.0)] * 5, FakeLLM()

    def scratch_rag(q: str) -> str:  # what step 15 does per question, minus the real retriever and model
        return finalize("".join(llm.stream(build_messages(q, hits))), hits).text

    lc_p = LCPipeline("t", ListRetriever(docs=docs), FakeChatModel())
    # the grader says yes, so the graph runs retrieve → grade → generate: two model calls
    lg_p = GraphPipeline("t", ListRetriever(docs=docs), _YesThenAnswer())

    def per_call_us(fn, n: int) -> float:
        fn()  # warm-up
        return round(min(timeit.repeat(fn, number=n, repeat=5)) / n * 1e6, 1)

    return {
        "three_steps_plain_us": per_call_us(lambda: plain(1), 20000),
        "three_steps_lcel_us": per_call_us(lambda: lcel.invoke(1), 2000),
        "rag_scratch_us": per_call_us(lambda: scratch_rag("What phase?"), 2000),
        "rag_lcel_stream_us": per_call_us(lambda: run(lc_p, "What phase?"), 300),
        "rag_langgraph_stream_us": per_call_us(lambda: run(lg_p, "What phase?"), 100),
    }


def _YesThenAnswer():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=["yes", "Phase 2 [NCT00000001]"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-deps", action="store_true", help="skip building two throw-away venvs")
    args = ap.parse_args()
    out: dict = {"loc": loc()}
    print(json.dumps(out["loc"], indent=1))
    out["import_ms"] = {k: import_ms(v) for k, v in IMPORTS.items()}
    print(out["import_ms"])
    t0 = time.perf_counter()
    out["overhead"] = overhead()
    print(out["overhead"], f"({time.perf_counter() - t0:.0f} s)")
    if not args.skip_deps:
        out["deps"] = deps()
        print({k: v for k, v in out["deps"].items() if k != "added"})
    merge_results(["engineering"], out)


if __name__ == "__main__":
    main()
