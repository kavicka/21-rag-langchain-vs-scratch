"""Measure every pipeline on step 15's eval set with step 15's scorer, and merge the numbers into results.json.

Retrieval (no LLM): each retriever returns its top 10 for the 141 questions that have a gold trial.
  evidence@k  a chunk that contains the gold fact (e.g. "Enrollment: 240 participants") is in the top k
  trial@k     any chunk of the gold trial is in the top k
Generation (LLM): one pipeline per process, so only one copy of the model is in memory.
  accuracy, grounded, false_abstain, abstain_unans, bad_citations: as in step 15
  accuracy_no_leak  accuracy without the intervention questions whose answer is already in the trial title
  ttft              time from the question to the first answer token (retrieval + prompt processing)
  LangGraph only:   llm_calls per question, rewrite rate, and how often the grader was right

Run:  uv run python eval/run_eval.py --retrieval
      uv run python eval/run_eval.py --generation --pipeline lc_tuned [--limit 20]
"""

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines import PIPELINES, get_pipeline, run  # noqa: E402
from rag.answer import finalize  # noqa: E402
from rag.config import ROOT, get_settings  # noqa: E402
from rag.retrieve import Hit  # noqa: E402
from rag.scoring import is_correct, name_match  # noqa: E402

RESULTS = ROOT / "results.json"
RUNS = ROOT / "eval" / "runs"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open()]


def hardware() -> str:
    import os

    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU cores, no GPU (torch device cpu)"


def _ws(text: str) -> str:
    return " ".join(text.split())


def has_evidence(chunk: dict, q: dict) -> bool:
    return chunk["nct_id"] == q["nct_id"] and bool(q["evidence"]) and _ws(q["evidence"]) in _ws(chunk["text"])


def leaky(q: dict) -> bool:
    """The answer is already in the question: the drug name is part of the trial title (step 15 error analysis)."""
    return q["kind"] == "intervention" and any(name_match(g, q["question"]) for g in q["gold"])


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def merge_results(path: list[str], data: dict) -> None:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    node = res
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = data
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- retrieval
def retrieval_specs(only: list[str] | None = None) -> dict[str, tuple[Callable[[str], list[dict]], dict]]:
    """name → (question → top-10 chunk dicts, index stats). Models load once per index."""
    from lc.common import as_chunk
    from lc.retrievers import TUNED_MODES, LCRetrievers
    from rag.retrieve import MODES, Retriever, tokenize

    s = get_settings()
    specs: dict = {}
    stats = json.loads((s.data_dir / "index" / "section" / "stats.json").read_text())
    if only and all(name.startswith("ablation:") for name in only):  # no models or Qdrant needed
        scratch = Retriever("section", s, dense=False)
    else:
        scratch = Retriever("section", s)
        for m in MODES:
            specs[f"scratch:{m}"] = (lambda q, m=m: [h.chunk for h in scratch.search(q, m, 10)], stats)

    # Ablation: LangChain's BM25Retriever (rank_bm25) on step 15's own chunks, with step 15's tokenizer.
    # Same chunks, same tokens: any difference to scratch:bm25 comes from the BM25 implementation alone.
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    bm = BM25Retriever.from_documents(
        [Document(c["text"], metadata=c) for c in scratch.chunks], k=10, preprocess_func=tokenize
    )
    specs["ablation:rank_bm25_on_scratch_chunks"] = (lambda q: [d.metadata for d in bm.invoke(q)], stats)
    # ...and with step 15's k1 = 1.2 instead of rank_bm25's 1.5, so only the idf formula differs
    bm12 = BM25Retriever.from_documents(
        [Document(c["text"], metadata=c) for c in scratch.chunks],
        k=10,
        preprocess_func=tokenize,
        bm25_params={"k1": 1.2},
    )
    specs["ablation:rank_bm25_k1_1.2_on_scratch_chunks"] = (lambda q: [d.metadata for d in bm12.invoke(q)], stats)
    if only and all(name.startswith("ablation:") for name in only):
        return specs

    for variant, modes in [("default", ("dense", "dense_instruction")), ("tuned", TUNED_MODES)]:
        r = LCRetrievers(variant, s)
        stats = json.loads((s.data_dir / "lc" / variant / "stats.json").read_text())
        for m in modes:
            ret = r.dense_with_instruction(10) if m == "dense_instruction" else r.get(m, 10)
            specs[f"lc_{variant}:{m}"] = (lambda q, ret=ret: [as_chunk(d) for d in ret.invoke(q)], stats)
    return specs


def eval_retrieval(qs: list[dict], only: list[str] | None) -> None:
    qs = [q for q in qs if q["nct_id"]]  # fake trials have no gold document
    specs = retrieval_specs(only)
    for name, (fn, stats) in specs.items():
        if only and name not in only:
            continue
        fn(qs[0]["question"])  # warm-up: first call pays lazy model/index loading
        ranks, ev_ranks, lat = [], [], []
        for q in qs:
            t0 = time.perf_counter()
            chunks = fn(q["question"])
            lat.append((time.perf_counter() - t0) * 1000)
            ids = [c["nct_id"] for c in chunks]
            ranks.append(ids.index(q["nct_id"]) + 1 if q["nct_id"] in ids else None)
            if q["evidence"]:
                ev = [i for i, c in enumerate(chunks, start=1) if has_evidence(c, q)]
                ev_ranks.append(ev[0] if ev else None)
        n, ne = len(ranks), len(ev_ranks)
        out = {
            **{f"evidence@{k}": round(sum(1 for x in ev_ranks if x and x <= k) / ne, 3) for k in (1, 5, 10)},
            "evidence_mrr@10": round(sum(1 / x for x in ev_ranks if x) / ne, 3),
            **{f"trial@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 5)},
            "latency_ms_p50": pct(lat, 0.5),
            "latency_ms_p95": pct(lat, 0.95),
            "n_chunks": stats["n_chunks"],
            "avg_words": stats["avg_words"],
        }
        print(name, out, flush=True)
        merge_results(["retrieval", name], out)
    merge_results(
        ["retrieval", "meta"],
        {"date": datetime.now(UTC).isoformat(timespec="seconds"), "hardware": hardware(), "n_questions": len(qs)},
    )


# ---------------------------------------------------------------- generation
def summarize(records: list[dict]) -> dict:
    a_rec = [x for x in records if x["kind"] != "unanswerable"]
    u_rec = [x for x in records if x["kind"] == "unanswerable"]
    no_leak = [x for x in a_rec if not x["leaky"]]
    out = {
        "n": len(records),
        "accuracy": round(sum(x["correct"] for x in a_rec) / len(a_rec), 3),
        "accuracy_no_leak": round(sum(x["correct"] for x in no_leak) / len(no_leak), 3),
        "grounded": round(sum(x["grounded"] for x in a_rec) / len(a_rec), 3),
        "false_abstain": round(sum(x["abstained"] for x in a_rec) / len(a_rec), 3),
        "abstain_unans": round(sum(x["abstained"] for x in u_rec) / len(u_rec), 3) if u_rec else None,
        "bad_citations": round(sum(bool(x["bad_citations"]) for x in records) / len(records), 3),
        "evidence_in_prompt": round(sum(x["evidence_in_prompt"] for x in a_rec) / len(a_rec), 3),
        "latency_ms_p50": pct([x["ms"] for x in records], 0.5),
        "latency_ms_p95": pct([x["ms"] for x in records], 0.95),
        "ttft_ms_p50": pct([x["ttft_ms"] for x in records if x["ttft_ms"]], 0.5),
        "by_kind": {
            k: round(statistics.mean(x["correct"] for x in a_rec if x["kind"] == k), 3)
            for k in sorted({x["kind"] for x in a_rec})
        },
    }
    graded = [x for x in records if x.get("first_grade") is not None]
    if graded:
        out["llm_calls_mean"] = round(statistics.mean(x["llm_calls"] for x in graded), 2)
        out["rewrite_rate"] = round(sum(x["rewrites"] > 0 for x in graded) / len(graded), 3)
        # the grader is right if it says "yes" exactly when the first retrieval had the gold fact
        ans = [x for x in graded if x["kind"] != "unanswerable"]
        out["grader_accuracy"] = round(sum(x["first_grade"] == x["first_evidence"] for x in ans) / len(ans), 3)
        out["grader_said_no_with_evidence"] = sum(x["first_evidence"] and not x["first_grade"] for x in ans)
        out["rewrite_fixed_evidence"] = sum(
            x["rewrites"] > 0 and not x["first_evidence"] and x["evidence_in_prompt"] for x in ans
        )
    return out


def eval_generation(qs: list[dict], name: str) -> None:
    s = get_settings()
    p = get_pipeline(name, s)
    llm_name = f"{s.llm_provider}:{s.llm_model}"
    RUNS.mkdir(parents=True, exist_ok=True)
    run(p, qs[0]["question"])  # warm-up (lazy loading), not measured
    records = []
    for i, q in enumerate(qs):
        r = run(p, q["question"])
        hits = [Hit(c, 0.0) for c in r.hits]
        a = finalize(r.text, hits)
        ans = q["kind"] != "unanswerable"
        correct = ans and not a.abstained and is_correct(q["kind"], q["gold"], a.text)
        rec = {
            "id": q["id"],
            "kind": q["kind"],
            "leaky": leaky(q),
            "answer": a.text,
            "correct": correct,
            "grounded": correct and q["nct_id"] in a.citations,
            "abstained": a.abstained,
            "bad_citations": a.unsupported_citations,
            "retrieved": [c["nct_id"] for c in r.hits],
            "evidence_in_prompt": any(has_evidence(c, q) for c in r.hits),
            "ms": round(r.ms),
            "ttft_ms": round(r.ttft_ms) if r.ttft_ms else None,
        }
        if r.steps:  # LangGraph: what the loop did
            grades = [st["relevant"] for st in r.steps if st["node"] == "grade"]
            rec |= {
                "first_grade": grades[0],
                "first_evidence": any(has_evidence(c, q) for c in r.first_hits),
                "rewrites": sum(st["node"] == "rewrite" for st in r.steps),
                "rewritten_query": next((st["query"] for st in r.steps if st["node"] == "rewrite"), None),
                "llm_calls": r.steps[-1]["llm_calls"],
            }
        records.append(rec)
        if i % 20 == 0:
            print(f"{name} {i + 1}/{len(qs)} {rec['ms']} ms correct={rec['correct']}", flush=True)
    (RUNS / f"generation_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    out = summarize(records)
    print(name, {k: v for k, v in out.items() if k != "by_kind"}, flush=True)
    merge_results(["generation", name], out)
    merge_results(
        ["generation", "meta", name],
        {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": hardware(),
            "llm": llm_name,
            "k": 4 if name.endswith("default") else 5,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="retrieval specs to run, e.g. lc_tuned:rerank")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--pipeline", choices=PIPELINES)
    ap.add_argument("--limit", type=int, default=None, help="every n-th question only, for a quick smoke run")
    args = ap.parse_args()
    qs = load_questions()
    if args.limit:
        qs = qs[:: max(1, len(qs) // args.limit)]
    if args.retrieval:
        eval_retrieval(qs, args.only)
    if args.generation:
        eval_generation(qs, args.pipeline)


if __name__ == "__main__":
    main()
