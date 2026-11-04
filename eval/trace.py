"""Trace the LangChain pipelines with a self-hosted Arize Phoenix (open source, runs locally, 0 €).

Why: "what did the pipeline actually do, and where did the time go" is the question you ask on every bad
answer. LangChain components emit callbacks, OpenInference turns them into OpenTelemetry spans, Phoenix stores
and shows them. Step 15 has no such layer: its per-stage scores in the UI were written by hand.

The script starts Phoenix (unless one already runs on :6006), sends a few questions through lc_tuned and
lg_tuned, then reads the spans back and sums the time per span name, so the README can say where time goes.
(The plan was Langfuse in Docker; its images could not be pulled from this cloud machine, see README.)

Run:  uv run --group trace python eval/trace.py [--n 8]
      then open http://localhost:6006 (the server keeps running in the background)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import hardware, load_questions, merge_results  # noqa: E402
from rag.config import ROOT, get_settings  # noqa: E402

PHOENIX = "http://localhost:6006"
PROJECT = "step21"


def ensure_phoenix() -> None:
    try:
        httpx.get(f"{PHOENIX}/healthz", timeout=2).raise_for_status()
        return
    except httpx.HTTPError:
        pass
    env = {**os.environ, "PHOENIX_WORKING_DIR": str(ROOT / "data" / "phoenix")}
    log = (ROOT / "data" / "phoenix.log").open("w")
    subprocess.Popen([sys.executable, "-m", "phoenix.server.main", "serve"], env=env, stdout=log, stderr=log)
    for _ in range(120):
        time.sleep(1)
        try:
            httpx.get(f"{PHOENIX}/healthz", timeout=2).raise_for_status()
            return
        except httpx.HTTPError:
            continue
    raise RuntimeError("Phoenix did not start, see data/phoenix.log")


def main() -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.client import Client
    from phoenix.otel import register

    from lc.llm import build_chat_model
    from pipelines import get_pipeline, run

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="questions per pipeline")
    args = ap.parse_args()

    ensure_phoenix()
    tp = register(endpoint=f"{PHOENIX}/v1/traces", project_name=PROJECT, batch=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=tp)

    s = get_settings()
    llm, retrievers = build_chat_model(s), {}
    qs = load_questions()
    picked = qs[:: len(qs) // args.n][: args.n]  # every k-th question: all question types, incl. unanswerable
    for name in ("lc_tuned", "lg_tuned"):
        p = get_pipeline(name, s, llm=llm, retrievers=retrievers)
        for q in picked:
            r = run(p, q["question"])
            print(name, q["kind"], round(r.ms), r.text[:80].replace("\n", " "), flush=True)
    tp.force_flush()
    time.sleep(2)

    df = Client(base_url=PHOENIX).spans.get_spans_dataframe(project_identifier=PROJECT, limit=100000)
    df["ms"] = (df["end_time"] - df["start_time"]).dt.total_seconds() * 1000
    roots = df[df["parent_id"].isna()]
    total = roots["ms"].sum()
    by_name = (
        df.groupby("name")["ms"].agg(["count", "sum", "median"]).sort_values("sum", ascending=False).round(1).head(15)
    )
    out = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
        "questions_per_pipeline": len(picked),
        "traces": int(len(roots)),
        "spans": int(len(df)),
        "spans_per_trace": round(len(df) / len(roots), 1),
        "root_ms_total": round(total, 1),
        "by_span": {
            n: {
                "count": int(r["count"]),
                "ms_total": r["sum"],
                "ms_median": r["median"],
                "share": round(r["sum"] / total, 3),
            }
            for n, r in by_name.iterrows()
        },
    }
    print(json.dumps(out, indent=1))
    merge_results(["trace"], out)


if __name__ == "__main__":
    main()
