"""One interface in front of every pipeline, so the eval and the API treat scratch and LangChain the same way.

stream(question) yields events in the order the UI needs them:
    ("sources", [chunk dicts])   the retrieved chunks (LangGraph can send this twice: after a rewrite)
    ("token", "text")            answer tokens
    ("step", {...})              LangGraph only: grade verdict, rewritten query

Pipelines:
    scratch     step 15 unchanged: section chunks, own BM25 + Qdrant, RRF, cross-encoder rerank, own LLM layer
    lc_default  LangChain tutorial defaults (recursive 1000-char chunks, dense only, k=4)
    lc_tuned    LangChain parts configured like step 15 (section chunks, BM25 + dense RRF, rerank, k=5)
    lg_tuned    LangGraph corrective loop (grade → rewrite → retrieve again) on top of lc_tuned's retriever
    lg_default  the same loop on top of lc_default's retriever: can an agent loop make up for weak retrieval?
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rag.config import Settings, get_settings

PIPELINES = ("scratch", "lc_default", "lc_tuned", "lg_tuned", "lg_default")
K = 5  # chunks in the prompt (step 15); lc_default keeps LangChain's default of 4

Event = tuple[str, Any]


@dataclass
class Run:
    text: str = ""
    hits: list[dict] = field(default_factory=list)  # the chunks the answer was generated from
    first_hits: list[dict] = field(default_factory=list)  # LangGraph: the chunks before any rewrite
    ms: float = 0.0
    ttft_ms: float | None = None
    steps: list[dict] = field(default_factory=list)


def run(p, question: str) -> Run:
    """Consume a pipeline's stream and time it: total latency and time to the first answer token."""
    r, t0 = Run(), time.perf_counter()
    parts: list[str] = []
    for kind, data in p.stream(question):
        if kind == "sources":
            r.first_hits = r.first_hits or data
            r.hits = data
        elif kind == "token":
            if r.ttft_ms is None:
                r.ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(data)
        elif kind == "step":
            r.steps.append(data)
    r.text, r.ms = "".join(parts), (time.perf_counter() - t0) * 1000
    return r


class ScratchPipeline:
    def __init__(self, s: Settings, mode: str = "rerank", k: int = K, retriever=None, llm=None):
        from rag.llm import get_llm
        from rag.retrieve import Retriever

        self.name, self.mode, self.k = "scratch", mode, k
        self.retriever, self.llm = retriever or Retriever("section", s), llm or get_llm(s)

    def stream(self, question: str) -> Iterator[Event]:
        from rag.answer import stream_answer

        hits = self.retriever.search(question, self.mode, self.k)
        yield "sources", [h.to_dict() for h in hits]
        for tok in stream_answer(self.llm, question, hits):
            yield "token", tok


class LCPipeline:
    def __init__(self, name: str, retriever, llm):
        from lc.chain import rag_chain

        self.name, self.chain = name, rag_chain(retriever, llm)

    def stream(self, question: str) -> Iterator[Event]:
        from lc.common import as_chunk

        for part in self.chain.stream(question):
            if "docs" in part:
                yield "sources", [as_chunk(d) for d in part["docs"]]
            if part.get("answer"):
                yield "token", part["answer"]


class GraphPipeline:
    def __init__(self, name: str, retriever, llm):
        from lc.graph import build_graph

        self.name, self.graph = name, build_graph(retriever, llm)

    def stream(self, question: str) -> Iterator[Event]:
        from lc.common import as_chunk

        for mode, data in self.graph.stream({"question": question}, stream_mode=["updates", "messages"]):
            if mode == "messages":
                msg, meta = data
                if meta.get("langgraph_node") == "generate" and msg.content:
                    yield "token", msg.content
                continue
            for node, update in data.items():
                if node == "retrieve":
                    yield "sources", [as_chunk(d) for d in update["docs"]]
                elif node == "grade":
                    yield "step", {"node": "grade", "relevant": update["relevant"]}
                elif node == "rewrite":
                    yield "step", {"node": "rewrite", "query": update["query"]}
                elif node == "generate":
                    yield "step", {"node": "generate", "llm_calls": update["llm_calls"]}


def get_pipeline(name: str, s: Settings | None = None, llm=None, retrievers: dict | None = None):
    """retrievers/llm can be passed in so several pipelines in one process share models (the API does that)."""
    s = s or get_settings()
    if name == "scratch":
        return ScratchPipeline(s)

    from lc.llm import build_chat_model
    from lc.retrievers import LCRetrievers

    llm = llm or build_chat_model(s)
    retrievers = retrievers if retrievers is not None else {}
    variant = "default" if name.endswith("default") else "tuned"
    if variant not in retrievers:
        retrievers[variant] = LCRetrievers(variant, s)
    r = retrievers[variant]
    # default: as_retriever() with no arguments, i.e. LangChain's own k=4
    retriever = r.store.as_retriever() if variant == "default" else r.get("rerank", K)
    cls = GraphPipeline if name.startswith("lg_") else LCPipeline
    return cls(name, retriever, llm)
