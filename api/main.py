"""FastAPI backend from step 15, extended so the same question can be answered by any of the pipelines.

The UI asks two pipelines at once and shows them side by side (the step's visual example). Every pipeline
streams the same Server-Sent Events: sources → token… → done, and LangGraph adds "step" events (grade verdict,
rewritten query) so you can watch the loop decide.

Pipelines are built lazily on first use and share one LangChain chat model and one set of retrievers,
so the cloud machine holds at most two copies of the LLM (scratch + LangChain).

Run:  uv run uvicorn api.main:app --port 8000
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipelines import PIPELINES, get_pipeline
from rag.answer import finalize
from rag.config import get_settings
from rag.index import load_docs
from rag.retrieve import Hit

PipelineName = Literal["scratch", "lc_default", "lc_tuned", "lg_tuned", "lg_default"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    pipeline: PipelineName = "scratch"


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipelines: dict | None = None, docs: list[dict] | None = None) -> FastAPI:
    """Factory, so tests can inject small pipelines with fake models instead of loading real ones."""
    s = get_settings()
    lock = threading.Lock()  # two lanes asking at once must not build the same pipeline twice

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipelines = dict(pipelines or {})
        app.state.shared = {"retrievers": {}, "llm": None}
        app.state.docs = {d["nct_id"]: d for d in (docs if docs is not None else load_docs(s))}
        yield

    def pipeline(app: FastAPI, name: str):
        with lock:
            if name not in app.state.pipelines:
                sh = app.state.shared
                if name != "scratch" and sh["llm"] is None:
                    from lc.llm import build_chat_model

                    sh["llm"] = build_chat_model(s)
                app.state.pipelines[name] = get_pipeline(name, s, llm=sh["llm"], retrievers=sh["retrievers"])
            return app.state.pipelines[name]

    app = FastAPI(title="Clinical trials RAG: scratch vs LangChain", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.get("/api/health")
    def health(request: Request) -> dict:
        return {
            "status": "ok",
            "llm": f"{s.llm_provider}:{s.llm_model}",
            "n_studies": len(request.app.state.docs),
            "pipelines": list(PIPELINES),
            "loaded": sorted(request.app.state.pipelines),
        }

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request) -> StreamingResponse:
        def events() -> Iterator[str]:
            t0 = time.perf_counter()
            hits: list[dict] = []
            parts: list[str] = []
            first_ms = None
            try:
                p = pipeline(request.app, req.pipeline)
                for kind, data in p.stream(req.question):
                    if kind == "sources":
                        hits = data
                        ms = round((time.perf_counter() - t0) * 1000, 1)
                        yield sse("sources", {"hits": data, "retrieval_ms": ms})
                    elif kind == "token":
                        first_ms = first_ms or (time.perf_counter() - t0) * 1000
                        parts.append(data)
                        yield sse("token", {"text": data})
                    elif kind == "step":
                        yield sse("step", data)
            except Exception as e:  # e.g. a model that failed to load: tell the UI instead of cutting the stream
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})
                return
            a = finalize("".join(parts), [Hit(h, 0.0) for h in hits])
            yield sse(
                "done",
                {
                    "citations": a.citations,
                    "abstained": a.abstained,
                    "unsupported_citations": a.unsupported_citations,
                    "first_token_ms": round(first_ms or 0),
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                },
            )

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/studies/{nct_id}")
    def study(nct_id: str, request: Request) -> dict:
        if (d := request.app.state.docs.get(nct_id)) is None:
            raise HTTPException(404, f"{nct_id} is not in the corpus")
        return {
            "nct_id": d["nct_id"],
            "title": d["title"],
            "sections": d["sections"],
            "url": f"https://clinicaltrials.gov/study/{d['nct_id']}",
        }

    return app


app = create_app()
