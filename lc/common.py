"""Pieces every LangChain variant shares: documents, the prompt and the source formatting.

The prompt text is imported from the scratch pipeline (rag/answer.py) on purpose. The prompt is the task
specification (answer only from sources, cite [NCT id], say "I don't know"), not a framework feature, so it is
held constant. Any difference in the eval then comes from chunking, retrieval or orchestration.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from rag.answer import SYSTEM_RAG

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_RAG), ("human", "Sources:\n{sources}\n\nQuestion: {question}")]
)


def format_sources(docs: list[Document]) -> str:
    """Same layout as rag.answer.build_messages: the trial id comes from metadata, so every chunk is citable."""
    return "\n\n".join(f"Source {i} [{d.metadata['nct_id']}]: {d.page_content}" for i, d in enumerate(docs, start=1))


def record_document(doc: dict) -> Document:
    """One whole trial record as one Document: what a loader gives you before any splitting."""
    return Document(
        page_content="\n\n".join(s["text"] for s in doc["sections"]),
        metadata={"nct_id": doc["nct_id"], "title": doc["title"]},
    )


def as_chunk(d: Document) -> dict:
    """Document → the chunk dict the shared eval and the API expect (same keys as rag.chunking.Chunk)."""
    m = d.metadata
    return {
        "chunk_id": m.get("chunk_id", ""),
        "nct_id": m["nct_id"],
        "title": m.get("title", ""),
        "section": m.get("section", ""),
        "text": d.page_content,
        "scores": {},  # LangChain retrievers return bare Documents: per-stage scores would need custom code
    }
