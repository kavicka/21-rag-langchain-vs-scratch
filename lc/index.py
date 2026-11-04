"""Chunk the corpus and build a Qdrant collection with LangChain components, for two variants.

default  what the LangChain RAG tutorial does: load each record, RecursiveCharacterTextSplitter(1000 chars,
         200 overlap), HuggingFaceEmbeddings with its defaults, retriever with its default k=4.
tuned    LangChain components configured to match step 15: split along the record's sections
         (MarkdownHeaderTextSplitter), ~220-word windows with 40 words overlap, a "[NCT id] title | section"
         header on every chunk, normalized bge vectors with bge's query instruction.

Run:  uv run python -m lc.index            # both variants
      uv run python -m lc.index --variant tuned
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from lc.common import record_document
from rag.config import Settings, get_settings
from rag.index import load_docs

Variant = Literal["default", "tuned"]
VARIANTS: tuple[Variant, ...] = ("default", "tuned")
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def word_count(text: str) -> int:
    return len(text.split())


def default_chunks(docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = splitter.split_documents([record_document(d) for d in docs])
    for c in chunks:
        c.metadata["chunk_id"] = f"{c.metadata['nct_id']}@{c.metadata['start_index']}"
    return chunks


def tuned_chunks(docs: list[dict]) -> list[Document]:
    by_section = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    # length_function counts words, so chunk_size/overlap mean the same as MAX_WORDS/OVERLAP in rag/chunking.py
    windows = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40, length_function=word_count)
    out: list[Document] = []
    for d in docs:
        markdown = "\n\n".join(f"## {s['name']}\n{s['text']}" for s in d["sections"])
        sections = by_section.split_text(markdown)
        seen: dict[str, int] = {}
        for sec in windows.split_documents(sections):
            name = sec.metadata["section"]
            i = seen[name] = seen.get(name, -1) + 1
            out.append(
                Document(
                    page_content=f"[{d['nct_id']}] {d['title']} | {name}\n{sec.page_content}",
                    metadata={
                        "nct_id": d["nct_id"],
                        "title": d["title"],
                        "section": name,
                        "chunk_id": f"{d['nct_id']}#{name.lower().replace(' ', '-')}-{i}",
                    },
                )
            )
    return out


CHUNKERS = {"default": default_chunks, "tuned": tuned_chunks}


def embeddings(variant: Variant, s: Settings) -> HuggingFaceEmbeddings:
    if variant == "default":
        return HuggingFaceEmbeddings(model_name=s.embed_model)
    return HuggingFaceEmbeddings(
        model_name=s.embed_model,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": BGE_QUERY},
    )


def variant_dir(s: Settings, variant: Variant) -> Path:
    return s.data_dir / "lc" / variant


def load_chunks(s: Settings, variant: Variant) -> list[Document]:
    with (variant_dir(s, variant) / "chunks.jsonl").open() as f:
        return [Document(**json.loads(line)) for line in f]


def open_store(s: Settings, variant: Variant) -> QdrantVectorStore:
    client = QdrantClient(path=str(variant_dir(s, variant) / "qdrant"))
    return QdrantVectorStore(client=client, collection_name=f"lc_{variant}", embedding=embeddings(variant, s))


def build(variant: Variant, s: Settings | None = None) -> dict:
    s = s or get_settings()
    chunks = CHUNKERS[variant](load_docs(s))
    out = variant_dir(s, variant)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    t0 = time.perf_counter()
    store = QdrantVectorStore.from_documents(
        chunks,
        embeddings(variant, s),
        path=str(out / "qdrant"),
        collection_name=f"lc_{variant}",
        force_recreate=True,
    )
    store.client.close()
    stats = {
        "variant": variant,
        "n_chunks": len(chunks),
        "avg_words": round(sum(word_count(c.page_content) for c in chunks) / len(chunks), 1),
        "embed_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default=None, help="default: both")
    args = ap.parse_args()
    for v in [args.variant] if args.variant else VARIANTS:
        build(v)


if __name__ == "__main__":
    main()
