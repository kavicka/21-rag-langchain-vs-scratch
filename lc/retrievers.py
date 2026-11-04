"""Retrievers built only from LangChain parts, one per mode, so the eval can line them up against rag/retrieve.py.

Mapping to step 15:
  bm25      BM25Retriever (rank_bm25.BM25Okapi) with step 15's tokenizer
  bm25_ws   the same with BM25Retriever's default tokenizer, text.split(): no lower-casing, punctuation kept
  dense     QdrantVectorStore.as_retriever()
  hybrid    EnsembleRetriever: weighted Reciprocal Rank Fusion, c=60, equal weights = plain RRF
  rerank    top 20 of the fusion → CrossEncoderReranker (same cross-encoder as step 15)
  rerank_all  the "obvious" LangChain wiring: ContextualCompressionRetriever over the whole fused list,
            so the cross-encoder scores every candidate both retrievers returned (up to 100), not 20
"""

from functools import cached_property

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from lc.index import Variant, load_chunks, open_store
from rag.config import Settings, get_settings
from rag.retrieve import tokenize

POOL = 50  # candidates per retriever before fusion, as in step 15
RERANK_POOL = 20
TUNED_MODES = ("bm25", "bm25_ws", "dense", "hybrid", "rerank", "rerank_all")


class TopN(BaseRetriever):
    """Cut another retriever's list to n. EnsembleRetriever has no such option; step 15 reranks only the top 20."""

    retriever: BaseRetriever
    n: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.retriever.invoke(query, config={"callbacks": run_manager.get_child()})[: self.n]


class LCRetrievers:
    """Builds every mode for one index variant; heavy parts (models, BM25 index) are created once and shared."""

    def __init__(self, variant: Variant, s: Settings | None = None):
        self.s = s or get_settings()
        self.variant = variant
        self.store = open_store(self.s, variant)

    @cached_property
    def chunks(self) -> list[Document]:
        return load_chunks(self.s, self.variant)

    @cached_property
    def _bm25(self) -> BM25Retriever:
        return BM25Retriever.from_documents(self.chunks, k=POOL, preprocess_func=tokenize)

    @cached_property
    def _bm25_ws(self) -> BM25Retriever:
        return BM25Retriever.from_documents(self.chunks, k=POOL)

    @cached_property
    def _cross_encoder(self) -> HuggingFaceCrossEncoder:
        return HuggingFaceCrossEncoder(model_name=self.s.rerank_model)

    def dense(self, k: int) -> BaseRetriever:
        return self.store.as_retriever(search_kwargs={"k": k})

    def dense_with_instruction(self, k: int) -> BaseRetriever:
        """Ablation for the default index: same chunks, but queries get bge's instruction prefix."""
        from langchain_qdrant import QdrantVectorStore

        from lc.index import embeddings

        store = QdrantVectorStore(
            client=self.store.client, collection_name=self.store.collection_name, embedding=embeddings("tuned", self.s)
        )
        return store.as_retriever(search_kwargs={"k": k})

    def ensemble(self) -> EnsembleRetriever:
        return EnsembleRetriever(retrievers=[self._bm25, self.dense(POOL)], weights=[0.5, 0.5], id_key="chunk_id")

    def get(self, mode: str, k: int) -> BaseRetriever:
        match mode:
            case "bm25":
                return TopN(retriever=self._bm25, n=k)
            case "bm25_ws":
                return TopN(retriever=self._bm25_ws, n=k)
            case "dense":
                return self.dense(k)
            case "hybrid":
                return TopN(retriever=self.ensemble(), n=k)
            case "rerank" | "rerank_all":
                base = self.ensemble() if mode == "rerank_all" else TopN(retriever=self.ensemble(), n=RERANK_POOL)
                return ContextualCompressionRetriever(
                    base_compressor=CrossEncoderReranker(model=self._cross_encoder, top_n=k), base_retriever=base
                )
        raise ValueError(mode)
