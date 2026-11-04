"""Fast tests for the LangChain side: no model downloads, fake retriever and fake chat model."""

from pathlib import Path

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from lc.common import as_chunk, format_sources
from lc.index import default_chunks, tuned_chunks
from lc.llm import FakeChatModel, no_thinking_template
from lc.retrievers import TopN
from pipelines import GraphPipeline, LCPipeline, run
from rag.answer import build_messages
from rag.chunking import chunk_section
from rag.ingest import render_study
from rag.retrieve import Hit
from tests.test_rag import STUDY

DOC = render_study(STUDY)


class ListRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.docs


def test_tuned_chunks_match_step_15_layout():
    lc, scratch = tuned_chunks([DOC]), chunk_section(DOC)
    assert all(c.page_content.startswith("[NCT00000001] Metformin in Prediabetes |") for c in lc)
    assert {c.metadata["section"] for c in lc} == {c.section for c in scratch}
    assert len({c.metadata["chunk_id"] for c in lc}) == len(lc)


def test_default_chunks_are_character_windows_without_header():
    chunks = default_chunks([DOC])
    assert all(len(c.page_content) <= 1000 for c in chunks)
    assert not chunks[0].page_content.startswith("[NCT")


def test_sources_look_the_same_as_in_the_scratch_prompt():
    d = Document("Enrollment: 240", metadata={"nct_id": "NCT00000001"})
    scratch = build_messages("How many?", [Hit({"nct_id": "NCT00000001", "text": "Enrollment: 240"}, 1.0)])
    assert format_sources([d]) in scratch[-1]["content"]


def test_top_n_cuts_the_list():
    docs = [Document(str(i), metadata={"nct_id": f"NCT0000000{i}"}) for i in range(5)]
    assert len(TopN(retriever=ListRetriever(docs=docs), n=2).invoke("q")) == 2


def test_lcel_chain_streams_sources_then_tokens_and_cites():
    docs = [Document("[NCT00000001] T | Design\nPhase: Phase 2", metadata={"nct_id": "NCT00000001"})]
    r = run(LCPipeline("t", ListRetriever(docs=docs), FakeChatModel()), "What phase?")
    assert as_chunk(docs[0]) == r.hits[0] and "[NCT00000001]" in r.text and r.ttft_ms is not None


@pytest.mark.parametrize(("verdicts", "rewrites"), [(["yes"], 0), (["no", "no"], 1)])
def test_graph_rewrites_at_most_once(verdicts, rewrites):
    docs = [Document("Phase: Phase 2", metadata={"nct_id": "NCT00000001"})]
    # replies in call order: grade, (rewrite, grade), generate
    replies = ["yes", "answer [NCT00000001]"] if rewrites == 0 else ["no", "better query", "no", "I don't know."]
    p = GraphPipeline("t", ListRetriever(docs=docs), FakeListChatModel(responses=replies))
    r = run(p, "What phase?")
    assert sum(s["node"] == "rewrite" for s in r.steps) == rewrites
    assert r.text == replies[-1]
    assert r.steps[-1]["llm_calls"] == len(replies)


QWEN = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B"


@pytest.mark.skipif(not QWEN.exists(), reason="Qwen3 tokenizer not downloaded")
def test_patched_template_gives_the_scratch_prompt():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    scratch = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tok.chat_template = no_thinking_template(tok.chat_template)
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == scratch
