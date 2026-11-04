"""Corrective RAG as a LangGraph state machine: something the linear step 15 pipeline does not have.

    START → retrieve → grade ─ yes, or already rewritten ─→ generate → END
                         └─ no ─→ rewrite → retrieve (once)

grade    the LLM reads the sources and says yes/no: "do they contain the answer?"
rewrite  if not, the LLM turns the question into a short search query and retrieval runs again
generate the same prompt as every other variant, always with the user's original question

Why this design: it is the standard "grade documents, rewrite on failure" loop from LangGraph's agentic-RAG
tutorials. It costs one extra LLM call per question (two more when it rewrites), so the eval measures
what that buys in accuracy against what it costs in latency.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from lc.common import PROMPT, format_sources

MAX_REWRITES = 1

GRADE = ChatPromptTemplate.from_messages(
    [
        ("system", "You check whether sources contain the answer to a question. Reply with one word: yes or no."),
        ("human", "Sources:\n{sources}\n\nQuestion: {question}\n\nDo the sources contain the answer?"),
    ]
)
REWRITE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question as a short search query for a database of clinical-trial records. "
            "Keep the full trial title and the field that is asked about. Reply with the query only.",
        ),
        ("human", "{question}"),
    ]
)


class State(TypedDict, total=False):
    question: str  # the user's question, never changed
    query: str  # what is sent to the retriever (the question, or its rewrite)
    docs: list[Document]
    relevant: bool
    rewrites: int
    llm_calls: int
    answer: str


def build_graph(retriever: BaseRetriever, llm: BaseChatModel):
    text = StrOutputParser()

    def retrieve(st: State) -> State:
        return {"docs": retriever.invoke(st.get("query") or st["question"])}

    def grade(st: State) -> State:
        verdict = (GRADE | llm | text).invoke({"sources": format_sources(st["docs"]), "question": st["question"]})
        return {"relevant": verdict.strip().lower().startswith("yes"), "llm_calls": st.get("llm_calls", 0) + 1}

    def rewrite(st: State) -> State:
        query = (REWRITE | llm | text).invoke({"question": st["question"]}).strip()
        return {"query": query, "rewrites": st.get("rewrites", 0) + 1, "llm_calls": st["llm_calls"] + 1}

    def generate(st: State) -> State:
        answer = (PROMPT | llm | text).invoke({"sources": format_sources(st["docs"]), "question": st["question"]})
        return {"answer": answer, "llm_calls": st["llm_calls"] + 1}

    def route(st: State) -> str:
        return "generate" if st["relevant"] or st.get("rewrites", 0) >= MAX_REWRITES else "rewrite"

    g = StateGraph(State)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite), ("generate", generate)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()
