"""The RAG chain in LCEL, the pattern from LangChain's "return sources" tutorial.

    {docs: retriever, question: passthrough} → assign(answer = format → prompt → chat model → string)

chain.stream(question) yields {"docs": [...]} first, then {"answer": token} pieces, which is exactly what
the SSE endpoint needs (sources first, then tokens).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from lc.common import PROMPT, format_sources


def rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    to_prompt = RunnableLambda(lambda x: {"sources": format_sources(x["docs"]), "question": x["question"]})
    answer = to_prompt | PROMPT | llm | StrOutputParser()
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer)
