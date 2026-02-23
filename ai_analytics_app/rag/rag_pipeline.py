"""
rag_pipeline.py — Retrieval-based QA chain over dataset summaries.

Given a FAISS vector store, builds a LangChain RAG chain that
answers questions grounded in the dataset analysis report.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert data analyst assistant.
Use ONLY the retrieved context below to answer the user's question
about this dataset.

Rules:
- Be precise with numbers and percentages — cite them from context.
- If the context doesn't contain the answer, say so honestly.
- Structure your answer clearly.
- Keep answers concise but complete.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\n---\nQuestion: {question}"),
])


# ── LLM factory ──────────────────────────────────────────────────────────────


def _get_llm() -> BaseChatModel:
    """Instantiate LLM based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=2048,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=2048,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ── Chain ────────────────────────────────────────────────────────────────────


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_rag_chain(vector_store: FAISS):
    """Return a runnable RAG chain: question → retrieval → LLM → answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = _get_llm()

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
