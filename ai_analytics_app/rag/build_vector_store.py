"""
build_vector_store.py — Build a FAISS index from dataset summary text.

Takes the text report from schema_analyzer, chunks it, embeds it,
and returns an in-memory FAISS vector store ready for retrieval.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_embeddings() -> Embeddings:
    """Return an embedding model based on EMBEDDING_PROVIDER env var."""
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Default: local HuggingFace (no API key required)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(summary_text: str) -> FAISS:
    """Chunk the summary text and build an in-memory FAISS vector store.

    Parameters
    ----------
    summary_text : str
        The human-readable dataset analysis report.

    Returns
    -------
    FAISS vector store ready for retrieval.
    """
    # Chunk on section boundaries first, then by size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n======", "\n\n", "\n", " "],
    )

    docs = [Document(page_content=summary_text, metadata={"source": "dataset_analysis"})]
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store
