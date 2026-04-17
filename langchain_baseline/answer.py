"""LangChain baseline answer pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "langchain_vectordb")
COLLECTION_NAME = "langchain_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LANGCHAIN_LLM_MODEL", "openai/gpt-4.1-mini")

_embeddings = None
_vectorstore = None
_lock = Lock()

SYSTEM_PROMPT_TEMPLATE = """You are a knowledgeable, friendly assistant representing the fictional company Insurellm.

Use the following context to answer the question. If you do not know the answer, say so.

Context:
{context}
"""


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        with _lock:
            if _vectorstore is None:
                _vectorstore = Chroma(
                    persist_directory=DB_NAME,
                    embedding_function=embeddings,
                    collection_name=COLLECTION_NAME,
                )
    return _vectorstore


def get_llm() -> ChatOpenAI:
    if OPENAI_API_KEY:
        return ChatOpenAI(
            model=LLM_MODEL, temperature=0.7, api_key=SecretStr(OPENAI_API_KEY)
        )
    if OPENROUTER_API_KEY:
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.7,
            api_key=SecretStr(OPENROUTER_API_KEY),
            base_url=OPENROUTER_BASE_URL,
        )
    raise ValueError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file.")


def fetch_context(question: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVAL_K}
    )
    return retriever.invoke(question)


def answer_question(question: str, history: list[dict[str, str]] | None = None):
    history = history or []
    docs = fetch_context(question)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context))
    ]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    response = get_llm().invoke(messages)
    return str(response.content), docs


if __name__ == "__main__":
    answer, docs = answer_question("What insurance products does Insurellm offer?")
    print(answer)
    print(f"\nRetrieved {len(docs)} documents")
