"""Advanced RAG answer pipeline.

Implements the public retrieval and answer generation pipeline for the repo.
Main techniques:
1. Query rewriting
2. Multi-query retrieval
3. LLM re-ranking
4. Streaming answer generation
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is required. Add it to your .env file.")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
embedding_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "openai/gpt-4.1-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "openai/gpt-4.1-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "openai/gpt-4.1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs_advanced"

RETRIEVAL_K = 20
FINAL_K = 10

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ordered from most relevant to least relevant"
    )


SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant representing the fictional company Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so answer only with information supported by the provided context.
If the context does not contain the answer, say so clearly.

Context:
{context}

Answer the user's question accurately, directly, and completely."""


def call_llm(messages: list[dict], model: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def call_llm_stream(messages: list[dict], model: str, temperature: float = 0.7):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    prompt = f"""You are in a conversation with a user about a fictional company called Insurellm.
You are about to search a knowledge base.

Conversation history:
{history or []}

Current user question:
{question}

Respond only with a short, refined search query that is most likely to surface useful knowledge base content.
Return only the query text."""

    rewritten = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=REWRITE_MODEL,
        temperature=0.3,
    ).strip()
    print(f"  [Query Rewrite] '{question}' -> '{rewritten}'")
    return rewritten


def fetch_context_unranked(question: str) -> list[Result]:
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME, input=[question]
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding], n_results=RETRIEVAL_K
    )

    chunks: list[Result] = []
    if results["documents"] and results["metadatas"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(Result(page_content=doc, metadata=dict(meta)))
    return chunks


def merge_chunks(chunks1: list[Result], chunks2: list[Result]) -> list[Result]:
    merged = chunks1[:]
    existing_contents = [chunk.page_content for chunk in chunks1]

    for chunk in chunks2:
        if chunk.page_content not in existing_contents:
            merged.append(chunk)

    print(f"  [Merge] {len(chunks1)} + {len(chunks2)} -> {len(merged)} unique chunks")
    return merged


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    system_prompt = """You are a document re-ranker.
You are given a question and a list of text chunks from a knowledge base.
The chunks are already approximately ordered by relevance, but you may improve that ordering.
Rank all chunks from most relevant to least relevant.
Reply only with valid JSON: {"order": [1, 3, 2, ...]}"""

    user_prompt = (
        f"The user asked:\n\n{question}\n\n"
        "Order all provided chunks from most relevant to least relevant.\n\n"
        "Chunks:\n\n"
    )
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"

    user_prompt += 'Reply only with valid JSON: {"order": [...]}.'

    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=RERANK_MODEL,
            temperature=0.1,
        )

        json_str = response
        if "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]

        order = json.loads(json_str)["order"]
        reranked: list[Result] = []
        for idx in order:
            if 1 <= idx <= len(chunks):
                reranked.append(chunks[idx - 1])

        for index, chunk in enumerate(chunks):
            if (index + 1) not in order:
                reranked.append(chunk)

        print(f"  [Re-rank] Reordered {len(reranked)} chunks")
        return reranked
    except Exception as exc:
        print(f"  [Re-rank] Failed: {exc}, using original order")
        return chunks


def fetch_context(original_question: str) -> list[Result]:
    pipeline_start = time.time()
    print("\n--- Advanced Retrieval Pipeline ---")

    rewritten_question = rewrite_query(original_question)
    chunks_original = fetch_context_unranked(original_question)
    chunks_rewritten = fetch_context_unranked(rewritten_question)
    merged_chunks = merge_chunks(chunks_original, chunks_rewritten)
    reranked_chunks = rerank(original_question, merged_chunks)

    final_chunks = reranked_chunks[:FINAL_K]
    print(
        f"  [Final] Using top {len(final_chunks)} chunks ({time.time() - pipeline_start:.1f}s)"
    )
    print("--- Pipeline Complete ---\n")
    return final_chunks


def make_rag_messages(
    question: str, history: list[dict], chunks: list[Result]
) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def answer_question(
    question: str, history: list[dict] | None = None
) -> tuple[str, list[Result]]:
    history = history or []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    answer = call_llm(messages=messages, model=ANSWER_MODEL, temperature=0.7)
    return answer, chunks


def answer_question_stream(question: str, history: list[dict] | None = None):
    history = history or []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)

    partial_answer = ""
    for token in call_llm_stream(
        messages=messages, model=ANSWER_MODEL, temperature=0.7
    ):
        partial_answer += token
        yield partial_answer, chunks


if __name__ == "__main__":
    test_question = "What insurance products does Insurellm offer?"
    answer, used_chunks = answer_question(test_question)
    print(answer)
    print(f"\nUsed {len(used_chunks)} chunks")
