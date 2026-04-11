"""
Advanced RAG Answer Module
==========================
Implements the retrieval and answer generation pipeline.

Advanced techniques:
1. Query rewriting
2. Multi-query retrieval
3. LLM re-ranking
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json


load_dotenv(override=True)

# ============ CONFIGURATION ============
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o")

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs_advanced"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 20
FINAL_K = 10

print("Loading HuggingFace embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Embedding model loaded: {EMBEDDING_MODEL_NAME}")

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)
print(f"ChromaDB loaded: {collection.count()} documents in collection")


# ============ DATA MODELS ============
class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ordered from most relevant to least relevant"
    )


# ============ SYSTEM PROMPT ============
SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant representing a fictional company used in a portfolio demo.
Your answer will be evaluated for accuracy, relevance and completeness, so answer only with information supported by context.
If you do not know the answer, say so.

For context, here are specific extracts from the knowledge base:

{context}

With this context, please answer the user's question. Be accurate, relevant and complete."""


def call_llm(messages: list[dict], model: str, temperature: float = 0.7) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"API Error: {e}")
        raise


# ============ ADVANCED RAG TECHNIQUES ============
def rewrite_query(question: str, history: list = []) -> str:
    prompt = f"""You are in a conversation with a user about a fictional company.
You are about to search a knowledge base.

Conversation history:
{history}

Current user question:
{question}

Respond only with a short, refined search query that is most likely to surface useful knowledge base content.
Return only the query text."""

    response = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=REWRITE_MODEL,
        temperature=0.3,
    )

    rewritten = response.strip()
    print(f"  [Query Rewrite] '{question}' -> '{rewritten}'")
    return rewritten


def fetch_context_unranked(question: str) -> list[Result]:
    query_embedding = embedding_model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K,
    )

    chunks = []
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
Rank all chunks from most relevant to least relevant.
Reply only with valid JSON: {"order": [1, 3, 2, ...]}"""

    user_prompt = f"Question: {question}\n\nRank these chunks by relevance:\n\n"
    for index, chunk in enumerate(chunks):
        content = (
            chunk.page_content[:400] + "..."
            if len(chunk.page_content) > 400
            else chunk.page_content
        )
        user_prompt += f"CHUNK {index + 1}:\n{content}\n\n"

    user_prompt += 'Reply with JSON only: {"order": [...]}'

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

        reranked = []
        for idx in order:
            if 1 <= idx <= len(chunks):
                reranked.append(chunks[idx - 1])

        for i, chunk in enumerate(chunks):
            if (i + 1) not in order:
                reranked.append(chunk)

        print(f"  [Re-rank] Reordered {len(reranked)} chunks")
        return reranked
    except Exception as e:
        print(f"  [Re-rank] Failed: {e}, using original order")
        return chunks


def fetch_context(original_question: str) -> list[Result]:
    print("\n--- Advanced Retrieval Pipeline ---")

    rewritten_question = rewrite_query(original_question)

    print("  [Retrieve] Fetching chunks...")
    chunks_original = fetch_context_unranked(original_question)
    chunks_rewritten = fetch_context_unranked(rewritten_question)

    merged_chunks = merge_chunks(chunks_original, chunks_rewritten)
    reranked_chunks = rerank(original_question, merged_chunks)

    final_chunks = reranked_chunks[:FINAL_K]
    print(f"  [Final] Using top {len(final_chunks)} chunks")
    print("--- Pipeline Complete ---\n")
    return final_chunks


# ============ ANSWER GENERATION ============
def make_rag_messages(
    question: str, history: list[dict], chunks: list[Result]
) -> list[dict]:
    context_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(f"Extract from {source}:\n{chunk.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def answer_question(
    question: str, history: list[dict] = []
) -> tuple[str, list[Result]]:
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)

    print(f"[Generating answer with {ANSWER_MODEL}...]")
    answer = call_llm(messages=messages, model=ANSWER_MODEL, temperature=0.7)
    return answer, chunks


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED RAG ANSWER")
    print("=" * 60)
    print(f"Rewrite Model: {REWRITE_MODEL}")
    print(f"Rerank Model: {RERANK_MODEL}")
    print(f"Answer Model: {ANSWER_MODEL}")
    print("=" * 60)

    test_q = "What products does the company offer?"
    print(f"\nTesting: {test_q}\n")

    answer, chunks = answer_question(test_q)

    print(f"\nANSWER:\n{answer}")
    print(f"\n[Used {len(chunks)} context chunks]")
