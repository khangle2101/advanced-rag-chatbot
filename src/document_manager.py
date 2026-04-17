"""Document manager for incremental ingestion.

Supports adding or deleting documents from the Advanced RAG knowledge base
without rebuilding the entire vector database.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
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

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs_advanced"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "openai/gpt-4.1")

AVERAGE_CHUNK_SIZE = 100

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
embedding_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


class Chunk(BaseModel):
    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="2-3 sentences summarizing the chunk")
    original_text: str = Field(description="The exact original text")


class Chunks(BaseModel):
    chunks: list[Chunk]


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHUNKING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def make_chunking_prompt(text: str, filename: str, doc_type: str = "uploaded") -> str:
    how_many = max(1, len(text) // AVERAGE_CHUNK_SIZE)
    return f"""You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a fictional company called Insurellm.
Document type: {doc_type}
Document source: {filename}

A chatbot will use these chunks to answer questions about the company.

INSTRUCTIONS:
1. Split into approximately {how_many} chunks (can be more or less as appropriate)
2. Each chunk should be self-contained enough to answer specific questions
3. Include about 25% overlap between chunks for context continuity
4. Do not leave anything out; the full document must be represented

For each chunk provide:
- headline: brief heading likely to match search queries
- summary: 2-3 sentences summarizing the chunk content
- original_text: the exact original text without modification

DOCUMENT:

{text}

Respond with valid JSON in this format:
{{"chunks": [{{"headline": "...", "summary": "...", "original_text": "..."}}]}}"""


def extract_json_from_response(response: str) -> dict:
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()

    start = response.find("{")
    end = response.rfind("}") + 1
    if start != -1 and end > start:
        response = response[start:end]
    return json.loads(response)


def chunk_document(text: str, filename: str, doc_type: str = "uploaded") -> list[dict]:
    prompt = make_chunking_prompt(text, filename, doc_type)
    try:
        response = call_llm(prompt)
        json_data = extract_json_from_response(response)
        chunks = Chunks.model_validate(json_data)

        results = []
        for chunk in chunks.chunks:
            results.append(
                {
                    "page_content": f"{chunk.headline}\n\n{chunk.summary}\n\n{chunk.original_text}",
                    "metadata": {
                        "source": filename,
                        "type": doc_type,
                        "uploaded_at": datetime.now().isoformat(),
                    },
                }
            )
        return results
    except Exception as exc:
        print(f"Chunking failed: {exc}")
        return [
            {
                "page_content": text,
                "metadata": {
                    "source": filename,
                    "type": doc_type,
                    "uploaded_at": datetime.now().isoformat(),
                },
            }
        ]


def get_next_id() -> int:
    existing_ids = collection.get()["ids"]
    if not existing_ids:
        return 0
    return max(int(doc_id) for doc_id in existing_ids) + 1


def upload_document(text: str, filename: str, doc_type: str = "uploaded") -> dict:
    try:
        print(f"[1/3] Chunking document: {filename}")
        chunks = chunk_document(text, filename, doc_type)
        if not chunks:
            return {
                "status": "error",
                "message": "Failed to chunk document",
                "chunks_added": 0,
            }

        print(f"[2/3] Creating embeddings for {len(chunks)} chunks")
        texts = [chunk["page_content"] for chunk in chunks]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, input=texts
        )
        vectors = [item.embedding for item in response.data]

        print("[3/3] Adding to ChromaDB")
        start_id = get_next_id()
        ids = [str(start_id + index) for index in range(len(chunks))]
        metadatas = [chunk["metadata"] for chunk in chunks]

        collection.add(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )
        return {
            "status": "success",
            "message": f"Successfully uploaded '{filename}'",
            "chunks_added": len(chunks),
            "total_documents": collection.count(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc), "chunks_added": 0}


def list_documents() -> list[dict]:
    all_data = collection.get(include=["metadatas"])
    if not all_data["metadatas"]:
        return []

    docs: dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in docs:
            docs[source] = {
                "source": source,
                "type": meta.get("type", "unknown"),
                "chunk_count": 0,
                "uploaded_at": meta.get("uploaded_at", "N/A"),
            }
        docs[source]["chunk_count"] += 1

    return sorted(docs.values(), key=lambda item: item["source"])


def delete_document(source: str) -> dict:
    try:
        all_data = collection.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            if meta.get("source") == source:
                ids_to_delete.append(doc_id)

        if not ids_to_delete:
            return {
                "status": "error",
                "message": f"Document '{source}' not found",
                "chunks_deleted": 0,
            }

        collection.delete(ids=ids_to_delete)
        return {
            "status": "success",
            "message": f"Deleted '{source}'",
            "chunks_deleted": len(ids_to_delete),
            "total_documents": collection.count(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc), "chunks_deleted": 0}


def get_stats() -> dict:
    docs = list_documents()
    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "document_types": sorted({doc["type"] for doc in docs}),
    }
