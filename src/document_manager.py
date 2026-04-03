"""
Document Manager - Incremental Ingestion
=========================================
Handles uploading new documents to the knowledge base without
re-indexing the entire database.

Features:
1. Upload single document -> Chunk -> Embed -> Append to DB
2. List all documents in knowledge base
3. Delete document from knowledge base

Author: Khang Le
"""

import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import json


load_dotenv(override=True)

# ============ CONFIGURATION ============
# API Configuration - set in .env file
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "gpt-4o-mini")

# Database paths
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
COLLECTION_NAME = "knowledge_base"

# Embedding model (local HuggingFace - free!)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Average chunk size
AVERAGE_CHUNK_SIZE = 100

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


# ============ DATA MODELS ============
class Chunk(BaseModel):
    """A single semantic chunk created by the LLM"""

    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="2-3 sentences summarizing the chunk")
    original_text: str = Field(description="The exact original text")


class Chunks(BaseModel):
    """Collection of chunks from a document"""

    chunks: list[Chunk]


# ============ LLM CALLS ============
def call_llm(prompt: str) -> str:
    """Call LLM API for chunking."""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    response = client.chat.completions.create(
        model=CHUNKING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ============ CHUNKING ============
def make_chunking_prompt(text: str, filename: str, doc_type: str = "uploaded") -> str:
    """Create prompt for LLM to chunk the document."""
    how_many = max(1, len(text) // AVERAGE_CHUNK_SIZE)

    return f"""You take a document and split it into overlapping chunks for a KnowledgeBase.

Document type: {doc_type}
Document source: {filename}

A chatbot will use these chunks to answer questions.

INSTRUCTIONS:
1. Split into approximately {how_many} chunks (can vary as appropriate)
2. Each chunk should be self-contained enough to answer specific questions
3. Include ~25% overlap between chunks for context continuity
4. Don't leave anything out - entire document must be represented

For each chunk provide:
- headline: Brief heading (few words) likely to match search queries
- summary: 2-3 sentences summarizing the chunk content
- original_text: The exact original text (don't modify it)

DOCUMENT:

{text}

Respond with valid JSON in this format:
{{"chunks": [{{"headline": "...", "summary": "...", "original_text": "..."}}]}}"""


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
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
    """
    Chunk a document using LLM-based semantic chunking.
    Returns list of dicts with: page_content, metadata
    """
    prompt = make_chunking_prompt(text, filename, doc_type)

    try:
        response = call_llm(prompt)
        json_data = extract_json_from_response(response)
        chunks = Chunks.model_validate(json_data)

        results = []
        for chunk in chunks.chunks:
            page_content = (
                f"{chunk.headline}\n\n{chunk.summary}\n\n{chunk.original_text}"
            )
            results.append(
                {
                    "page_content": page_content,
                    "metadata": {
                        "source": filename,
                        "type": doc_type,
                        "uploaded_at": datetime.now().isoformat(),
                    },
                }
            )

        return results

    except Exception as e:
        print(f"Chunking failed: {e}")
        # Fallback: return whole document as single chunk
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


# ============ DOCUMENT OPERATIONS ============
def get_next_id() -> int:
    """Get the next available ID for new documents."""
    existing_ids = collection.get()["ids"]
    if not existing_ids:
        return 0

    max_id = max(int(id_) for id_ in existing_ids)
    return max_id + 1


def upload_document(text: str, filename: str, doc_type: str = "uploaded") -> dict:
    """
    Upload a new document to the knowledge base.

    Process:
    1. Chunk the document using LLM
    2. Create embeddings
    3. Append to ChromaDB (incremental, not full re-index)

    Returns:
        dict with status, message, chunks_added
    """
    try:
        # Step 1: Chunk the document
        print(f"[1/3] Chunking document: {filename}")
        chunks = chunk_document(text, filename, doc_type)

        if not chunks:
            return {
                "status": "error",
                "message": "Failed to chunk document",
                "chunks_added": 0,
            }

        # Step 2: Create embeddings
        print(f"[2/3] Creating embeddings for {len(chunks)} chunks")
        texts = [chunk["page_content"] for chunk in chunks]
        vectors = embedding_model.encode(texts).tolist()

        # Step 3: Add to ChromaDB (incremental!)
        print(f"[3/3] Adding to ChromaDB")
        start_id = get_next_id()
        ids = [str(start_id + i) for i in range(len(chunks))]
        metadatas = [chunk["metadata"] for chunk in chunks]

        collection.add(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )

        print(f"Successfully added {len(chunks)} chunks from {filename}")

        return {
            "status": "success",
            "message": f"Successfully uploaded '{filename}'",
            "chunks_added": len(chunks),
            "total_documents": collection.count(),
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "chunks_added": 0}


def list_documents() -> list[dict]:
    """
    List all unique documents in the knowledge base.
    Returns list of dicts with: source, type, chunk_count
    """
    all_data = collection.get(include=["metadatas"])

    if not all_data["metadatas"]:
        return []

    # Group by source
    docs = {}
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

    # Sort by source name
    return sorted(docs.values(), key=lambda x: x["source"])


def delete_document(source: str) -> dict:
    """
    Delete all chunks belonging to a document.

    Args:
        source: The source filename to delete

    Returns:
        dict with status, message, chunks_deleted
    """
    try:
        # Find all IDs with this source
        all_data = collection.get(include=["metadatas"])

        ids_to_delete = []
        if all_data["ids"] and all_data["metadatas"]:
            for id_, meta in zip(all_data["ids"], all_data["metadatas"]):
                if meta.get("source") == source:
                    ids_to_delete.append(id_)

        if not ids_to_delete:
            return {
                "status": "error",
                "message": f"Document '{source}' not found",
                "chunks_deleted": 0,
            }

        # Delete from collection
        collection.delete(ids=ids_to_delete)

        return {
            "status": "success",
            "message": f"Deleted '{source}'",
            "chunks_deleted": len(ids_to_delete),
            "total_documents": collection.count(),
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "chunks_deleted": 0}


def get_stats() -> dict:
    """Get knowledge base statistics."""
    docs = list_documents()

    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "document_types": list(set(d["type"] for d in docs)),
    }


# ============ TESTING ============
if __name__ == "__main__":
    print("=" * 50)
    print("Document Manager Test")
    print("=" * 50)

    # Test stats
    stats = get_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Document types: {stats['document_types']}")

    # Test list documents
    print(f"\nDocuments in KB:")
    docs = list_documents()
    for doc in docs[:10]:  # Show first 10
        print(f"  - {doc['source']} ({doc['chunk_count']} chunks)")

    if len(docs) > 10:
        print(f"  ... and {len(docs) - 10} more")
