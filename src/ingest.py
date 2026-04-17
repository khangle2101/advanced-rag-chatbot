"""Advanced RAG ingestion pipeline.

Builds the ChromaDB vector store used by the public Advanced RAG demo.
Pipeline:
1. Load markdown documents from `knowledge-base/`
2. Use OpenRouter for semantic chunking
3. Create OpenAI embeddings via OpenRouter
4. Store chunks and vectors in ChromaDB
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is required. Add it to your .env file.")

CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "openai/gpt-4.1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs_advanced"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 100
MAX_WORKERS = 5
EMBED_BATCH_SIZE = 100

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
embedding_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading likely to match search queries")
    summary: str = Field(
        description="A short summary that captures the key ideas of the chunk"
    )
    original_text: str = Field(
        description="The original chunk text copied exactly from the document"
    )

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents() -> list[dict]:
    documents: list[dict] = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as handle:
                documents.append(
                    {"type": doc_type, "source": file.as_posix(), "text": handle.read()}
                )

    print(f"Loaded {len(documents)} documents")
    return documents


def make_chunking_prompt(document: dict) -> str:
    how_many = max(1, len(document["text"]) // AVERAGE_CHUNK_SIZE)
    return f"""You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a fictional company called Insurellm.
Document type: {document["type"]}
Document source: {document["source"]}

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

{document["text"]}

Respond with valid JSON in this format:
{{"chunks": [{{"headline": "...", "summary": "...", "original_text": "..."}}]}}"""


def call_llm(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CHUNKING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"Chunking attempt {attempt + 1} failed: {exc}")
            if attempt == max_retries - 1:
                raise
    return ""


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


def process_document(document: dict) -> list[Result]:
    prompt = make_chunking_prompt(document)
    try:
        response = call_llm(prompt)
        json_data = extract_json_from_response(response)
        chunks = Chunks.model_validate(json_data)
        return [chunk.as_result(document) for chunk in chunks.chunks]
    except Exception as exc:
        print(f"Error processing {document['source']}: {exc}")
        return [
            Result(
                page_content=document["text"],
                metadata={"source": document["source"], "type": document["type"]},
            )
        ]


def create_chunks_parallel(documents: list[dict]) -> list[Result]:
    all_chunks: list[Result] = []
    print(f"Processing {len(documents)} documents with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_doc = {
            executor.submit(process_document, doc): doc for doc in documents
        }
        for future in tqdm(
            as_completed(future_to_doc),
            total=len(documents),
            desc="Creating semantic chunks",
        ):
            try:
                all_chunks.extend(future.result())
            except Exception as exc:
                doc = future_to_doc[future]
                print(f"Failed to process {doc['source']}: {exc}")

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def create_embeddings(chunks: list[Result]) -> None:
    print("Creating vector store...")
    chroma = PersistentClient(path=DB_NAME)

    existing_collections = [collection.name for collection in chroma.list_collections()]
    if COLLECTION_NAME in existing_collections:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    texts = [chunk.page_content for chunk in chunks]

    print(f"Creating embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings: list[list[float]] = []
    for index in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[index : index + EMBED_BATCH_SIZE]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, input=batch
        )
        embeddings.extend(item.embedding for item in response.data)

    ids = [str(index) for index in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Stored {len(chunks)} chunks in ChromaDB")


def main() -> None:
    print("=" * 60)
    print("ADVANCED RAG INGEST")
    print("=" * 60)
    print(f"Knowledge Base: {KNOWLEDGE_BASE_PATH}")
    print(f"Vector Store: {DB_NAME}")
    print(f"Chunking Model: {CHUNKING_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print("=" * 60)

    print("\n[1/3] Loading documents...")
    documents = fetch_documents()

    print("\n[2/3] Semantic chunking...")
    chunks = create_chunks_parallel(documents)

    print("\n[3/3] Creating embeddings and vector store...")
    create_embeddings(chunks)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
