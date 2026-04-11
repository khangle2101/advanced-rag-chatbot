"""
Advanced RAG Ingest Module
==========================
Builds the vector database used by the chatbot.

Pipeline:
1. Load markdown documents from `knowledge-base/`
2. Use an OpenAI-compatible model for semantic chunking
3. Create local embeddings with HuggingFace
4. Store chunks and vectors in ChromaDB
"""

from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


load_dotenv(override=True)

# ============ CONFIGURATION ============
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs_advanced"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 100
MAX_WORKERS = 5

print("Loading HuggingFace embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Embedding model loaded: {EMBEDDING_MODEL_NAME}")


# ============ DATA MODELS ============
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


# ============ DOCUMENT LOADING ============
def fetch_documents() -> list[dict]:
    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"type": doc_type, "source": file.as_posix(), "text": f.read()}
                )

    print(f"Loaded {len(documents)} documents")
    return documents


# ============ LLM-BASED SEMANTIC CHUNKING ============
def make_chunking_prompt(document: dict) -> str:
    how_many = max(1, len(document["text"]) // AVERAGE_CHUNK_SIZE)

    return f"""You take a document and split it into overlapping chunks for a KnowledgeBase.

The document belongs to a fictional company used for a portfolio demo.
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
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
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
    except Exception as e:
        print(f"Error processing {document['source']}: {e}")
        return [
            Result(
                page_content=document["text"],
                metadata={"source": document["source"], "type": document["type"]},
            )
        ]


def create_chunks_parallel(documents: list[dict]) -> list[Result]:
    all_chunks = []

    print(
        f"Processing {len(documents)} documents with {MAX_WORKERS} parallel workers..."
    )

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
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                doc = future_to_doc[future]
                print(f"Failed to process {doc['source']}: {e}")

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


# ============ EMBEDDING & STORAGE ============
def create_embeddings(chunks: list[Result]):
    print("Creating ChromaDB vector store...")

    chroma = PersistentClient(path=DB_NAME)

    existing_collections = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing_collections:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]

    print("Creating embeddings with HuggingFace model...")
    vectors = embedding_model.encode(texts, show_progress_bar=True).tolist()

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    print(f"Vector store created at: {DB_NAME}")
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents")


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED RAG INGEST")
    print("=" * 60)
    print(f"Chunking Model: {CHUNKING_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Database: {DB_NAME}")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print("=" * 60)

    print("\n[1/3] Loading documents...")
    documents = fetch_documents()

    print("\n[2/3] Creating semantic chunks...")
    chunks = create_chunks_parallel(documents)

    print("\n[3/3] Creating embeddings and storing...")
    create_embeddings(chunks)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
