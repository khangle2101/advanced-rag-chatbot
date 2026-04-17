"""LangChain baseline ingestion pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(override=True)

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
DB_NAME = str(Path(__file__).parent.parent / "langchain_vectordb")
COLLECTION_NAME = "langchain_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    if Path(DB_NAME).exists():
        shutil.rmtree(DB_NAME)
        print("Removed existing LangChain vector store")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store created with {vectorstore._collection.count()} documents")
    return vectorstore


def main():
    print("LANGCHAIN BASELINE INGEST")
    docs = load_documents()
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)


if __name__ == "__main__":
    main()
