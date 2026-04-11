# AI Internal Knowledge Assistant

<p align="center">
  <a href="https://www.youtube.com/watch?v=Sxkf9a6Gkrs">
    <img src="https://img.shields.io/badge/Demo%20Video-Watch%20on%20YouTube-red?style=for-the-badge&logo=youtube" alt="Demo Video" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Gradio-UI-orange?style=flat-square" alt="Gradio" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-brightgreen?style=flat-square" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/RAG-Advanced-purple?style=flat-square" alt="Advanced RAG" />
  <img src="https://img.shields.io/badge/Embeddings-HuggingFace-yellow?style=flat-square" alt="Embeddings" />
  <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-black?style=flat-square" alt="OpenAI Compatible API" />
</p>

A portfolio project that turns a fictional company's internal documents into an AI assistant using an advanced RAG pipeline.

This project goes beyond a basic chatbot by combining semantic chunking, vector retrieval, query rewriting, multi-query retrieval, LLM re-ranking, and incremental document management in a single end-to-end application.

## Project Highlights

- Built an end-to-end advanced RAG application with ingestion, retrieval, re-ranking, and UI
- Implemented query rewriting and multi-query retrieval to improve answer quality
- Added admin-style document upload and deletion for a more realistic internal AI workflow
- Used local embeddings with an OpenAI-compatible API to balance cost and performance

## Demo

- Video demo: `https://www.youtube.com/watch?v=Sxkf9a6Gkrs`
- The app supports both question answering and admin-style document management

## Table of Contents

- [Why This Project Matters](#why-this-project-matters)
- [What The App Does](#what-the-app-does)
- [Core Techniques Used](#core-techniques-used)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Skills Demonstrated](#skills-demonstrated)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Engineering Decisions](#engineering-decisions)
- [Example Questions](#example-questions)
- [Future Improvements](#future-improvements)
- [License](#license)

## Why This Project Matters

Many RAG demos stop at: embed documents -> retrieve top chunks -> answer.

This project is designed to be closer to a real internal AI tool:

- document ingestion with LLM-based semantic chunking
- advanced retrieval with query rewriting and re-ranking
- local embeddings for lower cost
- admin-style document upload and deletion
- modular code structure for ingestion, retrieval, and UI

It demonstrates the kind of thinking needed for AI product development: balancing quality, cost, maintainability, and user workflow.

## What The App Does

- answers questions over a fictional internal knowledge base
- uses conversation history during Q&A
- lets an admin upload new `.md` and `.txt` files
- supports incremental ingestion without rebuilding the full vector database
- allows listing and deleting indexed documents

## Core Techniques Used

### 1. Semantic Chunking

Instead of splitting documents only by character count, the ingestion pipeline uses an LLM to create chunks with:

- a short headline
- a compact summary
- the original text

This gives each chunk stronger semantic structure for retrieval.

### 2. Query Rewriting

User questions are often vague or conversational. Before retrieval, the system rewrites the question into a more search-friendly form.

Example:

- user question: `What do you offer?`
- rewritten query: `List company products and services`

This improves recall.

### 3. Multi-Query Retrieval

The system retrieves with both:

- the original user question
- the rewritten query

Then it merges and deduplicates the results.

This reduces the chance of missing useful chunks.

### 4. LLM Re-Ranking

Vector similarity is useful, but it does not always produce the best final ranking.

After retrieval, the system asks an LLM to re-rank the candidate chunks based on actual relevance to the question.

This improves precision before answer generation.

### 5. Incremental Document Management

New files can be uploaded and indexed without reprocessing the full knowledge base.

This is handled through a document manager module and exposed through the Gradio UI.

## Architecture

```text
knowledge-base/ documents
        |
        v
src/ingest.py
  - load files
  - semantic chunking
  - local embeddings
  - store in ChromaDB
        |
        v
preprocessed_db/
        |
        v
src/answer.py
  - rewrite query
  - retrieve original query
  - retrieve rewritten query
  - merge results
  - rerank chunks
  - generate final answer
        |
        v
app.py
  - chat UI
  - admin upload/delete UI
```

## Project Structure

```text
├── app.py
├── src/
│   ├── answer.py
│   ├── ingest.py
│   └── document_manager.py
├── knowledge-base/
├── preprocessed_db/          # auto-generated
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Application UI | Gradio |
| LLM API | OpenAI-compatible API |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Validation | Pydantic |
| Language | Python |

## Skills Demonstrated

This project demonstrates hands-on ability in:

- LLM application development
- retrieval-augmented generation
- prompt design for chunking and rewriting
- vector search and semantic retrieval
- ranking and answer quality improvement
- Python backend organization
- lightweight AI product UI design
- document ingestion workflows

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` with your provider settings.

### 5. Add or review knowledge base documents

Put your markdown files under `knowledge-base/`, for example:

```text
knowledge-base/
├── company/
├── products/
├── policies/
└── contracts/
```

### 6. Build the vector database

```bash
python -m src.ingest
```

### 7. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | API key for your OpenAI-compatible provider | Yes |
| `OPENAI_BASE_URL` | Provider endpoint URL | No |
| `ANSWER_MODEL` | Model used for final answer generation | No |
| `REWRITE_MODEL` | Model used for query rewriting | No |
| `RERANK_MODEL` | Model used for re-ranking chunks | No |
| `CHUNKING_MODEL` | Model used during ingestion chunking | No |
| `ADMIN_PASSWORD` | Password for document management UI | Yes |

## Engineering Decisions

### Why local embeddings?

I use local HuggingFace embeddings to reduce recurring API cost while still getting solid retrieval quality for a portfolio-scale knowledge base.

### Why separate ingestion from answering?

Because they solve different problems:

- ingestion prepares and structures knowledge
- answering focuses on retrieval quality and response generation

This separation also makes the codebase easier to maintain and extend.

### Why include document management?

Because real AI tools are rarely static. Internal knowledge changes over time, so upload/delete flows are important for making the app feel closer to a practical internal product.

## Example Questions

- `What products does the company offer?`
- `What is the company culture?`
- `What are the employee benefits?`
- `What information is available in the contracts folder?`

## Future Improvements

- source citations in the chat response
- RAG evaluation dashboard
- role-based access control
- async/background ingestion jobs
- hybrid search and metadata filtering
- better observability for latency and retrieval quality

## Notes

- all data in this project is fictional
- this repo is intended for learning, portfolio, and experimentation
- the vector database is generated locally and should not be committed

## License

MIT License
