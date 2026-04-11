# AI Internal Knowledge Assistant

A RAG (Retrieval-Augmented Generation) chatbot for internal company knowledge management. Built as a portfolio project demonstrating advanced RAG techniques.

## Demo Video

[![Demo Video](https://img.shields.io/badge/YouTube-Watch_Demo-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=Sxkf9a6Gkrs)

[Watch the full demo on YouTube](https://www.youtube.com/watch?v=Sxkf9a6Gkrs)

## Features

### Chat Interface
- Natural language Q&A with company knowledge base
- Conversation history support
- Quick question examples

### Advanced RAG Pipeline
- **Query Rewriting**: Reformulates user questions for better retrieval
- **Multi-Query Retrieval**: Uses both original and rewritten queries for higher recall
- **LLM Re-Ranking**: Reorders retrieved chunks by relevance for higher precision

### Document Management (Admin Protected)
- Upload new documents (.md, .txt)
- Incremental ingestion (no full re-indexing needed)
- View all documents in knowledge base
- Delete documents
- Password-protected admin panel

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Any OpenAI-compatible API |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Database | ChromaDB |
| Web Framework | Gradio |
| Language | Python 3.10+ |

## Project Structure

```
├── app.py                    # Main Gradio web application
├── src/
│   ├── answer.py             # Advanced RAG answer generation
│   ├── ingest.py             # Document ingestion with LLM chunking
│   └── document_manager.py   # Document CRUD operations
├── knowledge-base/           # Source documents (markdown)
│   ├── company/
│   ├── products/
│   └── ...
├── preprocessed_db/          # ChromaDB storage (auto-generated)
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Create virtual environment

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
# Edit .env with your API keys
```

### 5. Prepare knowledge base

Add your markdown documents to the `knowledge-base/` folder, organized by type:

```
knowledge-base/
├── company/
│   └── about.md
├── products/
│   └── product1.md
└── policies/
    └── hr-policy.md
```

### 6. Ingest documents

```bash
python -m src.ingest
```

### 7. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_BASE_URL` | API endpoint URL | `https://api.openai.com/v1` |
| `ANSWER_MODEL` | Model for generating answers | `gpt-4o` |
| `REWRITE_MODEL` | Model for query rewriting | `gpt-4o-mini` |
| `RERANK_MODEL` | Model for re-ranking | `gpt-4o-mini` |
| `CHUNKING_MODEL` | Model for document chunking | `gpt-4o-mini` |
| `ADMIN_PASSWORD` | Password for document management | Required |

### Using Alternative API Providers

This project is compatible with any OpenAI-compatible API. The same provider can be used for semantic chunking, query rewriting, re-ranking, and final answer generation.

To use a different provider:

1. Set `OPENAI_BASE_URL` to your provider's endpoint
2. Set `OPENAI_API_KEY` to your provider's API key
3. Update model names to match your provider's available models

## RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   1. Query Rewriting                         │
│         "What insurance do you have?" →                      │
│         "List available insurance products"                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               2. Multi-Query Retrieval                       │
│     Retrieve chunks using BOTH original + rewritten query    │
│                    Merge & deduplicate                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. LLM Re-Ranking                         │
│        Use LLM to reorder chunks by relevance                │
│              Select top K most relevant                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  4. Answer Generation                        │
│      Generate response using context + conversation          │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT License - feel free to use this project for learning or as a starting point for your own RAG applications.

*This project uses fictional data for demonstration purposes. All company names, employee information, and contracts are entirely fictional.*
