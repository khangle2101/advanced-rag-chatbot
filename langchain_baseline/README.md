# LangChain Baseline

This folder contains the comparison baseline used against the main `Advanced RAG` implementation.

It is intentionally simpler:

- character-based chunking with `RecursiveCharacterTextSplitter`
- HuggingFace embeddings
- basic similarity retrieval
- direct answer generation without rewrite or rerank steps

The point of including this baseline is to show the difference between:

- fast framework-based prototyping with `LangChain`
- a more custom `Advanced RAG` pipeline tuned through evaluation

Run it with:

```bash
python -m langchain_baseline.ingest
python -m langchain_baseline.answer
```
