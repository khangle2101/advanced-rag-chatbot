# LangChain vs Advanced RAG

This repository contains both the main `Advanced RAG` implementation and a smaller `LangChain` baseline so the comparison can be reproduced directly from the codebase.

## Why This Comparison Matters

The goal was not only to build a working chatbot, but to answer a more engineering-focused question:

`Which retrieval decisions actually improve quality in a measurable way?`

That is why I compared:

- a `LangChain` baseline
- an `Advanced RAG` pipeline with retrieval tuning

## Design Trade-Offs

| Area | LangChain | Advanced RAG |
|---|---|---|
| Main strength | Fast prototyping | Higher retrieval control |
| Chunking | Character-based | Semantic chunking |
| Retrieval | Basic similarity search | Original + rewritten query retrieval |
| Ranking | Default retrieval order | LLM re-ranking |
| Tuning style | Framework defaults | Evaluation-driven tuning |

## Evaluation Summary

| Metric | LangChain | Advanced RAG |
|---|---:|---:|
| MRR | 0.7442 | 0.9290 |
| nDCG | 0.7602 | 0.9247 |
| Keyword Coverage | 86.4% | 96.6% |
| Accuracy | 4.05/5 | 4.79/5 |
| Completeness | 3.99/5 | 4.35/5 |
| Relevance | 4.68/5 | 4.77/5 |

## Main Lesson

The largest gains did not come from changing the answer model alone.

The biggest improvements came from:

- query rewriting
- multi-query retrieval
- reranking
- tuning retrieval parameters based on evaluation feedback

That is the core story of this project: move from a good baseline to a measurably stronger RAG system by improving retrieval quality.
