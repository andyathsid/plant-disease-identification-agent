# AutoRAG Retrieval Evaluation Setup

This setup evaluates different retrieval components for plant disease information using AutoRAG, Qdrant, and LangChain.

## 1. Installation

Ensure you have `uv` installed, then install the required packages:

```bash
uv pip install autorag langchain-google-genai langchain-openai qdrant-client pandas pyarrow
```

## 2. Environment Variables

Create or update your `.env` file with the following:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
VOYAGE_API_KEY=your_voyage_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_URL=http://localhost:6333
```

## 3. Execution Steps

### Step A: Generate Unified Corpus
Converts `plantvillage.jsonl` and `gardenology.jsonl` into a unified AutoRAG format.
```bash
uv run generate_corpus.py
```

### Step B: Generate Ground Truth (QA)
Uses LLM to generate questions based on the corpus for evaluation.
```bash
uv run generate_qa.py
```

### Step C: Run Evaluation
Runs the comparison between Gemini/OpenAI embeddings, BM25/Hybrid, and Voyage/Jina rerankers.
```bash
uv run python -m autorag.cli evaluate --config eval_config.yaml --corpus_data_path data/corpus.parquet --qa_data_path data/qa.parquet
```

## 4. Evaluation Metrics
The evaluation focuses on:
- **Retrieval Recall**: How many relevant documents were found.
- **Retrieval NDCG**: The quality of the ranking.
- **Retrieval F1**: Balance between precision and recall.

Results will be saved in the `benchmark` folder created by AutoRAG.
