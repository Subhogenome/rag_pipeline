# Scientific RAG Pipeline
### Elucidata GenAI Engineer Assignment

A production-ready chunking and retrieval pipeline for biomedical research PDFs.
Transforms raw scientific papers into a fully indexed, queryable RAG system with
section-aware parsing, LLM-powered metadata tagging, hybrid retrieval, a
cross-paper knowledge graph, and a cross-encoder reranker.

---

## Quick Start

```bash
pip install -r requirements.txt

python main.py                     # build index (~5 min with tag cache)
python create_golden.py --n 30     # generate evaluation ground truth
python evaluate.py                 # run metrics + ablation
python query.py                    # interactive chat
python evaluate.py --regen-golden  # regenerate golden set after index rebuild
```

---

## Project Structure

```
scientific_rag/
│
├── main.py                   # Entry point — orchestrates full index build
├── query.py                  # Interactive RAG chat interface (streaming)
├── evaluate.py               # Evaluation runner: metrics + ablation study
├── create_golden.py          # Generates ground-truth Q&A dataset from index
├── config.py                 # Typed configuration dataclasses (loaded from config.yaml)
├── config.yaml               # All pipeline parameters — nothing hardcoded
├── requirements.txt          # Python dependencies
├── .env                      # Credentials (not committed — see below)
│
├── pipeline/                 # Core processing modules
│   ├── pdf_parser.py         # PDF → structured sections with layout awareness
│   ├── chunker.py            # Sections → semantic chunks with citation guard
│   ├── tagger.py             # Azure LLM tagging: theme, summary, entities, HyDE
│   ├── confidence.py         # Per-chunk quality scoring (0.0–1.0)
│   ├── embedder.py           # PubMedBERT sentence embeddings + LSA fallback
│   ├── indexer.py            # HNSW × 2 + BM25 + SQLite metadata store
│   ├── graph.py              # Entity knowledge graph: consensus + contradiction
│   ├── retriever.py          # Hybrid retrieval: RRF fusion + graph walk + rerank
│   ├── reranker.py           # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   └── compressor.py         # Context compression before LLM (extractive TF-IDF)
│
├── chat/
│   └── azure_chat.py         # Azure AI Foundry chat with streaming output
│
├── evaluation/
│   └── evaluator.py          # Recall@K, MRR, NDCG@K, cross-paper recall, ablation
│
└── output/                   # Generated at runtime by main.py (not committed)
    ├── chunks.db             # SQLite: all chunk metadata
    ├── hnsw_chunk.bin        # HNSW vector index (chunk text embeddings)
    ├── hnsw_hyde.bin         # HNSW vector index (HyDE question embeddings)
    ├── bm25.pkl              # BM25 keyword index
    ├── graph.pkl             # Entity knowledge graph
    ├── embedder.pkl          # Fitted embedder + saved model name
    ├── tag_cache.json        # LLM tag cache — avoids re-tagging on rebuild
    ├── golden_qa.json        # Locked evaluation ground truth
    ├── index_summary.json    # Build statistics
    └── eval_report.json      # Full metrics + ablation results
```

---

## What Each File Does

### Entry Points

**`main.py`** — Runs the full 6-step pipeline: parse PDFs → tag chunks → score confidence → embed → build indexes → build knowledge graph. On re-runs, tags load from cache so the only slow step (Azure LLM tagging, ~90 min) is skipped.

**`query.py`** — Interactive chat interface. Loads the built index, accepts user questions, runs hybrid retrieval + reranking + compression, then streams the answer from Azure Llama-4-Maverick. Maintains multi-turn history.

**`evaluate.py`** — Runs the full evaluation suite. Loads the golden Q&A set, runs retrieval for each question, computes Recall@K / MRR / NDCG / Cross-paper Recall, and runs an ablation study disabling one component at a time. Validates that the golden set's chunk IDs still exist in the current index — warns and refuses to evaluate if stale.

**`create_golden.py`** — Generates a ground-truth evaluation dataset by asking Azure to produce questions from indexed chunks. Generates four question types: factual (single chunk), cross-paper (entity-linked pairs), figure-specific, and consensus. Run this once and lock the output.

**`config.py`** — Typed Python dataclasses (PipelineConfig, ChunkingConfig, EmbeddingConfig, HNSWConfig, AzureConfig, RerankerConfig, LLMLinguaConfig, RetrievalConfig, EvalConfig) that load from `config.yaml` with environment variable overrides.

---

### `pipeline/` — Core Modules

**`pdf_parser.py`** — Extracts text from PDFs using PyMuPDF. Detects two-column layouts and reads left column before right. Identifies section headers by font size, boldness, and pattern matching. Strips 13 categories of boilerplate (Additional files, Competing interests, Acknowledgements, etc.). Extracts figure and table captions as dedicated sections rather than discarding them.

**`chunker.py`** — Splits sections into chunks using spaCy sentence segmentation + TF-IDF cosine topic-shift detection. Never splits a sentence that ends with a citation marker ([12,14] or Smith et al., 2020). Section boundaries are hard cut points. Chunks: 80–650 tokens, 50-token overlap.

**`tagger.py`** — Sends each chunk to Azure Llama-4-Maverick and receives structured JSON: theme (3–5 word label), summary (1 sentence), entities (gene/protein/pathway names), hyde_question (hypothetical question the chunk answers), chunk_type (background/methods/results/conclusion). Saves all tags to `output/tag_cache.json` so rebuilds skip the API entirely. Uses a rolling TPM budget tracker to stay under the Azure rate limit.

**`confidence.py`** — Scores each chunk 0.0–1.0 using four signals: entity density (named entities per word), assertion strength (ratio of definitive to hedging words), section weight (Results=1.0, Methods=0.45, References=0.1), and token length (penalises very short or very long chunks). Used as a ranking multiplier in retrieval.

**`embedder.py`** — Produces two 768-dim vectors per chunk using `pritamdeka/S-PubMedBert-MS-MARCO` (a PubMedBERT model fine-tuned for biomedical passage ranking). One vector embeds the chunk text; the other embeds the LLM-generated HyDE question. Falls back to TF-IDF + SVD (LSA) if the sentence-transformers model is unavailable. Saves the model name inside `embedder.pkl` and verifies it on load to prevent silent embedding space mismatches.

**`indexer.py`** — Builds and manages three indexes: (1) HNSW on chunk text vectors, (2) HNSW on HyDE question vectors — both using hnswlib with M=16, ef=200, cosine space. (3) BM25 on enriched metadata text (theme + summary + entities). Stores all chunk metadata in SQLite (`chunks.db`). Saves the actual embedding dimension to metadata so loading works regardless of whether ST or LSA was used.

**`graph.py`** — Builds a NetworkX graph where nodes are chunks and edges connect chunks sharing biological entities. Cross-paper edges are weighted 1.5× (within-paper: 0.8×). Detects contradictions: cross-paper chunk pairs with opposing assertion polarity (increase vs decrease vocabulary). Detects consensus: cross-paper pairs with matching assertion direction — populates each chunk's `consensus_papers` field.

**`retriever.py`** — Eight-stage hybrid retrieval: (1) Dense HNSW on chunk vectors, (2) Dense HNSW on HyDE vectors, (3) BM25 on metadata, (4) Weighted RRF fusion, (5) Graph-walk expansion (2 hops from top results), (6) Reranking with entity exact-match boost + confidence multiplier + consensus bonus + contradiction penalty + figure boost for visual queries, (7) TF-IDF deduplication (cosine >0.85 → drop lower-scored), (8) Dynamic score-distribution cutoff (no fixed K).

**`reranker.py`** — Cross-encoder reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB). Reads each (query, chunk) pair jointly with full attention. Final score = 0.70 × cross-encoder + 0.30 × retrieval score (preserving graph/consensus signals). Downloads on first use, cached locally by sentence-transformers. Gracefully skipped if model unavailable.

**`compressor.py`** — Compresses retrieved chunks to a 400-token budget before sending to the LLM. Default: extractive TF-IDF — scores each sentence by cosine similarity to the query, keeps highest-scoring in original order. Optional: LLMLingua2 BERT compression (set `use_llmlingua2: true` in config.yaml, requires 700MB free disk).

---

### `chat/`

**`azure_chat.py`** — Sends compressed context + user query to Azure AI Foundry (Llama-4-Maverick-17B). Formats retrieved chunks with paper title, section, theme, and confidence as metadata headers. Returns streaming token iterator. Maintains conversation history (last 3 turns).

---

### `evaluation/`

**`evaluator.py`** — Computes retrieval metrics: Recall@K, Precision@K, MRR, NDCG@K, Cross-paper Recall. Handles graded relevance for NDCG (consensus chunks score 2, regular relevant chunks score 1). Computes chunk quality metrics: token statistics, entity/theme coverage, cosine redundancy, section distribution. Formats the ablation table for console output.

---

## Pipeline Architecture

```
6 Biomedical PDFs
  ↓ pdf_parser.py  — 2-col layout, section detection, figure chunks, noise filtering
447 Chunks (410 body + 37 figure/table)
  ↓ tagger.py      — Azure Llama-4-Maverick: theme, summary, entities, HyDE, type
  ↓ confidence.py  — entity density × assertion strength × section weight
  ↓ embedder.py    — PubMedBERT 768-dim: chunk_vec + hyde_vec per chunk
  ↓ indexer.py     — Dual HNSW + BM25 + SQLite
  ↓ graph.py       — Entity graph: 434 consensus pairs, 73 contradiction pairs
                    ↓ Query time
  ↓ retriever.py   — HNSW + HyDE + BM25 → RRF → graph walk → rerank → dedup → cutoff
  ↓ reranker.py    — Cross-encoder: Recall@1 from 0.133 → 0.450 (+238%)
  ↓ compressor.py  — Extractive TF-IDF → 400-token context
  ↓ azure_chat.py  — Llama-4-Maverick → streamed answer with citations
```

---

## Evaluation Results

| Metric | Value |
|---|---|
| MRR | **0.562** |
| Recall @ 1 | **0.450** |
| Recall @ 5 | **0.517** |
| NDCG @ 5 | **0.564** |
| Cross-paper Recall | **0.967** |
| Mean chunks returned | ~10 (dynamic) |
| Redundancy (cosine) | 0.497 |
| Entity coverage | 95.3% |

### Ablation Study (K=5, cross-encoder active)

| Configuration | Recall@1 | Recall@5 | MRR | NDCG@5 | XP-Recall |
|---|---|---|---|---|---|
| BM25 only | 0.383 | 0.450 | 0.467 | 0.468 | 0.983 |
| Dense only (PubMedBERT) | 0.433 | 0.533 | 0.553 | 0.556 | 0.983 |
| Dense + BM25 (RRF) | 0.450 | 0.517 | 0.562 | 0.564 | 0.967 |
| + Graph walk | 0.450 | 0.517 | 0.562 | 0.564 | 0.967 |
| Full pipeline | **0.450** | **0.517** | **0.562** | **0.564** | **0.967** |

---

## Configuration

All parameters in `config.yaml` — nothing hardcoded in source files.

```yaml
chunking:
  min_tokens: 80
  max_tokens: 650
  overlap_tokens: 50
  semantic_split_threshold: 0.35

embedding:
  model_name: "pritamdeka/S-PubMedBert-MS-MARCO"
  fallback_model: "all-MiniLM-L6-v2"

azure:
  tpm_limit: 6000       # set to 80% of your actual Azure quota

retrieval:
  score_threshold: 0.50
  min_dip_rate: 0.15
  min_k: 3
  max_k: 15

reranker:
  enabled: true
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  ce_weight: 0.70

llmlingua:
  use_llmlingua2: false  # true requires 700MB free disk
  target_token: 400
```

---

## Credentials

Create a `.env` file in the project root (not committed):

```
AZURE_ENDPOINT=https://...services.ai.azure.com/openai/v1/
AZURE_DEPLOY=Llama-4-Maverick-17B-128E-Instruct-FP8
AZURE_KEY=your-key-here
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Figure captions as own chunks | Key findings in these papers exist only in captions |
| Citation-aware split guard | Never separate a claim from its [12,14] citation |
| Noise section filtering | 45 boilerplate chunks removed (9% of corpus) |
| HyDE dual-index | Query-to-question matching beats query-to-dense-text |
| PubMedBERT embedder | Gene names are concepts, not rare tokens |
| Dynamic cutoff | Score distribution decides K, not an arbitrary constant |
| Cross-encoder reranker | Recall@1 +238% — biggest single improvement |
| Tag cache | Rebuild in 5 min vs 90 min without re-tagging |
| Embedding consistency guard | Prevents silent mismatch between index and query space |

---

## Known Limitations

- **Graph-walk shows no ablation lift at 447 chunks** — HNSW covers the same neighbourhood at this corpus size. Graph-walk earns value at 10k+ chunks.
- **Golden set is single-paper only** — Auto-generation could not produce cross-paper synthesis questions at scale. Cross-paper performance validated manually.
- **Entity disambiguation** — TAF6 and TAF6-delta are separate graph nodes. A UMLS entity linker would strengthen cross-paper edges.
