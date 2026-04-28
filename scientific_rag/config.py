"""config.py — load config.yaml + override from .env / environment variables."""
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


@dataclass
class ChunkingConfig:
    min_tokens: int = 80
    max_tokens: int = 700
    overlap_tokens: int = 80
    semantic_split_threshold: float = 0.35
    include_references: bool = False
    include_captions: bool = True
    include_abstract: bool = True


@dataclass
class EmbeddingConfig:
    dims: int = 256
    batch_size: int = 64
    model_name: str = "all-MiniLM-L6-v2"       # primary ST model
    fallback_model: str = "all-MiniLM-L6-v2"  # fallback if primary unavailable


@dataclass
class HNSWConfig:
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 80
    space: str = "cosine"


@dataclass
class AzureConfig:
    """Single Azure AI Foundry config used for ALL LLM calls — tagging, chat, eval."""
    api_version: str = "2024-05-01-preview"
    # tagging / extraction settings (fast, cheap)
    tag_max_tokens: int = 600
    tag_temperature: float = 0.1
    tag_batch_size: int = 2
    tag_request_delay: float = 0.3
    tag_max_retries: int = 3
    tpm_limit: int = 6000            # tokens per minute — set to 80% of your actual Azure quota
    # chat / answer settings
    chat_max_tokens: int = 1500
    chat_temperature: float = 0.2
    system_prompt: str = (
        "You are a scientific research assistant. Answer questions using ONLY the "
        "provided context chunks from biomedical papers. Be precise, cite which "
        "paper each claim comes from, and flag if context is insufficient."
    )

    @property
    def endpoint(self) -> str:
        return os.getenv("AZURE_ENDPOINT", "")

    @property
    def deployment(self) -> str:
        return os.getenv("AZURE_DEPLOY", "")

    @property
    def api_key(self) -> str:
        key = os.getenv("AZURE_KEY", "")
        if not key:
            raise EnvironmentError("Missing env var: AZURE_KEY")
        return key


@dataclass
class LLMLinguaConfig:
    enabled: bool = True
    target_token: int = 400
    rate: float = 0.4
    rank_method: str = "order"
    use_llmlingua2: bool = False   # set True to enable LLMLingua2 (downloads ~700MB BERT model)


@dataclass
class RetrievalConfig:
    # Dynamic cutoff (replaces fixed top_k)
    score_threshold:      float = 0.75   # keep chunks >= threshold × top_score
    min_dip_rate:         float = 0.08   # min relative drop to trigger a cut
    min_k:                int   = 1      # always return at least this many
    max_k:                int   = 15     # hard ceiling
    # Retrieval weights
    hnsw_candidates:      int   = 40
    bm25_weight:          float = 0.35
    dense_weight:         float = 0.45
    graph_weight:         float = 0.20
    rrf_k:                int   = 60
    graph_walk_depth:     int   = 2
    graph_walk_max_nodes: int   = 4
    theme_boost:          float = 1.25
    contradiction_penalty: float = 0.85


@dataclass
class RerankerConfig:
    enabled:    bool  = True
    model_name: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 80MB, CPU-friendly
    batch_size: int   = 32
    ce_weight:  float = 0.70   # cross-encoder score weight (vs 0.30 retrieval score)


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    n_golden_questions: int = 20
    faithfulness_threshold: float = 0.7


@dataclass
class PipelineConfig:
    pdf_dir: str = "./pdfs"
    output_dir: str = "./output"
    log_level: str = "INFO"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    llmlingua: LLMLinguaConfig = field(default_factory=LLMLinguaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker:   RerankerConfig = field(default_factory=RerankerConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "PipelineConfig":
        cfg_path = Path(path)
        raw: dict = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}

        cfg = cls()
        _apply(cfg, "chunking",   ChunkingConfig,   raw)
        _apply(cfg, "embedding",  EmbeddingConfig,  raw)
        _apply(cfg, "hnsw",       HNSWConfig,       raw)
        _apply(cfg, "azure",      AzureConfig,      raw)
        _apply(cfg, "llmlingua",  LLMLinguaConfig,  raw)
        _apply(cfg, "retrieval",  RetrievalConfig,  raw)
        _apply(cfg, "reranker",   RerankerConfig,   raw)
        _apply(cfg, "evaluation", EvalConfig,       raw)

        # top-level overrides from env
        for key in ("pdf_dir", "output_dir", "log_level"):
            env_val = os.getenv(key.upper())
            if env_val:
                setattr(cfg, key, env_val)
            elif key in raw:
                setattr(cfg, key, raw[key])

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        return cfg


def _apply(cfg: PipelineConfig, attr: str, klass, raw: dict) -> None:
    if attr in raw:
        current = getattr(cfg, attr)
        for k, v in raw[attr].items():
            if hasattr(current, k):
                setattr(current, k, v)
