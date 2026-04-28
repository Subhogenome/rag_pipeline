"""
pipeline/reranker.py — Cross-encoder reranking stage.

Sits between retrieval and final cutoff:
  HNSW + BM25 + Graph  →  top-N candidates  →  CrossEncoder  →  re-sorted top-K

The cross-encoder reads (query, chunk_text) together as a single input,
giving it full attention over both simultaneously — far more precise than
bi-encoder cosine similarity which embeds them independently.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 80MB, runs on CPU
  - Trained on MS-MARCO passage ranking
  - ~40–80ms for a batch of 15 pairs on CPU

Fallback: if model unavailable, returns input unchanged (no reranking).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── Model loader (singleton) ──────────────────────────────────────────────────

_model       = None
_model_tried = False


def _load_model(model_name: str):
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder [{model_name}]...")
        _model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Cross-encoder loaded ✓ ({model_name})")
    except Exception as e:
        logger.warning(
            f"Cross-encoder unavailable ({str(e)[:80]})\n"
            f"  → Skipping reranking. Install: pip install sentence-transformers"
        )
        _model = None
    return _model


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks a list of RetrievedChunk objects using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker.from_config(cfg)
        results  = reranker.rerank(query, results)
    """

    def __init__(self, model_name: str, batch_size: int = 32, enabled: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.enabled    = enabled
        self._model     = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def rerank(self, query: str, results: list) -> list:
        """
        Rerank retrieved chunks by cross-encoder score.

        Args:
            query:   user query string
            results: list of RetrievedChunk from retriever.py

        Returns:
            Re-sorted list of RetrievedChunk with updated .score and .norm_score
        """
        if not self.enabled or not results:
            return results

        model = self._get_model()
        if model is None:
            return results   # graceful fallback — no crash

        # Build (query, chunk_text) pairs
        # Prefix chunk with paper + section for additional context
        pairs = []
        for r in results:
            c    = r.chunk
            text = f"[{c.paper_title[:40]} — {c.section}] {c.text}"
            pairs.append((query, text[:512]))  # CrossEncoder max_length cap

        try:
            import torch
            with torch.no_grad():
                scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

            # Combine cross-encoder score with original retrieval score
            # Weighted: 70% cross-encoder, 30% original (preserves graph/consensus signals)
            ce_scores = np.array(scores, dtype=float)
            # Normalise cross-encoder scores to [0, 1]
            ce_min, ce_max = ce_scores.min(), ce_scores.max()
            if ce_max > ce_min:
                ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
            else:
                ce_norm = np.ones_like(ce_scores) * 0.5

            orig_scores = np.array([r.norm_score for r in results])

            final_scores = 0.70 * ce_norm + 0.30 * orig_scores

            # Re-sort by combined score
            order = np.argsort(final_scores)[::-1]
            reranked = []
            top = float(final_scores[order[0]]) if len(order) else 1.0
            for idx in order:
                r            = results[idx]
                r.score      = float(final_scores[idx])
                r.norm_score = float(final_scores[idx]) / max(top, 1e-9)
                r.sources    = list(set(r.sources) | {"reranked"})
                reranked.append(r)

            logger.debug(
                f"Cross-encoder reranked {len(results)} chunks. "
                f"Top CE score: {ce_scores[order[0]]:.3f}"
            )
            return reranked

        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed ({str(e)[:80]}) — using original order")
            return results

    @classmethod
    def from_config(cls, cfg) -> "CrossEncoderReranker":
        rerank_cfg = getattr(cfg, "reranker", None)
        if rerank_cfg is None:
            return cls(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                batch_size=32,
                enabled=True,
            )
        return cls(
            model_name=getattr(rerank_cfg, "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            batch_size=getattr(rerank_cfg, "batch_size", 32),
            enabled=getattr(rerank_cfg, "enabled", True),
        )


# ── Convenience function ──────────────────────────────────────────────────────

def rerank(query: str, results: list, cfg) -> list:
    """
    Single-call interface. Creates reranker from config and reranks.
    Called from query.py and retriever pipeline.
    """
    reranker = CrossEncoderReranker.from_config(cfg)
    return reranker.rerank(query, results)
