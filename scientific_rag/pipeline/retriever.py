"""
pipeline/retriever.py — Hybrid retrieval with deduplication + improved reranking.

Improvements over v1:
  - Deduplication: if two chunks have cosine sim > 0.85, drop the lower-scored one
  - Entity exact match boost: strong signal when query terms match chunk entities exactly
  - Figure chunk boost: figure chunks get extra weight for visual/quantitative queries
  - Dynamic cutoff: unchanged, but min_k=3 ensures enough context
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from config import RetrievalConfig
from pipeline.chunker import Chunk
from pipeline.indexer import MultiIndex
from pipeline.graph   import ChunkGraph

# Lazy import to avoid circular dependency
def _get_reranker(cfg):
    try:
        from pipeline.reranker import CrossEncoderReranker
        return CrossEncoderReranker.from_config(cfg)
    except ImportError:
        return None

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk:          Chunk
    score:          float
    norm_score:     float
    sources:        List[str] = field(default_factory=list)
    graph_expanded: bool = False


# ── Dynamic cutoff ─────────────────────────────────────────────────────────────

def _dynamic_cutoff(
    scores:        List[float],
    threshold:     float = 0.50,
    min_dip_rate:  float = 0.15,
    min_k:         int   = 3,
    max_k:         int   = 15,
) -> Tuple[int, float]:
    if not scores:
        return min_k, 0.0

    top = scores[0]
    if top == 0:
        return min_k, 0.0

    floor   = top * threshold
    n_above = sum(1 for s in scores if s >= floor)
    n_above = max(n_above, min_k)

    if n_above <= 1:
        return max(min_k, 1), 0.0

    survivors = scores[:n_above]
    dips: List[Tuple[float, int]] = []
    for i in range(1, len(survivors)):
        prev = survivors[i - 1]
        curr = survivors[i]
        if prev > 0:
            dips.append(((prev - curr) / prev, i))

    if not dips:
        return max(min_k, min(n_above, max_k)), 0.0

    max_dip_rate, cut_at = max(dips, key=lambda x: x[0])
    n_keep = cut_at if max_dip_rate >= min_dip_rate else n_above
    return max(min_k, min(n_keep, max_k)), max_dip_rate


# ── Deduplication ──────────────────────────────────────────────────────────────

def _deduplicate(
    chunks: List[Chunk],
    scores: List[float],
    sim_threshold: float = 0.85,
) -> Tuple[List[Chunk], List[float]]:
    """
    Remove near-duplicate chunks.
    If two chunks share >85% cosine similarity, keep the higher-scored one.
    This directly addresses the redundancy=0.69 problem.
    """
    if len(chunks) <= 1:
        return chunks, scores

    try:
        vec   = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf = vec.fit_transform([c.text for c in chunks])
        sims  = cosine_similarity(tfidf)
    except Exception:
        return chunks, scores

    keep = [True] * len(chunks)
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            if sims[i, j] > sim_threshold:
                # Drop the lower-scored one
                if scores[i] >= scores[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    kept_chunks = [c for c, k in zip(chunks, keep) if k]
    kept_scores = [s for s, k in zip(scores,  keep) if k]
    removed = len(chunks) - len(kept_chunks)
    if removed:
        logger.debug(f"Deduplication: removed {removed} near-duplicate chunks")
    return kept_chunks, kept_scores


# ── RRF ────────────────────────────────────────────────────────────────────────

def _rrf(rankings: List[List[str]], weights: List[float], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    return scores


# ── Entity exact match ─────────────────────────────────────────────────────────

def _entity_match_boost(chunk: Chunk, query_tokens: Set[str]) -> float:
    """
    Boost score when query tokens exactly match chunk entities.
    Gene names and protein names are exact matches — not semantic ones.
    This fixes the Recall@1 problem where the right chunk doesn't rank first
    because 'TAF6' in query vs 'TAF6δ' in chunk loses semantic similarity.
    """
    if not chunk.entities or not query_tokens:
        return 1.0
    entity_tokens = set()
    for ent in chunk.entities:
        for tok in ent.lower().split():
            entity_tokens.add(tok)
    # How many query tokens appear in entity names
    overlap = len(query_tokens & entity_tokens)
    # Strong boost: each matching entity token adds 15%
    return 1.0 + (overlap * 0.15)


# ── Figure query detection ─────────────────────────────────────────────────────

def _is_figure_query(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in ("figure", "fig", "table", "shown", "panel",
                                 "quantitative", "measured", "graph", "plot",
                                 "chart", "illustrate", "depicted"))


# ── Main retrieval ─────────────────────────────────────────────────────────────

def retrieve(
    query:    str,
    embedder,
    index:    MultiIndex,
    graph:    ChunkGraph,
    cfg,             # full PipelineConfig or RetrievalConfig
) -> List[RetrievedChunk]:
    # Support both full PipelineConfig and bare RetrievalConfig
    rcfg = getattr(cfg, "retrieval", cfg)

    query_vec   = embedder.encode_one(query)
    query_toks  = set(query.lower().split())
    candidates  = rcfg.hnsw_candidates
    fig_query   = _is_figure_query(query)

    # ── 1–3. Three searches ───────────────────────────────────────────────
    dense_chunk  = index.search_dense(query_vec, k=candidates, use_hyde=False)
    dense_hyde   = index.search_dense(query_vec, k=candidates, use_hyde=True)
    bm25_results = index.search_bm25(query, k=candidates)

    dense_chunk_ids = [cid for cid, _ in dense_chunk]
    dense_hyde_ids  = [cid for cid, _ in dense_hyde]
    bm25_ids        = [cid for cid, _ in bm25_results]

    # ── 4. Weighted RRF ───────────────────────────────────────────────────
    rrf_scores = _rrf(
        rankings=[dense_chunk_ids, dense_hyde_ids, bm25_ids],
        weights =[rcfg.dense_weight * 0.6, rcfg.dense_weight * 0.4, rcfg.bm25_weight],
        k=rcfg.rrf_k,
    )
    if rrf_scores:
        max_rrf    = max(rrf_scores.values())
        rrf_scores = {k: v / max_rrf for k, v in rrf_scores.items()}

    chunk_sources: Dict[str, List[str]] = {}
    for cid in dense_chunk_ids: chunk_sources.setdefault(cid, []).append("dense")
    for cid in dense_hyde_ids:  chunk_sources.setdefault(cid, []).append("hyde")
    for cid in bm25_ids:        chunk_sources.setdefault(cid, []).append("bm25")

    # ── 5. Graph-walk expansion ───────────────────────────────────────────
    pre_graph_top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rcfg.max_k]
    top_ids       = [cid for cid, _ in pre_graph_top]
    expanded_ids  = graph.walk(
        seed_ids=top_ids,
        depth=rcfg.graph_walk_depth,
        max_nodes=rcfg.graph_walk_max_nodes,
        exclude=set(top_ids),
    )
    for cid in expanded_ids:
        if cid not in rrf_scores:
            rrf_scores[cid] = rcfg.graph_weight * 0.5
            chunk_sources.setdefault(cid, []).append("graph")

    # ── 6. Fetch + rerank ─────────────────────────────────────────────────
    all_ids    = list(rrf_scores.keys())
    chunks_map = {c.chunk_id: c for c in index.get_chunks_by_ids(all_ids)}

    scored: List[Tuple[float, str]] = []
    for cid, base in rrf_scores.items():
        chunk = chunks_map.get(cid)
        if not chunk:
            continue

        # Theme overlap boost
        theme_text    = f"{chunk.theme} {chunk.summary} {' '.join(chunk.entities)}".lower()
        overlap_ratio = len(query_toks & set(theme_text.split())) / max(len(query_toks), 1)
        theme_mult    = 1.0 + (rcfg.theme_boost - 1.0) * min(overlap_ratio * 3, 1.0)

        # Entity exact match boost (NEW — fixes Recall@1)
        entity_mult   = _entity_match_boost(chunk, query_toks)

        # Figure chunk boost for visual queries (NEW)
        figure_mult   = 1.3 if (fig_query and getattr(chunk, "is_figure_chunk", False)) else 1.0

        # Consensus bonus
        consensus_mult = 1.0 + 0.05 * min(len(chunk.consensus_papers), 4)

        # Contradiction penalty
        conflict_mult = rcfg.contradiction_penalty if chunk.conflict_chunk_ids else 1.0

        final = (base
                 * theme_mult
                 * entity_mult
                 * figure_mult
                 * consensus_mult
                 * conflict_mult
                 * (1.0 + 0.1 * chunk.confidence))
        scored.append((final, cid))

    scored.sort(reverse=True)

    # ── 7. Deduplication (NEW — fixes redundancy=0.69) ────────────────────
    ordered_chunks = [chunks_map[cid] for _, cid in scored if cid in chunks_map]
    ordered_scores = [s for s, cid in scored if cid in chunks_map]
    deduped_chunks, deduped_scores = _deduplicate(ordered_chunks, ordered_scores)

    # Normalise to [0, 1]
    top_score   = deduped_scores[0] if deduped_scores else 1.0
    norm_scores = [s / top_score for s in deduped_scores]

    # ── 8. Dynamic cutoff ─────────────────────────────────────────────────
    n_keep, dip_rate = _dynamic_cutoff(
        scores       = norm_scores,
        threshold    = rcfg.score_threshold,
        min_dip_rate = rcfg.min_dip_rate,
        min_k        = rcfg.min_k,
        max_k        = rcfg.max_k,
    )

    logger.info(
        f"Dynamic cutoff: {len(scored)} candidates → "
        f"{len(deduped_chunks)} after dedup → {n_keep} kept "
        f"(threshold={rcfg.score_threshold:.0%}, max_dip={dip_rate:.1%})"
    )

    results: List[RetrievedChunk] = []
    for i in range(min(n_keep, len(deduped_chunks))):
        chunk = deduped_chunks[i]
        results.append(RetrievedChunk(
            chunk=chunk,
            score=deduped_scores[i],
            norm_score=norm_scores[i],
            sources=chunk_sources.get(chunk.chunk_id, []),
            graph_expanded=chunk.chunk_id in set(expanded_ids),
        ))

    # ── 9. Cross-encoder reranking ────────────────────────────────────────
    reranker = _get_reranker(cfg)
    if reranker is not None and results:
        results = reranker.rerank(query, results)
        logger.info(f"  Cross-encoder reranked {len(results)} chunks")

    return results
