"""evaluation/evaluator.py — Retrieval metrics: Recall@K, Precision@K, MRR, NDCG@K, Cross-Paper."""
from __future__ import annotations
import json, logging, math
from dataclasses import dataclass, field
from typing import List, Dict, Set
from config import EvalConfig
from pipeline.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    question:           str
    relevant_chunk_ids: List[str]
    relevant_papers:    List[str]
    answer:             str  = ""
    difficulty:         str  = "single"  # "single" | "cross_paper" | "figure" | "consensus"


@dataclass
class EvalResult:
    question:           str
    difficulty:         str
    retrieved_ids:      List[str]
    retrieved_scores:   List[float]
    n_retrieved:        int          # dynamic — not fixed K
    recall_at_k:        Dict[int, float] = field(default_factory=dict)
    precision_at_k:     Dict[int, float] = field(default_factory=dict)
    ndcg_at_k:          Dict[int, float] = field(default_factory=dict)
    mrr:                float = 0.0
    cross_paper_recall: float = 0.0
    first_hit_rank:     int   = -1   # rank of first correct chunk (-1 = not found)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _dcg(relevances: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def _ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    rels  = [1 if cid in relevant else 0 for cid in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    dcg   = _dcg(rels)
    idcg  = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _graded_relevance(chunk_id: str, relevant: Set[str], consensus_papers: List[str]) -> int:
    """
    Graded relevance for NDCG:
      2 = directly relevant AND confirmed across multiple papers (consensus)
      1 = directly relevant
      0 = not relevant
    This rewards chunks that are cross-paper validated.
    """
    if chunk_id not in relevant:
        return 0
    return 2 if len(consensus_papers) > 0 else 1


# ── Single question evaluation ─────────────────────────────────────────────────

def evaluate_single(
    question: EvalQuestion,
    retrieved: List[RetrievedChunk],
    k_values: List[int],
) -> EvalResult:
    retrieved_ids    = [r.chunk.chunk_id for r in retrieved]
    retrieved_scores = [r.norm_score     for r in retrieved]
    relevant_set     = set(question.relevant_chunk_ids)
    relevant_papers  = set(question.relevant_papers)

    result = EvalResult(
        question=question.question,
        difficulty=question.difficulty,
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        n_retrieved=len(retrieved),
    )

    # Recall / Precision / NDCG at each K
    for k in k_values:
        top_k = retrieved_ids[:k]
        hits  = sum(1 for cid in top_k if cid in relevant_set)

        result.recall_at_k[k]    = hits / max(len(relevant_set), 1)
        result.precision_at_k[k] = hits / k if k > 0 else 0.0
        result.ndcg_at_k[k]      = _ndcg_at_k(top_k, relevant_set, k)

    # MRR — position of first correct chunk
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_set:
            result.mrr            = 1.0 / rank
            result.first_hit_rank = rank
            break

    # Cross-paper recall — did we retrieve from all required papers?
    if relevant_papers:
        retrieved_papers         = {r.chunk.paper_title for r in retrieved}
        result.cross_paper_recall = (
            len(relevant_papers & retrieved_papers) / len(relevant_papers)
        )

    return result


# ── Aggregate across questions ─────────────────────────────────────────────────

def aggregate(results: List[EvalResult], k_values: List[int]) -> Dict:
    if not results:
        return {}

    n         = len(results)
    single    = [r for r in results if r.difficulty == "single"]
    cross     = [r for r in results if r.difficulty == "cross_paper"]

    def _mean(lst, attr):
        vals = [getattr(r, attr) for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    def _mean_k(lst, attr, k):
        vals = [getattr(r, attr).get(k, 0) for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    agg = {
        "n_total":   n,
        "n_single":  len(single),
        "n_cross":   len(cross),
        "mean_n_retrieved": sum(r.n_retrieved for r in results) / n,
        "mrr":       _mean(results, "mrr"),
        "mrr_single": _mean(single, "mrr") if single else 0.0,
        "mrr_cross":  _mean(cross,  "mrr") if cross  else 0.0,
        "cross_paper_recall": _mean(results, "cross_paper_recall"),
        "recall_at_k":    {},
        "precision_at_k": {},
        "ndcg_at_k":      {},
    }
    for k in k_values:
        agg["recall_at_k"][k]    = _mean_k(results, "recall_at_k",    k)
        agg["precision_at_k"][k] = _mean_k(results, "precision_at_k", k)
        agg["ndcg_at_k"][k]      = _mean_k(results, "ndcg_at_k",      k)

    return agg


# ── Ablation helpers ───────────────────────────────────────────────────────────

def ablation_row(label: str, agg: Dict, k: int = 5) -> str:
    """Format one row of the ablation table."""
    r  = agg.get("recall_at_k",    {}).get(k, 0)
    p  = agg.get("precision_at_k", {}).get(k, 0)
    nd = agg.get("ndcg_at_k",      {}).get(k, 0)
    mr = agg.get("mrr", 0)
    cp = agg.get("cross_paper_recall", 0)
    return f"  {label:<35} {r:.3f}   {p:.3f}   {nd:.3f}   {mr:.3f}   {cp:.3f}"


# ── Chunk quality (no retrieval needed) ───────────────────────────────────────

def chunk_quality_metrics(chunks) -> Dict:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    token_lengths = [c.token_count for c in chunks]
    n             = len(chunks)

    # Redundancy: pairwise cosine on sample
    sample = chunks[:min(200, n)]
    try:
        vec   = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vec.fit_transform([c.text for c in sample])
        sims  = cosine_similarity(tfidf)
        np.fill_diagonal(sims, 0)
        redundancy_mean  = float(sims.max(axis=1).mean())
        high_redundancy  = int((sims.max(axis=1) > 0.92).sum())
    except Exception:
        redundancy_mean, high_redundancy = 0.0, 0

    fig_chunks       = sum(1 for c in chunks if getattr(c, "is_figure_chunk", False))
    entity_coverage  = sum(1 for c in chunks if len(c.entities) > 0) / max(n, 1)
    theme_coverage   = sum(1 for c in chunks if c.theme)              / max(n, 1)
    consensus_chunks = sum(1 for c in chunks if c.consensus_papers)
    conflict_chunks  = sum(1 for c in chunks if c.conflict_chunk_ids)

    return {
        "total_chunks":           n,
        "figure_table_chunks":    fig_chunks,
        "body_chunks":            n - fig_chunks,
        "mean_tokens":            float(np.mean(token_lengths)),
        "median_tokens":          float(np.median(token_lengths)),
        "std_tokens":             float(np.std(token_lengths)),
        "min_tokens":             int(np.min(token_lengths)),
        "max_tokens":             int(np.max(token_lengths)),
        "entity_coverage_pct":    round(entity_coverage  * 100, 1),
        "theme_coverage_pct":     round(theme_coverage   * 100, 1),
        "consensus_chunks":       consensus_chunks,
        "conflict_chunks":        conflict_chunks,
        "redundancy_mean_cosine": round(redundancy_mean, 4),
        "high_redundancy_chunks": high_redundancy,
        "section_distribution":   _section_dist(chunks),
    }


def _section_dist(chunks) -> Dict:
    dist: Dict[str, int] = {}
    for c in chunks:
        key      = c.section[:30]
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10])


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(agg: Dict, quality: Dict) -> None:
    K = sorted(agg.get("recall_at_k", {}).keys())
    k_display = 5 if 5 in K else (K[-1] if K else 5)

    print("\n" + "═" * 65)
    print("  RETRIEVAL EVALUATION REPORT")
    print("═" * 65)
    print(f"  Questions evaluated : {agg.get('n_total', 0)} "
          f"({agg.get('n_single',0)} single-paper, {agg.get('n_cross',0)} cross-paper)")
    print(f"  Mean chunks returned: {agg.get('mean_n_retrieved', 0):.1f} "
          f"(dynamic cutoff — not fixed K)")
    print()

    print(f"  {'Metric':<20} {'All':>8} {'Single':>8} {'Cross':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'MRR':<20} {agg.get('mrr',0):>8.4f} "
          f"{agg.get('mrr_single',0):>8.4f} {agg.get('mrr_cross',0):>8.4f}")
    print(f"  {'Cross-paper Recall':<20} {agg.get('cross_paper_recall',0):>8.4f}")
    print()

    print(f"  {'K':<6} {'Recall@K':>10} {'Precision@K':>12} {'NDCG@K':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*10}")
    for k in K:
        print(f"  {k:<6} "
              f"{agg['recall_at_k'][k]:>10.4f} "
              f"{agg['precision_at_k'][k]:>12.4f} "
              f"{agg['ndcg_at_k'][k]:>10.4f}")
    print()

    print("  CHUNK QUALITY")
    print(f"  {'─'*40}")
    print(f"  Total chunks        : {quality.get('total_chunks', 0)} "
          f"({quality.get('figure_table_chunks',0)} fig/table, "
          f"{quality.get('body_chunks',0)} body)")
    print(f"  Token range         : {quality.get('min_tokens',0)}–"
          f"{quality.get('max_tokens',0)} "
          f"(mean={quality.get('mean_tokens',0):.0f}, "
          f"σ={quality.get('std_tokens',0):.0f})")
    print(f"  Entity coverage     : {quality.get('entity_coverage_pct',0):.1f}%")
    print(f"  Theme coverage      : {quality.get('theme_coverage_pct',0):.1f}%")
    print(f"  Consensus chunks    : {quality.get('consensus_chunks',0)}")
    print(f"  Conflict chunks     : {quality.get('conflict_chunks',0)}")
    print(f"  Redundancy (cosine) : {quality.get('redundancy_mean_cosine',0):.4f}")
    print("═" * 65)
