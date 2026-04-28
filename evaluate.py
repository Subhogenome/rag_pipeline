"""
evaluate.py — Full evaluation with golden set validation + ablation.

Usage:
    python evaluate.py                     # full run
    python evaluate.py --regen-golden      # force regenerate golden set
    python evaluate.py --no-ablation       # skip ablation study
"""
from __future__ import annotations
import argparse, copy, json, logging, sys
from pathlib import Path

from config            import PipelineConfig, RetrievalConfig
from pipeline.embedder import load_embedder
from pipeline.indexer  import MultiIndex
from pipeline.graph    import ChunkGraph
from pipeline.retriever import retrieve
from evaluation.evaluator import (
    EvalQuestion, chunk_quality_metrics,
    evaluate_single, aggregate, print_report, ablation_row,
)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def _validate_golden(questions: list, index: MultiIndex) -> tuple[list, int]:
    """
    Check how many golden chunk_ids exist in current index.
    If <50% match, the golden set is stale (built against old index).
    Returns (valid_questions, match_pct).
    """
    all_ids   = {c.chunk_id for c in index.get_all_chunks()}
    matched   = 0
    total_refs = 0

    for q in questions:
        for cid in q.relevant_chunk_ids:
            total_refs += 1
            if cid in all_ids:
                matched += 1

    match_pct = int(matched / max(total_refs, 1) * 100)
    return questions, match_pct


def _load_golden(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    return [EvalQuestion(**q) for q in raw]


def _save_golden(questions: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump([q.__dict__ for q in questions], f, indent=2)


def _ablation_configs(base_cfg: RetrievalConfig):
    configs = []
    def _mod(**kwargs):
        c = copy.deepcopy(base_cfg)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c
    configs.append(("BM25 only",           _mod(dense_weight=0.0, graph_weight=0.0, bm25_weight=1.0)))
    configs.append(("Dense only",          _mod(bm25_weight=0.0,  graph_weight=0.0)))
    configs.append(("Dense + BM25",        _mod(graph_weight=0.0)))
    configs.append(("+ Graph walk",        _mod()))
    configs.append(("Full pipeline",       base_cfg))
    return configs


def run_eval(questions, embedder, index, graph, cfg) -> dict:
    results = []
    for q in questions:
        retrieved = retrieve(q.question, embedder, index, graph, cfg)
        er        = evaluate_single(q, retrieved, cfg.evaluation.k_values)
        results.append(er)
    return aggregate(results, cfg.evaluation.k_values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--golden",        default=None)
    parser.add_argument("--regen-golden",  action="store_true",
                        help="Force regenerate golden set even if file exists")
    parser.add_argument("--no-ablation",   action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    setup_logging(cfg.log_level)
    logger = logging.getLogger("evaluate")
    out    = Path(cfg.output_dir)

    if not (out / "hnsw_chunk.bin").exists():
        logger.error("Index not found. Run `python main.py` first.")
        sys.exit(1)

    # ── Load artifacts ────────────────────────────────────────────────────
    logger.info("Loading index artifacts...")
    embedder   = load_embedder(str(out / "embedder.pkl"), cfg.embedding)
    if embedder is None:
        from pipeline.embedder import build_embedder, embed_chunks, save_embedder
        logger.warning("Embedder needs rebuild — loading chunks and refitting...")
        tmp_index = MultiIndex(cfg.hnsw, dims=cfg.embedding.dims, output_dir=str(out))
        tmp_index.load()
        tmp_chunks = tmp_index.get_all_chunks()
        embedder = build_embedder(tmp_chunks, cfg.embedding)
        save_embedder(embedder, str(out / "embedder.pkl"))

    index      = MultiIndex(cfg.hnsw, dims=embedder.dims, output_dir=str(out))
    index.load()
    graph      = ChunkGraph.load(str(out / "graph.pkl"))
    all_chunks = index.get_all_chunks()

    logger.info(f"Index: {len(all_chunks)} chunks")

    # ── Chunk quality ─────────────────────────────────────────────────────
    logger.info("Computing chunk quality metrics...")
    quality = chunk_quality_metrics(all_chunks)

    # ── Golden QA — validate or regenerate ───────────────────────────────
    golden_path = args.golden or str(out / "golden_qa.json")
    needs_regen = args.regen_golden

    if Path(golden_path).exists() and not needs_regen:
        logger.info(f"Loading golden QA from {golden_path}")
        questions  = _load_golden(golden_path)
        _, match_pct = _validate_golden(questions, index)
        logger.info(f"Golden set chunk_id match: {match_pct}%")

        if match_pct < 50:
            logger.warning(
                f"Only {match_pct}% of golden chunk_ids exist in current index!\n"
                f"  The golden set was built against a different index.\n"
                f"  → Auto-regenerating golden set..."
            )
            needs_regen = True

    if needs_regen or not Path(golden_path).exists():
        logger.error(
            "Golden QA not found. Generate it first:\n"
            "  python create_golden.py --n 30\n"
            "Then re-run: python evaluate.py"
        )
        print_report({}, quality)
        return

    if not questions:
        logger.warning("No questions available.")
        print_report({}, quality)
        return

    logger.info(f"Evaluating {len(questions)} questions...")

    # ── Main eval ─────────────────────────────────────────────────────────
    main_agg = run_eval(questions, embedder, index, graph, cfg)
    print_report(main_agg, quality)

    # ── Ablation ──────────────────────────────────────────────────────────
    if not args.no_ablation:
        k = 5 if 5 in cfg.evaluation.k_values else cfg.evaluation.k_values[-1]
        logger.info(f"\nRunning ablation study (K={k})...")

        print(f"\n{'═'*68}")
        print(f"  ABLATION TABLE  (K={k})")
        print(f"  {'Component':<35} {'R@K':>5} {'P@K':>7} {'NDCG':>6} {'MRR':>6} {'XP-R':>6}")
        print(f"  {'─'*35} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*6}")

        ablation_results = {}
        for label, r_cfg in _ablation_configs(cfg.retrieval):
            cfg_copy           = copy.deepcopy(cfg)
            cfg_copy.retrieval = r_cfg
            agg                = run_eval(questions, embedder, index, graph, cfg_copy)
            ablation_results[label] = agg
            print(ablation_row(label, agg, k=k))

        print(f"{'═'*68}")

        report = {
            "main_metrics":  main_agg,
            "chunk_quality": quality,
            "ablation":      ablation_results,
        }
        report_path = out / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nFull report saved → {report_path}")

        # Key insight
        full = ablation_results.get("Full pipeline", main_agg)
        bm25 = ablation_results.get("BM25 only", {})
        if full and bm25:
            r_gain  = full.get("recall_at_k",{}).get(k,0)  - bm25.get("recall_at_k",{}).get(k,0)
            cp_gain = full.get("cross_paper_recall",0) - bm25.get("cross_paper_recall",0)
            mrr_gain = full.get("mrr",0) - bm25.get("mrr",0)
            print(f"\n  KEY GAINS vs BM25 baseline:")
            print(f"  Recall@{k}           : {r_gain:+.3f}")
            print(f"  MRR                : {mrr_gain:+.3f}")
            print(f"  Cross-paper Recall : {cp_gain:+.3f}\n")


if __name__ == "__main__":
    main()
