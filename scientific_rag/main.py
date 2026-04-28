"""main.py — build the full RAG index from PDFs."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

from config          import PipelineConfig
from pipeline.pdf_parser import parse_pdf
from pipeline.chunker    import chunk_paper
from pipeline.tagger     import tag_chunks, export_tag_cache
from pipeline.confidence import score_all
from pipeline.embedder   import build_embedder, embed_chunks, save_embedder
from pipeline.indexer    import MultiIndex
from pipeline.graph      import ChunkGraph


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cfg: PipelineConfig) -> None:
    logger = logging.getLogger("main")
    logger.info("═" * 60)
    logger.info("  Scientific RAG Pipeline — Index Build")
    logger.info("═" * 60)

    pdf_dir = Path(cfg.pdf_dir)
    pdfs    = [p for p in pdf_dir.glob("*.pdf")
               if "assignment" not in p.name.lower()]
    if not pdfs:
        logger.error(f"No PDFs found in {pdf_dir}")
        sys.exit(1)
    logger.info(f"Found {len(pdfs)} PDFs")

    # ── Step 1: Parse ─────────────────────────────────────────────────────
    logger.info("\n[1/6] Parsing PDFs...")
    all_chunks, total_fig = [], 0
    for pdf_path in sorted(pdfs):
        paper  = parse_pdf(str(pdf_path),
                           include_captions=cfg.chunking.include_captions,
                           include_references=cfg.chunking.include_references)
        chunks = chunk_paper(paper, cfg.chunking)
        fig_n  = sum(1 for c in chunks if c.is_figure_chunk)
        total_fig += fig_n
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)} "
                f"({total_fig} figure/table, {len(all_chunks)-total_fig} body)")

    # ── Step 2: Tag (uses cache if available) ──────────────────────────────
    logger.info("\n[2/6] Tagging chunks with Azure AI (cache-aware)...")
    cache_path = str(Path(cfg.output_dir) / "tag_cache.json")
    all_chunks = tag_chunks(all_chunks, cfg.azure, cache_path=cache_path)

    # Export cache so future rebuilds are instant
    export_tag_cache(all_chunks, cache_path)

    # ── Step 3: Confidence scoring ─────────────────────────────────────────
    logger.info("\n[3/6] Scoring chunk confidence...")
    all_chunks = score_all(all_chunks)
    conf_vals  = [c.confidence for c in all_chunks]
    logger.info(f"  Confidence — mean: {sum(conf_vals)/len(conf_vals):.3f}  "
                f"min: {min(conf_vals):.3f}  max: {max(conf_vals):.3f}")

    # ── Step 4: Embed ──────────────────────────────────────────────────────
    logger.info("\n[4/6] Building embedder + embedding chunks...")
    embedder = build_embedder(all_chunks, cfg.embedding)
    chunk_vecs, hyde_vecs = embed_chunks(all_chunks, embedder)
    save_embedder(embedder, str(Path(cfg.output_dir) / "embedder.pkl"))

    # ── Step 5: Build indexes ──────────────────────────────────────────────
    logger.info("\n[5/6] Building HNSW + BM25 + SQLite indexes...")
    index = MultiIndex(cfg.hnsw, dims=chunk_vecs.shape[1], output_dir=cfg.output_dir)
    index.build(all_chunks, chunk_vecs, hyde_vecs)
    index.save()

    # ── Step 6: Knowledge graph ────────────────────────────────────────────
    logger.info("\n[6/6] Building knowledge graph (contradiction + consensus)...")
    graph = ChunkGraph()
    graph.build(all_chunks)

    contradiction_pairs = graph.contradiction_pairs(all_chunks)
    chunk_map           = {c.chunk_id: c for c in all_chunks}
    for cid_a, cid_b, shared in contradiction_pairs:
        if cid_a in chunk_map: chunk_map[cid_a].conflict_chunk_ids.append(cid_b)
        if cid_b in chunk_map: chunk_map[cid_b].conflict_chunk_ids.append(cid_a)

    consensus_count = graph.detect_consensus(all_chunks)

    for c in all_chunks:
        if c.conflict_chunk_ids or c.consensus_papers:
            index._upsert_chunk(c)
    index._conn.commit()

    graph.save(str(Path(cfg.output_dir) / "graph.pkl"))

    # ── Summary ────────────────────────────────────────────────────────────
    papers           = list({c.paper_title for c in all_chunks})
    consensus_chunks = sum(1 for c in all_chunks if c.consensus_papers)

    summary = {
        "total_chunks":        len(all_chunks),
        "figure_table_chunks": total_fig,
        "body_chunks":         len(all_chunks) - total_fig,
        "total_papers":        len(papers),
        "papers":              papers,
        "contradiction_pairs": len(contradiction_pairs),
        "consensus_pairs":     consensus_count,
        "consensus_chunks":    consensus_chunks,
        "embedding_dim":       int(chunk_vecs.shape[1]),
        "output_dir":          cfg.output_dir,
    }
    with open(Path(cfg.output_dir) / "index_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "═" * 60)
    logger.info("  ✓ Index built successfully")
    logger.info(f"  Papers              : {summary['total_papers']}")
    logger.info(f"  Total chunks        : {summary['total_chunks']}")
    logger.info(f"    ↳ Body chunks     : {summary['body_chunks']}")
    logger.info(f"    ↳ Figure/table    : {summary['figure_table_chunks']}")
    logger.info(f"  Embedding dim       : {summary['embedding_dim']}")
    logger.info(f"  Contradiction pairs : {summary['contradiction_pairs']}")
    logger.info(f"  Consensus pairs     : {summary['consensus_pairs']}")
    logger.info(f"  Consensus chunks    : {summary['consensus_chunks']}")
    logger.info(f"  Tag cache           : {cache_path}")
    logger.info("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scientific RAG Index")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--pdf_dir",    default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)
    if args.pdf_dir:    cfg.pdf_dir    = args.pdf_dir
    if args.output_dir: cfg.output_dir = args.output_dir

    setup_logging(cfg.log_level)
    run(cfg)


if __name__ == "__main__":
    main()
