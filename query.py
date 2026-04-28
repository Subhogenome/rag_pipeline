"""query.py — interactive RAG chat interface.

Usage:
    python query.py [--config config.yaml] [--query "your question"]
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

from config import PipelineConfig
from pipeline.indexer import MultiIndex
from pipeline.graph import ChunkGraph
from pipeline.retriever import retrieve
from pipeline.compressor import compress_context
from chat.azure_chat import chat_stream


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def load_artifacts(cfg: PipelineConfig):
    out = Path(cfg.output_dir)
    if not (out / "hnsw_chunk.bin").exists():
        print("❌  Index not found. Run `python main.py` first.")
        sys.exit(1)

    from pipeline.embedder import load_embedder
    embedder = load_embedder(str(out / "embedder.pkl"), cfg.embedding)
    index    = MultiIndex(cfg.hnsw, dims=cfg.embedding.dims, output_dir=str(out))
    index.load()
    graph    = ChunkGraph.load(str(out / "graph.pkl"))
    return embedder, index, graph


def answer_query(query: str, cfg: PipelineConfig, embedder, index, graph, history: list) -> str:
    print(f"\n🔍  Retrieving context...")
    results = retrieve(query, embedder, index, graph, cfg)

    if not results:
        return "No relevant context found."

    print(f"   → {len(results)} chunks retrieved")
    for i, r in enumerate(results, 1):
        sources_str = "+".join(set(r.sources))
        print(f"   {i}. [{sources_str}] {r.chunk.paper_title[:45]} | {r.chunk.section} "
              f"| theme: '{r.chunk.theme}' | conf: {r.chunk.confidence:.2f} "
              f"| score: {r.score:.4f}")

    # Compress context with LLMLingua
    print(f"\n📦  Compressing context (LLMLingua)...")
    chunk_texts = [
        f"[{r.chunk.paper_title[:50]} — {r.chunk.section}]\n{r.chunk.text}"
        for r in results
    ]
    compressed = compress_context(query, chunk_texts, cfg.llmlingua)

    # Stream response from Azure
    print(f"\n🤖  Azure AI (Llama-4-Maverick) response:\n{'─'*55}")
    full_response = ""
    for token in chat_stream(query, results, compressed, cfg.azure, history):
        print(token, end="", flush=True)
        full_response += token
    print(f"\n{'─'*55}")

    return full_response


def interactive_mode(cfg: PipelineConfig, embedder, index, graph) -> None:
    history = []
    print("\n🧬  Scientific RAG — Interactive Chat")
    print("   Papers indexed:", Path(cfg.output_dir).name)
    print("   Type 'quit' to exit, 'clear' to reset history\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "clear":
            history.clear()
            print("History cleared.")
            continue

        answer = answer_query(query, cfg, embedder, index, graph, history)
        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific RAG Query Interface")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--query",      default=None, help="Single query (non-interactive)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    setup_logging("WARNING")  # quiet in chat mode
    embedder, index, graph = load_artifacts(cfg)

    if args.query:
        answer_query(args.query, cfg, embedder, index, graph, [])
    else:
        interactive_mode(cfg, embedder, index, graph)


if __name__ == "__main__":
    main()
