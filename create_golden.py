"""
create_golden.py — Generate ground-truth evaluation dataset from your indexed chunks.

Run from project root:
    python create_golden.py
    python create_golden.py --n 30 --output output/golden_qa.json
"""
import sys, json, re, random, logging, argparse
from pathlib import Path
from collections import defaultdict, Counter

# ── must be run from project root ────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_golden")


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group()) if m else {}
        except Exception:
            return {}


def _call(client, deployment: str, prompt: str, max_tokens: int = 400) -> dict:
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return _parse_json(resp.choices[0].message.content or "{}")


# ── Question generators ────────────────────────────────────────────────────────

def gen_factual(chunks, client, deployment, n):
    """Single-chunk factual questions from high-confidence results/conclusion chunks."""
    pool = [c for c in chunks
            if c.chunk_type in ("results", "conclusion", "background")
            and c.confidence > 0.55
            and len(c.text) > 80]
    random.shuffle(pool)

    questions = []
    print(f"  Factual pool: {len(pool)} candidates")

    for c in pool:
        if len(questions) >= n:
            break
        prompt = (
            f"You are creating evaluation questions for a biomedical RAG system.\n"
            f"Given this research paper chunk, write ONE specific question that is "
            f"directly answered by this chunk. The question must require reading "
            f"this chunk — not answerable from general knowledge alone.\n\n"
            f"Paper: {c.paper_title[:60]}\n"
            f"Section: {c.section}\n"
            f"Text: {c.text[:500]}\n\n"
            f'Respond ONLY as JSON: {{"question": "...", "answer": "..."}}'
        )
        try:
            data = _call(client, deployment, prompt)
            q    = (data.get("question") or "").strip()
            ans  = (data.get("answer")   or "").strip()
            if len(q) < 15:
                continue
            questions.append({
                "question":           q,
                "relevant_chunk_ids": [c.chunk_id],
                "relevant_papers":    [c.paper_title],
                "answer":             ans,
                "difficulty":         "single",
            })
            print(f"  ✓ [{len(questions)}/{n}] {q[:80]}")
        except Exception as e:
            logger.debug(f"Factual gen error: {e}")

    return questions


def gen_cross_paper(chunks, client, deployment, n):
    """Cross-paper questions from entity-linked chunk pairs."""
    entity_map = defaultdict(list)
    for c in chunks:
        for ent in (c.entities or [])[:4]:
            if len(ent) > 3:
                entity_map[ent.upper()].append(c)

    # Build cross-paper pairs
    pairs = []
    for entity, ent_chunks in entity_map.items():
        by_paper = defaultdict(list)
        for c in ent_chunks:
            by_paper[c.paper_title].append(c)
        papers = list(by_paper.keys())
        if len(papers) < 2:
            continue
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                ca = random.choice(by_paper[papers[i]])
                cb = random.choice(by_paper[papers[j]])
                pairs.append((entity, ca, cb))

    random.shuffle(pairs)
    questions = []
    print(f"  Cross-paper pool: {len(pairs)} pairs")

    for entity, ca, cb in pairs:
        if len(questions) >= n:
            break
        prompt = (
            f"Two chunks from DIFFERENT research papers both mention: {entity}\n\n"
            f"Write ONE question that requires BOTH chunks to answer fully — "
            f"comparing, contrasting, or synthesising information from both papers.\n\n"
            f"Paper A — {ca.paper_title[:50]} [{ca.section}]:\n{ca.text[:350]}\n\n"
            f"Paper B — {cb.paper_title[:50]} [{cb.section}]:\n{cb.text[:350]}\n\n"
            f'Respond ONLY as JSON: {{"question": "...", "answer": "..."}}'
        )
        try:
            data = _call(client, deployment, prompt)
            q    = (data.get("question") or "").strip()
            ans  = (data.get("answer")   or "").strip()
            if len(q) < 15:
                continue
            questions.append({
                "question":           q,
                "relevant_chunk_ids": [ca.chunk_id, cb.chunk_id],
                "relevant_papers":    list({ca.paper_title, cb.paper_title}),
                "answer":             ans,
                "difficulty":         "cross_paper",
            })
            print(f"  ✓ [{len(questions)}/{n}] {q[:80]}")
        except Exception as e:
            logger.debug(f"Cross-paper gen error: {e}")

    return questions


def gen_figure(chunks, client, deployment, n):
    """Questions specifically answered by figure/table chunks."""
    fig_chunks = [c for c in chunks if getattr(c, "is_figure_chunk", False)]
    random.shuffle(fig_chunks)
    questions = []
    print(f"  Figure pool: {len(fig_chunks)} figure/table chunks")

    if not fig_chunks:
        print("  ⚠ No figure chunks in index — skipping figure questions")
        print("    (rebuild index with updated pdf_parser.py to get figure chunks)")
        return []

    for c in fig_chunks:
        if len(questions) >= n:
            break
        prompt = (
            f"Given this figure or table caption from a research paper, "
            f"write ONE specific question that is answered by what this "
            f"figure SHOWS or MEASURES.\n\n"
            f"Paper: {c.paper_title[:60]}\n"
            f"Figure: {getattr(c,'figure_id','')}\n"
            f"Caption: {c.text[:400]}\n\n"
            f'Respond ONLY as JSON: {{"question": "...", "answer": "..."}}'
        )
        try:
            data = _call(client, deployment, prompt)
            q    = (data.get("question") or "").strip()
            ans  = (data.get("answer")   or "").strip()
            if len(q) < 15:
                continue
            questions.append({
                "question":           q,
                "relevant_chunk_ids": [c.chunk_id],
                "relevant_papers":    [c.paper_title],
                "answer":             ans,
                "difficulty":         "figure",
            })
            print(f"  ✓ [{len(questions)}/{n}] {q[:80]}")
        except Exception as e:
            logger.debug(f"Figure gen error: {e}")

    return questions


def gen_consensus(chunks, client, deployment, n):
    """Questions requiring evidence confirmed across multiple papers."""
    cons_chunks = [c for c in chunks
                   if getattr(c, "consensus_papers", [])
                   and c.chunk_type in ("results", "conclusion")
                   and c.confidence > 0.55]
    random.shuffle(cons_chunks)
    questions = []
    print(f"  Consensus pool: {len(cons_chunks)} consensus-tagged chunks")

    if not cons_chunks:
        print("  ⚠ No consensus chunks in index — skipping consensus questions")
        print("    (rebuild index with updated graph.py to get consensus detection)")
        return []

    for c in cons_chunks:
        if len(questions) >= n:
            break
        prompt = (
            f"This research finding is confirmed by multiple papers.\n"
            f"Write ONE question where the ideal answer explicitly cites "
            f"multiple papers as converging evidence "
            f"(e.g. 'What evidence across studies supports...', "
            f"'How consistently across papers...', 'What do multiple studies show...').\n\n"
            f"Paper: {c.paper_title[:50]}\n"
            f"Also confirmed by: {', '.join(c.consensus_papers[:3])}\n"
            f"Text: {c.text[:400]}\n\n"
            f'Respond ONLY as JSON: {{"question": "...", "answer": "..."}}'
        )
        try:
            data = _call(client, deployment, prompt)
            q    = (data.get("question") or "").strip()
            ans  = (data.get("answer")   or "").strip()
            if len(q) < 15:
                continue
            all_papers = list({c.paper_title} | set(c.consensus_papers[:2]))
            questions.append({
                "question":           q,
                "relevant_chunk_ids": [c.chunk_id],
                "relevant_papers":    all_papers,
                "answer":             ans,
                "difficulty":         "consensus",
            })
            print(f"  ✓ [{len(questions)}/{n}] {q[:80]}")
        except Exception as e:
            logger.debug(f"Consensus gen error: {e}")

    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate golden evaluation dataset")
    parser.add_argument("--n",      type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    from config import PipelineConfig
    cfg = PipelineConfig.load(args.config)
    print(f"\nConfig loaded. Output dir: {cfg.output_dir}")

    # ── Load index ────────────────────────────────────────────────────────
    out = Path(cfg.output_dir)
    if not (out / "hnsw_chunk.bin").exists():
        print("ERROR: Index not found. Run `python main.py` first.")
        sys.exit(1)

    from pipeline.indexer import MultiIndex
    index  = MultiIndex(cfg.hnsw, dims=cfg.embedding.dims, output_dir=str(out))
    index.load()
    chunks = index.get_all_chunks()
    print(f"Loaded {len(chunks)} chunks")

    chunk_type_counts = Counter(c.chunk_type for c in chunks)
    print(f"Chunk types: {dict(chunk_type_counts)}")
    print(f"Figure chunks: {sum(1 for c in chunks if getattr(c,'is_figure_chunk',False))}")
    print(f"Consensus chunks: {sum(1 for c in chunks if getattr(c,'consensus_papers',[]))}")

    # ── Azure client ──────────────────────────────────────────────────────
    from openai import OpenAI
    client     = OpenAI(base_url=cfg.azure.endpoint, api_key=cfg.azure.api_key)
    deployment = cfg.azure.deployment
    print(f"Azure deployment: {deployment}\n")

    # ── Generate questions ────────────────────────────────────────────────
    n = args.n
    n_factual   = int(n * 0.40)
    n_cross     = int(n * 0.25)
    n_figure    = int(n * 0.20)
    n_consensus = n - n_factual - n_cross - n_figure

    print(f"Generating {n} questions: {n_factual} factual | {n_cross} cross-paper | "
          f"{n_figure} figure | {n_consensus} consensus\n")

    all_questions = []

    print(f"[1/4] Factual questions (target: {n_factual})...")
    all_questions += gen_factual(chunks, client, deployment, n_factual)

    print(f"\n[2/4] Cross-paper questions (target: {n_cross})...")
    all_questions += gen_cross_paper(chunks, client, deployment, n_cross)

    print(f"\n[3/4] Figure questions (target: {n_figure})...")
    all_questions += gen_figure(chunks, client, deployment, n_figure)

    print(f"\n[4/4] Consensus questions (target: {n_consensus})...")
    all_questions += gen_consensus(chunks, client, deployment, n_consensus)

    random.shuffle(all_questions)

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = args.output or str(out / "golden_qa.json")
    with open(output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    diff_counts = Counter(q["difficulty"] for q in all_questions)
    print(f"\n{'═'*55}")
    print(f"  GOLDEN DATASET COMPLETE")
    print(f"{'═'*55}")
    print(f"  Total questions : {len(all_questions)}")
    for d, cnt in sorted(diff_counts.items()):
        print(f"  {d:<20}: {cnt}")
    print(f"  Saved to        : {output_path}")
    print(f"{'═'*55}")
    print(f"\nSample questions:")
    for q in all_questions[:5]:
        print(f"  [{q['difficulty']}] {q['question'][:75]}")
    print(f"\nRun `python evaluate.py` to compute retrieval metrics.")


if __name__ == "__main__":
    main()
