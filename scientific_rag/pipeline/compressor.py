"""
pipeline/compressor.py — Context compression before sending to LLM.

PRIMARY   → LLMLingua2 (microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)
            BERT-based, ~700MB, downloads automatically on first run.
            Much lighter than original LLMLingua (which needed Llama-2-7b).

FALLBACK  → Extractive TF-IDF (no model, no download, runs in milliseconds)
            Used automatically if LLMLingua2 is unavailable or fails.

Config:
    llmlingua:
      use_llmlingua2: true    # set false to force extractive
      target_token: 400
"""
from __future__ import annotations
import logging, re
from typing import List

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


# ── LLMLingua2 ────────────────────────────────────────────────────────────────

_lingua2       = None
_lingua2_tried = False


def _get_lingua2():
    global _lingua2, _lingua2_tried
    if _lingua2_tried:
        return _lingua2
    _lingua2_tried = True
    try:
        from llmlingua import PromptCompressor
        logger.info("Loading LLMLingua2 model (first run: ~700MB download)...")
        _lingua2 = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        logger.info("LLMLingua2 loaded ✓")
    except Exception as e:
        logger.warning(
            f"LLMLingua2 unavailable ({str(e)[:80]})\n"
            f"  → Falling back to extractive TF-IDF compression.\n"
            f"  → To fix: pip install llmlingua accelerate"
        )
        _lingua2 = None
    return _lingua2


# ── Extractive TF-IDF fallback ────────────────────────────────────────────────

def _extractive_compress(query: str, context: str, target_tokens: int) -> str:
    """
    Score every sentence by TF-IDF cosine similarity to the query.
    Keep highest-scoring sentences in original order within token budget.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    if not sentences:
        return context

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec    = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf  = vec.fit_transform([query] + sentences)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except Exception:
        words = context.split()
        return " ".join(words[:target_tokens])

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    kept   = set()
    budget = target_tokens

    for idx in ranked:
        toks = _count_tokens(sentences[idx])
        if toks <= budget:
            kept.add(idx)
            budget -= toks
        if budget <= 0:
            break

    return " ".join(s for i, s in enumerate(sentences) if i in kept)


# ── Public API ────────────────────────────────────────────────────────────────

def compress_context(query: str, chunks_text: List[str], cfg) -> str:
    """
    Compress retrieved chunk texts to cfg.target_token budget.
    Tries LLMLingua2 first; falls back to extractive TF-IDF silently.
    """
    if not chunks_text:
        return ""

    full_context   = "\n\n---\n\n".join(chunks_text)
    current_tokens = _count_tokens(full_context)

    if current_tokens <= cfg.target_token:
        return full_context

    # ── Path 1: LLMLingua2 ───────────────────────────────────────────────
    if getattr(cfg, "use_llmlingua2", True):
        lingua = _get_lingua2()
        if lingua is not None:
            try:
                result = lingua.compress_prompt(
                    context=chunks_text,
                    instruction="",
                    question=query,
                    target_token=cfg.target_token,
                    rank_method="longllmlingua",
                    force_tokens=["\n", ".", "?", "!"],
                )
                compressed = result.get("compressed_prompt", full_context)
                after      = _count_tokens(compressed)
                logger.info(
                    f"LLMLingua2: {current_tokens} → {after} tokens "
                    f"({current_tokens / max(after, 1):.1f}x compression)"
                )
                return compressed
            except Exception as e:
                logger.warning(f"LLMLingua2 compress failed ({str(e)[:80]}) — using extractive")

    # ── Path 2: Extractive TF-IDF ────────────────────────────────────────
    compressed = _extractive_compress(query, full_context, cfg.target_token)
    after      = _count_tokens(compressed)
    logger.info(f"Extractive: {current_tokens} → {after} tokens")
    return compressed
