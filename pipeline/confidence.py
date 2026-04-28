"""pipeline/confidence.py — multi-factor confidence scoring for each chunk."""
from __future__ import annotations
import re
from typing import List
from pipeline.chunker import Chunk

# Assertion markers → high information density
ASSERTION_WORDS = re.compile(
    r"\b(significantly|demonstrate|show(?:ed|s)?|found|confirm|reveal|indicate|"
    r"suggest|establish|prove|demonstrate|observe|identify|detect|measure|quantify|"
    r"result(?:ed|s)?|conclude|determine|increase|decrease|inhibit|activate)\b",
    re.IGNORECASE,
)
# Hedge words → lower confidence
HEDGE_WORDS = re.compile(
    r"\b(may|might|could|possibly|presumably|likely|perhaps|unclear|unknown|"
    r"potentially|speculate|hypothesize|appear(?:s|ed)?|seem(?:s|ed)?)\b",
    re.IGNORECASE,
)
# Section weights (results/conclusions are most informative for RAG)
SECTION_WEIGHTS = {
    "results": 1.0,
    "conclusion": 0.95,
    "discussion": 0.85,
    "abstract": 0.80,
    "background": 0.55,
    "introduction": 0.55,
    "methods": 0.45,
    "references": 0.10,
    "preamble": 0.30,
    "other": 0.50,
}


def _entity_density(chunk: Chunk) -> float:
    """Named entity count / total words → information density proxy."""
    words = len(chunk.text.split())
    if words == 0:
        return 0.0
    # Count entities from LLM tagging + raw all-caps words
    raw_entities = re.findall(r"\b[A-Z][A-Z0-9δαβγ\-]{2,}\b", chunk.text)
    total_entities = len(set(chunk.entities) | set(raw_entities))
    return min(1.0, total_entities / max(words * 0.05, 1))  # cap at 1


def _assertion_strength(chunk: Chunk) -> float:
    """Ratio of assertion words to hedge words → how definitive the chunk is."""
    text = chunk.text
    assertions = len(ASSERTION_WORDS.findall(text))
    hedges = len(HEDGE_WORDS.findall(text))
    total = assertions + hedges
    if total == 0:
        return 0.5
    return min(1.0, assertions / total)


def _section_weight(chunk: Chunk) -> float:
    section_lower = chunk.section.lower()
    for key, weight in SECTION_WEIGHTS.items():
        if key in section_lower:
            return weight
    return 0.50


def _length_score(chunk: Chunk) -> float:
    """Prefer chunks in 150–500 token sweet spot."""
    t = chunk.token_count
    if t < 50:
        return 0.3
    if t < 150:
        return 0.6
    if t <= 500:
        return 1.0
    return max(0.5, 1.0 - (t - 500) / 500)


def score_confidence(chunk: Chunk) -> float:
    ed = _entity_density(chunk)
    ass = _assertion_strength(chunk)
    sw = _section_weight(chunk)
    ls = _length_score(chunk)
    # Weighted combination
    score = 0.25 * ed + 0.30 * ass + 0.30 * sw + 0.15 * ls
    return round(min(1.0, max(0.0, score)), 4)


def score_all(chunks: List[Chunk]) -> List[Chunk]:
    for chunk in chunks:
        chunk.confidence = score_confidence(chunk)
    return chunks
