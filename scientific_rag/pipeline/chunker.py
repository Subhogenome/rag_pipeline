"""pipeline/chunker.py — Citation-aware, section-aware semantic chunking."""
from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

from pipeline.pdf_parser import ParsedPaper, Section
from config import ChunkingConfig

logger = logging.getLogger(__name__)

# ── Citation patterns — never split on these boundaries ───────────────────────
# Covers [12], [12,14], [12–14], (Smith et al., 2020), (Smith, 2020)
CITATION_TAIL = re.compile(
    r"(\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]"   # numeric [12] [12,14] [12-14]
    r"|\(\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*,\s*\d{4}\s*\)"  # (Smith et al., 2020)
    r"|\(\s*[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*,\s*\d{4}\s*\)"  # (Smith and Jones, 2020)
    r")[\.,:;]?\s*$"  # optional trailing punctuation after citation
)

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spacy en_core_web_sm")
    except OSError:
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
        logger.info("Using blank spacy + sentencizer")
    return _nlp


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _split_sentences(text: str) -> List[str]:
    nlp = _get_nlp()
    doc = nlp(text[:50_000])
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _sentence_similarity_matrix(sentences: List[str]) -> np.ndarray:
    if len(sentences) < 2:
        return np.array([1.0])
    try:
        vec   = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vec.fit_transform(sentences)
        sims  = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(tfidf[i], tfidf[i + 1])[0][0]
            sims.append(float(sim))
        return np.array(sims)
    except Exception:
        return np.ones(len(sentences) - 1)


def _ends_with_citation(sentence: str) -> bool:
    """Return True if sentence ends with a citation marker — never split here."""
    return bool(CITATION_TAIL.search(sentence.strip()))


def _find_semantic_break_points(sentences: List[str], threshold: float) -> List[int]:
    """
    Return indices AFTER which a new chunk should start.
    Rules:
      - Cosine similarity drop below (mean - threshold) → break
      - BUT never break if the previous sentence ends with a citation
        (the next sentence is the continuation of that claim)
    """
    sims     = _sentence_similarity_matrix(sentences)
    mean_sim = sims.mean() if len(sims) else 0.5
    cutoff   = max(0.05, mean_sim - threshold)

    breaks = []
    for i, sim in enumerate(sims):
        if sim < cutoff:
            # Citation guard: don't break after a sentence that ends with [12]
            if _ends_with_citation(sentences[i]):
                logger.debug(f"  Citation guard prevented break after: '{sentences[i][-60:]}'")
                continue
            breaks.append(i + 1)
    return breaks


@dataclass
class Chunk:
    chunk_id:   str
    text:       str
    section:    str
    paper_title: str
    paper_path: str
    page_start: int
    token_count: int
    chunk_index: int
    is_figure_chunk: bool = False   # True if from a figure/table caption
    figure_id:       str  = ""      # "Figure 3"
    # LLM-filled later
    theme:        str  = ""
    summary:      str  = ""
    entities:     List[str] = field(default_factory=list)
    hyde_question: str = ""
    chunk_type:   str  = ""
    confidence:   float = 0.5
    consensus_papers:   List[str] = field(default_factory=list)
    conflict_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _make_chunks_from_sentences(
    sentences: List[str],
    section: Section,
    cfg: ChunkingConfig,
    paper_path: str,
    chunk_offset: int,
) -> List[Chunk]:
    breaks = set(_find_semantic_break_points(sentences, cfg.semantic_split_threshold))

    chunks: List[Chunk] = []
    current_sents: List[str] = []
    current_tokens = 0

    def _flush(sents: List[str]) -> None:
        text = " ".join(sents).strip()
        if _count_tokens(text) < cfg.min_tokens:
            return
        idx      = chunk_offset + len(chunks)
        chunk_id = f"{section.paper_title[:30]}_{section.name[:20]}_{idx}".replace(" ", "_")
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            section=section.name,
            paper_title=section.paper_title,
            paper_path=paper_path,
            page_start=section.page_start,
            token_count=_count_tokens(text),
            chunk_index=idx,
            is_figure_chunk=section.is_caption,
            figure_id=section.figure_id,
        ))

    for i, sent in enumerate(sentences):
        sent_tokens  = _count_tokens(sent)
        would_exceed = (current_tokens + sent_tokens) > cfg.max_tokens
        is_break     = i in breaks

        if (would_exceed or is_break) and current_sents:
            # Citation guard at flush boundary: if the LAST sentence of the
            # current buffer ends with a citation, pull one more sentence in
            # before flushing — keeps claim + citation together.
            if _ends_with_citation(current_sents[-1]) and i < len(sentences) - 1:
                current_sents.append(sent)
                current_tokens += sent_tokens
                _flush(current_sents)
                current_sents  = []
                current_tokens = 0
                continue

            _flush(current_sents)
            # Overlap: carry last few sentences forward
            overlap_budget = cfg.overlap_tokens
            overlap_sents: List[str] = []
            for s in reversed(current_sents):
                if _count_tokens(" ".join(overlap_sents)) + _count_tokens(s) > overlap_budget:
                    break
                overlap_sents.insert(0, s)
            current_sents  = overlap_sents
            current_tokens = _count_tokens(" ".join(current_sents))

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        _flush(current_sents)

    return chunks


def chunk_paper(paper: ParsedPaper, cfg: ChunkingConfig) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    for section in paper.sections:
        if section.is_reference and not cfg.include_references:
            continue
        if not section.text.strip():
            continue

        text      = re.sub(r"\s+", " ", section.text).strip()
        sentences = _split_sentences(text)
        if not sentences:
            continue

        # Figure/table caption chunks: keep as a single chunk (they're short)
        # unless they exceed max_tokens
        if section.is_caption:
            tok = _count_tokens(text)
            if tok >= cfg.min_tokens:
                idx      = len(all_chunks)
                chunk_id = f"{section.paper_title[:30]}_{section.figure_id[:15]}_{idx}".replace(" ", "_")
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    section=section.name,
                    paper_title=section.paper_title,
                    paper_path=paper.path,
                    page_start=section.page_start,
                    token_count=tok,
                    chunk_index=idx,
                    is_figure_chunk=True,
                    figure_id=section.figure_id,
                ))
            continue

        new_chunks = _make_chunks_from_sentences(
            sentences=sentences,
            section=section,
            cfg=cfg,
            paper_path=paper.path,
            chunk_offset=len(all_chunks),
        )
        all_chunks.extend(new_chunks)

    fig_chunks = sum(1 for c in all_chunks if c.is_figure_chunk)
    logger.info(f"  {paper.title[:50]}: {len(all_chunks)} chunks "
                f"({fig_chunks} figure/table) across {len(paper.sections)} sections")
    return all_chunks
