"""pipeline/pdf_parser.py — Figure/table as own chunks + improved section detection."""
from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import fitz

logger = logging.getLogger(__name__)

SECTION_PATTERNS = re.compile(
    r"^(abstract|background|introduction|methods?|materials?\s+and\s+methods?|"
    r"results?|discussion|conclusions?|acknowledgements?|references?|"
    r"supplementary|appendix|ethics|data\s+availability|"
    r"author\s+contributions?|competing\s+interests?|funding|"
    r"experimental\s+procedures?|statistical\s+analysis)[\s:\.\d]*$",
    re.IGNORECASE,
)
REFERENCE_HEADER = re.compile(r"^references?[\s:\.]*$", re.IGNORECASE)
# Matches "Figure 3.", "Fig. 2", "Table 1." at the start of a line
CAPTION_START    = re.compile(r"^(Fig(?:ure)?\.?\s*\d+[A-Za-z]?|Table\s*\d+)[\.:\s–-]", re.IGNORECASE)

# ── Noise sections — stripped before chunking ────────────────────────────────
# These contain file listings, competing interests, funding boilerplate,
# author bios — none answer scientific questions, all hurt retrieval precision.
NOISE_SECTIONS = re.compile(
    r"^("
    r"additional\s+(files?|information|data|material|supplementary)"
    r"|supplementary\s+(files?|data|material|information|methods?|tables?|figures?)"
    r"|competing\s+interests?"
    r"|authors?\s+contributions?"
    r"|funding\s+information"
    r"|data\s+availability"
    r"|ethics\s+(approval|statement|declaration)"
    r"|consent\s+to\s+publish"
    r"|acknowledgements?"
    r"|abbreviations?"
    r"|publisher.?s?\s+note"
    r"|peer\s+review"
    r")\s*[:.\d]*$",
    re.IGNORECASE,
)



@dataclass
class TextBlock:
    text: str
    font_size: float
    is_bold: bool
    page: int
    x0: float
    y0: float


@dataclass
class Section:
    name: str
    text: str
    page_start: int
    is_reference: bool = False
    is_caption:   bool = False   # True = this section IS a figure/table chunk
    figure_id:    str  = ""      # "Figure 3" or "Table 2"
    paper_title:  str  = ""


@dataclass
class ParsedPaper:
    title:    str
    path:     str
    sections: List[Section] = field(default_factory=list)
    abstract: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def _detect_columns(blocks: list, page_width: float) -> bool:
    if not blocks:
        return False
    mid   = page_width / 2
    left  = sum(1 for b in blocks if b.get("bbox", [0])[2] < mid * 1.1)
    right = sum(1 for b in blocks if b.get("bbox", [0])[0] > mid * 0.9)
    return left > 3 and right > 3


def _sort_blocks_2col(blocks: list, page_width: float) -> list:
    mid   = page_width / 2
    left  = sorted([b for b in blocks if b.get("bbox", [0])[0] < mid],  key=lambda b: b["bbox"][1])
    right = sorted([b for b in blocks if b.get("bbox", [0])[0] >= mid], key=lambda b: b["bbox"][1])
    return left + right


def _extract_text_blocks(doc: fitz.Document) -> List[TextBlock]:
    all_blocks: List[TextBlock] = []
    for page_idx, page in enumerate(doc):
        raw     = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        two_col = _detect_columns(raw.get("blocks", []), page.rect.width)
        blocks  = raw.get("blocks", [])
        if two_col:
            blocks = _sort_blocks_2col(blocks, page.rect.width)
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                if not text or len(text) < 3:
                    continue
                font_size = max(s.get("size", 10) for s in spans)
                is_bold   = any(
                    "bold" in s.get("font", "").lower() or s.get("flags", 0) & 16
                    for s in spans
                )
                bbox = block["bbox"]
                all_blocks.append(TextBlock(
                    text=text, font_size=font_size, is_bold=is_bold,
                    page=page_idx + 1, x0=bbox[0], y0=bbox[1],
                ))
    return all_blocks


def _is_header(block: TextBlock, body_size: float) -> bool:
    text = block.text.strip()
    if len(text) > 120:
        return False
    if SECTION_PATTERNS.match(text):
        return True
    if re.match(r"^\d+[\.\s]+[A-Z][a-z]", text) and len(text) < 80:
        return True
    size_boost = block.font_size > body_size * 1.05
    return (size_boost or block.is_bold) and len(text) < 80 and text[0].isupper()


def _estimate_body_font_size(blocks: List[TextBlock]) -> float:
    from collections import Counter
    sizes = [round(b.font_size, 1) for b in blocks if b.font_size > 6]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _extract_title(doc: fitz.Document) -> str:
    if len(doc) == 0:
        return "Unknown"
    page    = doc[0]
    raw     = page.get_text("dict")
    max_sz, title = 0.0, ""
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span["text"].strip()
                sz  = span.get("size", 0)
                if sz > max_sz and 10 < len(txt) < 300:
                    max_sz, title = sz, txt
    return title or Path(doc.name).stem


def parse_pdf(
    path: str,
    include_captions:   bool = True,
    include_references: bool = False,
) -> ParsedPaper:
    doc   = fitz.open(path)
    title = _extract_title(doc)
    logger.info(f"Parsing: {title} ({len(doc)} pages)")

    blocks    = _extract_text_blocks(doc)
    body_size = _estimate_body_font_size(blocks)

    sections: List[Section] = []
    current   = Section(name="PREAMBLE", text="", page_start=1, paper_title=title)
    in_refs   = False

    # Buffer for collecting multi-line captions
    caption_buf: str  = ""
    caption_id:  str  = ""
    caption_pg:  int  = 1

    def _flush_caption():
        nonlocal caption_buf, caption_id
        if caption_buf.strip() and include_captions:
            sections.append(Section(
                name=f"Caption: {caption_id}",
                text=caption_buf.strip(),
                page_start=caption_pg,
                is_caption=True,
                figure_id=caption_id,
                paper_title=title,
            ))
        caption_buf = ""
        caption_id  = ""

    for block in blocks:
        text = block.text.strip()

        # ── Reference section ─────────────────────────────────────────────
        if REFERENCE_HEADER.match(text):
            if current.text.strip():
                sections.append(current)
            _flush_caption()
            in_refs  = True
            current  = Section(name="References", text="", page_start=block.page,
                               is_reference=True, paper_title=title)
            continue

        if in_refs and not include_references:
            continue

        # ── Noise sections — skip entirely ────────────────────────────────
        # Additional files, competing interests, acknowledgements, etc.
        # are boilerplate that hurts retrieval precision without adding
        # scientific information.
        if _is_header(block, body_size) and NOISE_SECTIONS.match(text):
            if current.text.strip():
                sections.append(current)
            _flush_caption()
            # Open a noise section — content will be discarded
            current = Section(name=f"[NOISE]{text}", text="",
                              page_start=block.page, paper_title=title)
            continue

        # Skip content if we are inside a noise section
        if current.name.startswith("[NOISE]"):
            continue

        # ── Figure / table caption ────────────────────────────────────────
        cap_match = CAPTION_START.match(text)
        if cap_match:
            _flush_caption()                      # close previous caption if open
            caption_id  = cap_match.group(1).strip()
            caption_pg  = block.page
            caption_buf = text
            continue

        # If we're inside a caption, keep accumulating until next header/caption
        if caption_buf:
            # A new section header or very short line ends the caption
            if _is_header(block, body_size) or len(text) < 20:
                _flush_caption()
                # Fall through to header handling below
            else:
                caption_buf += " " + text
                continue

        # ── Section header ────────────────────────────────────────────────
        if _is_header(block, body_size):
            if current.text.strip():
                sections.append(current)
            current = Section(name=text, text="", page_start=block.page, paper_title=title)
        else:
            current.text += " " + text

    # Flush anything remaining
    _flush_caption()
    if current.text.strip():
        sections.append(current)

    abstract = ""
    for s in sections:
        if "abstract" in s.name.lower():
            abstract = s.text.strip()
            break

    # Count noise sections that were built (they have empty text and [NOISE] prefix)
    # Also track via a separate counter we maintain in the loop
    caption_count = sum(1 for s in sections if s.is_caption)
    # Noise sections are stripped during parsing (content skipped), so count
    # the ones that ended up with [NOISE] prefix (they have empty text)
    all_sections_incl_noise = sections  # already filtered by _flush
    clean_sections = [s for s in sections if not s.name.startswith("[NOISE]")]
    noise_count    = len(sections) - len(clean_sections)
    logger.info(
        f"  → {len(clean_sections)} sections "
        f"({caption_count} figure/table, {noise_count} noise stripped), "
        f"abstract={'yes' if abstract else 'no'}"
    )
    paper = ParsedPaper(title=title, path=path, sections=clean_sections, abstract=abstract)
    return paper
    return ParsedPaper(title=title, path=path, sections=sections, abstract=abstract)
