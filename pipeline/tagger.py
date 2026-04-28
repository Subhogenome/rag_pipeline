"""
pipeline/tagger.py — Azure OpenAI chunk tagging with tag-cache support.

If a cache file exists (output/tag_cache.json), tags are loaded from it
instead of calling Azure — saves ~90 minutes on rebuild.
"""
from __future__ import annotations
import json, logging, re, time
from pathlib import Path
from typing import List, Dict
from openai import OpenAI, RateLimitError
from config import AzureConfig
from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

TAG_PROMPT_TEMPLATE = (
    "You are a biomedical research annotator. Given a text chunk from a scientific paper, "
    "return a JSON object with EXACTLY these keys and NO other text:\n\n"
    '{{"theme": "<3-5 word topic label>", '
    '"summary": "<1 sentence: what this chunk claims or describes>", '
    '"entities": ["<gene/protein/pathway/concept>"], '
    '"hyde_question": "<the most natural question a researcher would ask that this chunk answers>", '
    '"chunk_type": "<one of: background|methods|results|conclusion|caption|other>"}}\n\n'
    "Paper: {paper_title}\nSection: {section}\nChunk:\n{text}"
)


def _azure_client(cfg: AzureConfig) -> OpenAI:
    return OpenAI(base_url=cfg.endpoint, api_key=cfg.api_key,
                  max_retries=3, timeout=60.0)


def _parse_tag_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$",         "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def _apply_tags(chunk: Chunk, tags: dict) -> None:
    chunk.theme         = tags.get("theme", "")
    chunk.summary       = tags.get("summary", "")
    chunk.entities      = tags.get("entities", []) or []
    chunk.hyde_question = tags.get("hyde_question", "")
    chunk.chunk_type    = tags.get("chunk_type", "other")


def _fallback_tag(chunk: Chunk) -> None:
    sl = chunk.section.lower()
    chunk.chunk_type = (
        "methods"    if "method"   in sl else
        "results"    if "result"   in sl else
        "conclusion" if ("discuss" in sl or "conclus" in sl) else
        "background" if ("abstract" in sl or "background" in sl or "introduc" in sl) else
        "other"
    )
    chunk.theme         = chunk.section
    chunk.summary       = chunk.text[:150] + "..."
    chunk.entities      = list(set(re.findall(
        r"\b[A-Z][A-Z0-9\u03b4\u03b1\u03b2\u03b3\-]{2,}\b", chunk.text)))[:8]
    chunk.hyde_question = f"What does {chunk.section} of '{chunk.paper_title[:40]}' say?"


# ── Tag cache ──────────────────────────────────────────────────────────────────

def _load_cache(cache_path: str) -> Dict[str, dict]:
    """Load existing tags from cache keyed by chunk_id."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        logger.info(f"Tag cache loaded: {len(data)} entries from {cache_path}")
        return data
    except Exception as e:
        logger.warning(f"Cache load failed ({e}), starting fresh")
        return {}


def _save_cache(cache: Dict[str, dict], cache_path: str) -> None:
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def export_tag_cache(chunks: List[Chunk], cache_path: str) -> None:
    """Export all chunk tags to cache file for reuse on rebuild."""
    cache = {}
    for c in chunks:
        if c.theme:
            cache[c.chunk_id] = {
                "theme":         c.theme,
                "summary":       c.summary,
                "entities":      c.entities,
                "hyde_question": c.hyde_question,
                "chunk_type":    c.chunk_type,
            }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Tag cache exported: {len(cache)} entries → {cache_path}")


# ── Token budget ───────────────────────────────────────────────────────────────

class _TokenBudget:
    def __init__(self, tpm_limit: int = 6000):  # default overridden by cfg.azure.tpm_limit
        self.tpm_limit = tpm_limit
        self._window   = 60.0
        self._usage: list = []

    def _purge_old(self):
        cutoff    = time.time() - self._window
        self._usage = [(t, n) for t, n in self._usage if t > cutoff]

    def tokens_in_window(self) -> int:
        self._purge_old()
        return sum(n for _, n in self._usage)

    def record(self, tokens: int):
        self._usage.append((time.time(), tokens))

    def wait_if_needed(self, about_to_use: int):
        while True:
            self._purge_old()
            used     = self.tokens_in_window()
            headroom = self.tpm_limit - used
            if headroom >= about_to_use:
                return
            oldest_ts = self._usage[0][0] if self._usage else time.time()
            wait      = (oldest_ts + self._window) - time.time() + 0.5
            if wait > 0:
                logger.info(f"  TPM budget {used}/{self.tpm_limit} — waiting {wait:.1f}s...")
                time.sleep(wait)


# ── Main entry point ───────────────────────────────────────────────────────────

def tag_chunks(
    chunks: List[Chunk],
    cfg: AzureConfig,
    cache_path: str = "output/tag_cache.json",
) -> List[Chunk]:
    """
    Tag chunks via Azure OpenAI with cache support.
    Chunks whose chunk_id exists in cache are tagged instantly from cache —
    no API call needed. Only genuinely new chunks call Azure.
    """
    cache = _load_cache(cache_path)

    # Apply cached tags first
    cached_count = 0
    to_tag = []
    for chunk in chunks:
        if chunk.chunk_id in cache:
            _apply_tags(chunk, cache[chunk.chunk_id])
            cached_count += 1
        else:
            to_tag.append(chunk)

    if cached_count:
        logger.info(f"  {cached_count} chunks loaded from tag cache")

    if not to_tag:
        logger.info("  All chunks served from cache — no API calls needed ✓")
        return chunks

    logger.info(f"  {len(to_tag)} new chunks need tagging via Azure AI [{cfg.deployment}] "
                f"(TPM cap={cfg.tpm_limit})...")

    # Tag remaining chunks via Azure
    try:
        client = _azure_client(cfg)
    except EnvironmentError as e:
        logger.warning(f"Azure credentials missing ({e}). Using fallback tags.")
        for c in to_tag:
            _fallback_tag(c)
        return chunks

    budget    = _TokenBudget(tpm_limit=cfg.tpm_limit)
    tagged    = 0
    start     = time.time()

    for i, chunk in enumerate(to_tag):
        estimated = min(len(chunk.text.split()) * 2, 500) + 150
        budget.wait_if_needed(estimated)

        prompt = TAG_PROMPT_TEMPLATE.format(
            paper_title=chunk.paper_title,
            section=chunk.section,
            text=chunk.text[:1500],
        )
        success = False
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=cfg.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=cfg.tag_max_tokens,
                    temperature=cfg.tag_temperature,
                    response_format={"type": "json_object"},
                )
                actual = resp.usage.total_tokens if resp.usage else estimated
                budget.record(actual)
                raw  = resp.choices[0].message.content or ""
                tags = _parse_tag_response(raw)
                if tags:
                    _apply_tags(chunk, tags)
                    cache[chunk.chunk_id] = tags
                    tagged  += 1
                    success  = True
                break
            except RateLimitError:
                logger.warning(f"  429 on chunk {i+1} — backing off 15s...")
                time.sleep(15)
            except Exception as e:
                es = str(e)
                if any(x in es.lower() for x in ("allowlist", "403", "401", "forbidden")):
                    logger.warning("Azure blocked — fallback for remaining chunks")
                    for remaining in to_tag[i:]:
                        _fallback_tag(remaining)
                    _save_cache(cache, cache_path)
                    return chunks
                if attempt == 0:
                    time.sleep(2)
                break

        if not success:
            _fallback_tag(chunk)

        # Save cache every 50 chunks so progress is never lost
        if (i + 1) % 50 == 0:
            _save_cache(cache, cache_path)
            elapsed  = time.time() - start
            rate     = (i + 1) / elapsed * 60
            eta      = (len(to_tag) - i - 1) / rate if rate > 0 else 0
            logger.info(f"  [{i+1}/{len(to_tag)}] tagged={tagged} "
                        f"rate={rate:.0f}/min ETA={eta:.1f}min")

    _save_cache(cache, cache_path)
    logger.info(f"  → {tagged} new + {cached_count} cached = {tagged+cached_count} total tagged")
    return chunks
