"""chat/azure_chat.py — Azure AI Foundry (Llama-4-Maverick) RAG chat."""
from __future__ import annotations
import logging
from typing import List, Iterator
from openai import OpenAI
from config import AzureConfig
from pipeline.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _build_context(results: List[RetrievedChunk], compressed_context: str) -> str:
    """Format retrieved chunks with metadata headers for the LLM."""
    if compressed_context:
        return compressed_context
    parts = []
    for i, r in enumerate(results, 1):
        c = r.chunk
        header = (
            f"[Chunk {i} | Paper: {c.paper_title[:60]} | "
            f"Section: {c.section} | Theme: {c.theme} | "
            f"Confidence: {c.confidence:.2f} | Sources: {','.join(r.sources)}]"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n" + "\n\n---\n\n".join(parts)


def _get_client(cfg: AzureConfig) -> OpenAI:
    """Build OpenAI-compatible client pointing at Azure AI Foundry."""
    return OpenAI(
        base_url=cfg.endpoint,
        api_key=cfg.api_key,
    )


def chat(
    query: str,
    results: List[RetrievedChunk],
    compressed_context: str,
    cfg: AzureConfig,
    chat_history: list | None = None,
) -> str:
    """Single-turn RAG chat. Returns full response string."""
    client = _get_client(cfg)
    context = _build_context(results, compressed_context)

    messages = [{"role": "system", "content": cfg.system_prompt}]
    if chat_history:
        messages.extend(chat_history[-6:])  # Keep last 3 turns

    user_message = f"""Context from research papers:
{context}

Question: {query}

Answer based only on the context above. Cite paper titles for key claims."""

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=cfg.deployment,
            messages=messages,
            max_tokens=cfg.chat_max_tokens,
            temperature=cfg.chat_temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer
    except Exception as e:
        logger.error(f"Azure chat error: {e}")
        return f"[Error calling Azure endpoint: {e}]"


def chat_stream(
    query: str,
    results: List[RetrievedChunk],
    compressed_context: str,
    cfg: AzureConfig,
    chat_history: list | None = None,
) -> Iterator[str]:
    """Streaming RAG chat. Yields text chunks as they arrive."""
    client = _get_client(cfg)
    context = _build_context(results, compressed_context)

    messages = [{"role": "system", "content": cfg.system_prompt}]
    if chat_history:
        messages.extend(chat_history[-6:])

    user_message = f"""Context from research papers:
{context}

Question: {query}

Answer based only on the context above. Cite paper titles for key claims."""

    messages.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=cfg.deployment,
            messages=messages,
            max_tokens=cfg.chat_max_tokens,
            temperature=cfg.chat_temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        logger.error(f"Azure stream error: {e}")
        yield f"[Error calling Azure endpoint: {e}]"
