"""pipeline/embedder.py — sentence-transformers with LSA fallback + HyDE dual-index."""
from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

from config import EmbeddingConfig
from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Try sentence-transformers first ──────────────────────────────────────────
_ST_MODEL = None
_ST_AVAILABLE = False

def _try_load_st(model_name: str, fallback_model: str = "all-MiniLM-L6-v2"):
    global _ST_MODEL, _ST_AVAILABLE
    if _ST_MODEL is not None:
        return _ST_AVAILABLE
    for attempt_model in [model_name, fallback_model]:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL     = SentenceTransformer(attempt_model)
            _ST_AVAILABLE = True
            dim = _ST_MODEL.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformers [{attempt_model}] dim={dim}")
            if attempt_model != model_name:
                logger.info(f"  (primary model {model_name} unavailable, using fallback)")
            return True
        except Exception as e:
            logger.info(f"Model [{attempt_model}] unavailable ({str(e)[:60]})")
    logger.info("All ST models unavailable — using LSA fallback")
    _ST_AVAILABLE = False
    return False


# ── LSA (TF-IDF + SVD) fallback ──────────────────────────────────────────────
class _LSAEmbedder:
    """
    TF-IDF + TruncatedSVD.
    Trade-off vs sentence-transformers:
      + No download, runs fully offline, fits in <1s on 1000 chunks
      - Weaker semantic similarity, no cross-lingual, misses paraphrases
    """
    def __init__(self, dims: int):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.dims   = dims
        self._tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)
        self._svd   = TruncatedSVD(n_components=dims, random_state=42)
        self._fitted = False

    def fit(self, texts: List[str]) -> "_LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        # SVD n_components must be < n_samples; cap gracefully
        actual_dims = min(self.dims, len(texts) - 1, 30000)
        if actual_dims != self.dims:
            logger.info(f"LSA dims capped: {self.dims} → {actual_dims} (corpus too small)")
            self.dims = actual_dims
            self._svd = TruncatedSVD(n_components=actual_dims, random_state=42)
        logger.info(f"Fitting LSA embedder on {len(texts)} texts (dims={self.dims})...")
        tfidf = self._tfidf.fit_transform(texts)
        self._svd.fit(tfidf)
        self._fitted = True
        ev = self._svd.explained_variance_ratio_.sum()
        logger.info(f"  LSA explained variance: {ev:.3f}")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        from sklearn.preprocessing import normalize
        tfidf = self._tfidf.transform(texts)
        vecs  = self._svd.transform(tfidf)
        return normalize(vecs, norm="l2")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self._tfidf, "svd": self._svd,
                         "dims": self.dims, "backend": "lsa"}, f)

    @classmethod
    def load(cls, path: str) -> "_LSAEmbedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        emb = cls(state["dims"])
        emb._tfidf   = state["tfidf"]
        emb._svd     = state["svd"]
        emb._fitted  = True
        return emb


# ── Sentence-Transformers wrapper ─────────────────────────────────────────────
class _STEmbedder:
    """
    sentence-transformers (all-MiniLM-L6-v2 or BAAI/bge-small-en-v1.5).
    Trade-off vs LSA:
      + Much better semantic similarity, handles paraphrases, biomedical text
      - Requires downloading model (~90MB), needs internet on first run
    """
    def __init__(self, model):
        self._model  = model
        self.dims    = model.get_sentence_embedding_dimension()
        self._fitted = True   # ST needs no fitting

    def fit(self, texts: List[str]) -> "_STEmbedder":
        return self   # no-op

    def encode(self, texts: List[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            vecs = self._model.encode(texts, batch_size=32, show_progress_bar=False,
                                      normalize_embeddings=True)
        return np.array(vecs)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model_name": self._model.model_card_data.model_name
                         if hasattr(self._model, "model_card_data") else "unknown",
                         "dims": self.dims, "backend": "sentence_transformers"}, f)
        # NOTE: model weights are cached by sentence-transformers separately;
        # we only save the config so load() knows which model to re-load.

    @classmethod
    def load(cls, path: str, model_name: str) -> "_STEmbedder":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return cls(model)


# ── Public API ────────────────────────────────────────────────────────────────

def build_embedder(chunks: List[Chunk], cfg: EmbeddingConfig):
    """
    Build and fit the embedder. Prefers sentence-transformers; falls back to LSA.
    Corpus includes both chunk text AND hyde questions for a richer vocabulary.
    """
    corpus = [c.text for c in chunks]
    hyde   = [c.hyde_question for c in chunks if c.hyde_question]

    fallback = getattr(cfg, "fallback_model", "all-MiniLM-L6-v2")  # from config.yaml
    if _try_load_st(cfg.model_name, fallback):
        emb = _STEmbedder(_ST_MODEL)
        emb.fit(corpus + hyde)   # no-op for ST but keeps interface uniform
        return emb
    else:
        emb = _LSAEmbedder(cfg.dims)
        emb.fit(corpus + hyde)
        return emb


def embed_chunks(chunks: List[Chunk], embedder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (chunk_vecs, hyde_vecs) — both shape (N, D).
    HyDE vectors embed the hypothetical question instead of chunk text,
    which improves query-time matching because user queries are question-shaped.
    """
    texts  = [c.text for c in chunks]
    hydes  = [c.hyde_question if c.hyde_question else c.text for c in chunks]

    chunk_vecs = embedder.encode(texts)
    hyde_vecs  = embedder.encode(hydes)

    backend = "sentence-transformers" if isinstance(embedder, _STEmbedder) else "LSA"
    logger.info(f"Embedded {len(chunks)} chunks via {backend} (dim={chunk_vecs.shape[1]})")
    return chunk_vecs, hyde_vecs


def save_embedder(embedder, path: str) -> None:
    if isinstance(embedder, _STEmbedder):
        # Save model_name so load_embedder can verify it matches at query time
        model_name = getattr(embedder._model, "model_name_or_path",
                     getattr(embedder._model, "_model_name", "unknown"))
        with open(path, "wb") as f:
            pickle.dump({
                "backend":    "sentence_transformers",
                "model_name": model_name,
                "dims":       embedder.dims,
            }, f)
        logger.info(f"Embedder saved → {path} (model={model_name}, dims={embedder.dims})")
    else:
        embedder.save(path)
        logger.info(f"Embedder saved → {path} (LSA, dims={embedder.dims})")


def load_embedder(path: str, cfg: EmbeddingConfig):
    with open(path, "rb") as f:
        meta = pickle.load(f)

    if meta.get("backend") == "sentence_transformers":
        saved_model = meta.get("model_name", "unknown")
        saved_dims  = meta.get("dims", "unknown")

        # CRITICAL: query embedding model must match index embedding model
        # If they differ, cosine similarity is meaningless (different vector spaces)
        if saved_model != "unknown" and saved_model != cfg.model_name:
            logger.warning(
                "Embedding model MISMATCH detected! "
                f"Index={saved_model}(dim={saved_dims}) vs config={cfg.model_name}. "
                f"Loading saved model to keep vectors consistent. "
                "To switch models, rebuild with `python main.py`."
            )
            # Load the model that was used at index time, not what config says
            if _try_load_st(saved_model, cfg.fallback_model):
                logger.info(f"Embedder loaded ← {path} (model={saved_model}, dims={saved_dims})")
                return _STEmbedder(_ST_MODEL)
        else:
            if _try_load_st(cfg.model_name, getattr(cfg, "fallback_model", "all-MiniLM-L6-v2")):
                logger.info(f"Embedder loaded ← {path} (model={cfg.model_name}, dims={saved_dims})")
                return _STEmbedder(_ST_MODEL)

        logger.warning("sentence-transformers unavailable at load time — rebuilding LSA")
        return None

    else:
        emb = _LSAEmbedder(meta["dims"])
        with open(path, "rb") as f:
            state = pickle.load(f)
        emb._tfidf  = state["tfidf"]
        emb._svd    = state["svd"]
        emb._fitted = True
        logger.info(f"Embedder loaded ← {path} (LSA, dims={meta['dims']})")
        logger.info(f"LSA embedder loaded ← {path}")
        return emb
