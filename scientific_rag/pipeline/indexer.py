"""pipeline/indexer.py — dual HNSW + BM25 + SQLite metadata store."""
from __future__ import annotations
import json, logging, pickle, sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import hnswlib
from rank_bm25 import BM25Okapi
from config import HNSWConfig
from pipeline.chunker import Chunk
# LSAEmbedder import removed — embedder is now backend-agnostic

logger = logging.getLogger(__name__)


class MultiIndex:
    """
    Holds:
      - HNSW index on chunk-text vectors      (dense retrieval)
      - HNSW index on HyDE-question vectors   (question-oriented retrieval)
      - BM25 index on theme + summary text    (sparse keyword retrieval)
      - SQLite metadata store
    """

    def __init__(self, cfg: HNSWConfig, dims: int, output_dir: str):
        self.cfg = cfg
        self.dims = dims
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._hnsw_chunk: Optional[hnswlib.Index] = None
        self._hnsw_hyde:  Optional[hnswlib.Index] = None
        self._bm25:       Optional[BM25Okapi] = None
        self._bm25_ids:   List[str] = []
        self._db_path     = self.output_dir / "chunks.db"
        self._conn:       Optional[sqlite3.Connection] = None

        self._init_db()

    # ── SQLite ────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                data     TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.commit()

    def _upsert_chunk(self, chunk: Chunk) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, data) VALUES (?, ?)",
            (chunk.chunk_id, json.dumps(chunk.to_dict())),
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        row = self._conn.execute(
            "SELECT data FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return Chunk.from_dict(json.loads(row["data"])) if row else None

    def get_chunks_by_ids(self, ids: List[str]) -> List[Chunk]:
        result = []
        for cid in ids:
            c = self.get_chunk(cid)
            if c:
                result.append(c)
        return result

    def get_all_chunks(self) -> List[Chunk]:
        rows = self._conn.execute("SELECT data FROM chunks").fetchall()
        return [Chunk.from_dict(json.loads(r["data"])) for r in rows]

    def chunk_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ── HNSW helpers ─────────────────────────────────────────────────────────
    def _new_hnsw(self) -> hnswlib.Index:
        idx = hnswlib.Index(space=self.cfg.space, dim=self.dims)
        idx.init_index(
            max_elements=100_000,
            ef_construction=self.cfg.ef_construction,
            M=self.cfg.M,
        )
        idx.set_ef(self.cfg.ef_search)
        return idx

    # ── Build ────────────────────────────────────────────────────────────────
    def build(
        self,
        chunks: List[Chunk],
        chunk_vecs: np.ndarray,
        hyde_vecs: np.ndarray,
    ) -> None:
        logger.info(f"Building multi-index for {len(chunks)} chunks...")

        # Store metadata
        self._conn.execute("DELETE FROM chunks")
        for chunk in chunks:
            self._upsert_chunk(chunk)
        self._conn.commit()

        # Integer labels for HNSW (position in chunks list)
        labels = np.arange(len(chunks), dtype=np.int32)

        # Chunk-text HNSW
        self._hnsw_chunk = self._new_hnsw()
        self._hnsw_chunk.add_items(chunk_vecs, labels)

        # HyDE HNSW
        self._hnsw_hyde = self._new_hnsw()
        self._hnsw_hyde.add_items(hyde_vecs, labels)

        # BM25 on theme + summary + entities (rich text signal)
        bm25_texts = []
        self._bm25_ids = []
        for c in chunks:
            rich_text = f"{c.theme} {c.summary} {' '.join(c.entities)} {c.section}"
            bm25_texts.append(rich_text.lower().split())
            self._bm25_ids.append(c.chunk_id)
        self._bm25 = BM25Okapi(bm25_texts)

        # Persist chunk_id ↔ integer mapping
        id_map = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._conn.execute(
            "INSERT OR REPLACE INTO index_meta VALUES ('id_map', ?)",
            (json.dumps(id_map),),
        )
        # Save actual dims so load() works regardless of embedding backend
        self._conn.execute(
            "INSERT OR REPLACE INTO index_meta VALUES ('dims', ?)",
            (str(chunk_vecs.shape[1]),),
        )
        self._conn.commit()
        logger.info("  Multi-index built ✓")

    # ── Query ─────────────────────────────────────────────────────────────────
    def search_dense(
        self, query_vec: np.ndarray, k: int, use_hyde: bool = False
    ) -> List[Tuple[str, float]]:
        """Returns [(chunk_id, distance)] from HNSW search."""
        idx = self._hnsw_hyde if use_hyde else self._hnsw_chunk
        if idx is None:
            return []
        k = min(k, idx.get_current_count())
        labels, distances = idx.knn_query(query_vec.reshape(1, -1), k=k)
        all_chunks = self.get_all_chunks()
        id_map_row = self._conn.execute(
            "SELECT value FROM index_meta WHERE key='id_map'"
        ).fetchone()
        if not id_map_row:
            return []
        id_map = json.loads(id_map_row["value"])
        inv_map = {v: k for k, v in id_map.items()}
        results = []
        for label, dist in zip(labels[0], distances[0]):
            cid = inv_map.get(int(label))
            if cid:
                results.append((cid, float(dist)))
        return results

    def search_bm25(self, query: str, k: int) -> List[Tuple[str, float]]:
        """BM25 on theme+summary+entity text."""
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_k = np.argsort(scores)[::-1][:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top_k]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        self._hnsw_chunk.save_index(str(self.output_dir / "hnsw_chunk.bin"))
        self._hnsw_hyde.save_index(str(self.output_dir  / "hnsw_hyde.bin"))
        with open(self.output_dir / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._bm25_ids}, f)
        logger.info(f"Index saved → {self.output_dir}")

    def load(self) -> "MultiIndex":
        # Read actual dims from saved meta (handles ST vs LSA dimension difference)
        row = self._conn.execute("SELECT value FROM index_meta WHERE key='dims'").fetchone()
        if row:
            self.dims = int(row["value"])
        # HNSW chunk
        self._hnsw_chunk = hnswlib.Index(space=self.cfg.space, dim=self.dims)
        self._hnsw_chunk.load_index(str(self.output_dir / "hnsw_chunk.bin"), max_elements=100_000)
        self._hnsw_chunk.set_ef(self.cfg.ef_search)

        # HNSW hyde
        self._hnsw_hyde = hnswlib.Index(space=self.cfg.space, dim=self.dims)
        self._hnsw_hyde.load_index(str(self.output_dir / "hnsw_hyde.bin"), max_elements=100_000)
        self._hnsw_hyde.set_ef(self.cfg.ef_search)

        # BM25
        with open(self.output_dir / "bm25.pkl", "rb") as f:
            state = pickle.load(f)
        self._bm25    = state["bm25"]
        self._bm25_ids = state["ids"]

        logger.info(f"Index loaded ← {self.output_dir} ({self.chunk_count()} chunks)")
        return self

    def is_built(self) -> bool:
        return (self.output_dir / "hnsw_chunk.bin").exists()
