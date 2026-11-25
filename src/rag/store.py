"""Index vectoriel dans PostgreSQL + pgvector."""

from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings
from pgvector.psycopg import register_vector
from psycopg import Connection

from rag.chunking import Chunk

BATCH_SIZE = 128  # nombre de textes envoyés par appel à l'API d'embeddings


@dataclass
class Result:
    chunk: Chunk
    score: float  # similarité cosinus, 1 = identique


def index_chunks(conn: Connection, chunks: list[Chunk], embedder: Embeddings, dim: int = 1024) -> None:
    """Remplace tout l'index par `chunks`."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        f"""CREATE TABLE chunks (
            id serial PRIMARY KEY,
            fiche_id text NOT NULL,
            title text NOT NULL,
            url text NOT NULL,
            text text NOT NULL,
            embedding vector({dim}) NOT NULL
        )"""
    )
    with conn.cursor() as cur:
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            vectors = embedder.embed_documents([c.text for c in batch])
            cur.executemany(
                "INSERT INTO chunks (fiche_id, title, url, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                [(c.fiche_id, c.title, c.url, c.text, np.array(v)) for c, v in zip(batch, vectors)],
            )
    # Index HNSW : recherche approximative des plus proches voisins, rapide même sur beaucoup de vecteurs.
    conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.commit()


def search(conn: Connection, question: str, embedder: Embeddings, top_k: int = 5) -> list[Result]:
    register_vector(conn)
    query = np.array(embedder.embed_query(question))
    # <=> est la distance cosinus de pgvector ; similarité = 1 - distance.
    rows = conn.execute(
        """SELECT fiche_id, title, url, text, 1 - (embedding <=> %s) AS score
           FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
        (query, query, top_k),
    ).fetchall()
    return [Result(Chunk(*row[:4]), float(row[4])) for row in rows]
