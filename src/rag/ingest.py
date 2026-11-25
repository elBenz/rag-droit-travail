"""Ingestion : XML → sections → chunks → embeddings → pgvector.

Usage : uv run python -m rag.ingest data/raw/vosdroits
"""

import sys
from pathlib import Path

from rag.chunking import chunk_fiche
from rag.config import EMBEDDING_DIM, connect, get_embedder
from rag.parsing import load_fiches
from rag.store import index_chunks


def main(directory: str) -> None:
    fiches = load_fiches(Path(directory))
    chunks = [c for f in fiches for c in chunk_fiche(f)]
    print(f"{len(fiches)} fiches, {len(chunks)} chunks : calcul des embeddings…")
    with connect() as conn:
        index_chunks(conn, chunks, get_embedder(), dim=EMBEDDING_DIM)
    print("Index prêt.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
