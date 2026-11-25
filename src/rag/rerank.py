"""Reranking : la recherche vectorielle ramène large, un cross-encoder réordonne finement.

L'embedding compare deux vecteurs calculés séparément (rapide, mais grossier). Le reranker lit la
question et chaque candidat ensemble et note leur pertinence : plus précis, mais trop lent pour tout
le corpus. D'où le schéma : `candidates` extraits par la recherche, puis `top_k` après reranking.
"""

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from psycopg import Connection

from rag.store import Result, search


def rerank(question: str, results: list[Result], reranker: BaseDocumentCompressor, top_n: int) -> list[Result]:
    if not results:
        return []
    docs = [Document(r.chunk.text, metadata={"i": i}) for i, r in enumerate(results)]
    ranked = reranker.compress_documents(docs, question)[:top_n]
    return [Result(results[d.metadata["i"]].chunk, float(d.metadata["relevance_score"])) for d in ranked]


def retrieve(
    conn: Connection,
    question: str,
    embedder: Embeddings,
    reranker: BaseDocumentCompressor | None = None,
    top_k: int = 5,
    candidates: int = 30,
) -> list[Result]:
    if reranker is None:
        return search(conn, question, embedder, top_k=top_k)
    return rerank(question, search(conn, question, embedder, top_k=candidates), reranker, top_k)
