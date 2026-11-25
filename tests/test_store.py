from rag.chunking import Chunk
from rag.store import index_chunks, search

CHUNKS = [
    Chunk("F1", "Congés payés", "https://u/F1", "Congés payés > Apprenti\nun apprenti a droit à cinq semaines de congés"),
    Chunk("F2", "Rupture conventionnelle", "https://u/F2", "Rupture conventionnelle\nindemnité de rupture et délai de rétractation"),
    Chunk("F3", "Période d'essai", "https://u/F3", "Période d'essai\ndurée maximale de la période d'essai en CDI"),
]


def test_search_returns_closest_chunks_first(conn, embedder):
    index_chunks(conn, CHUNKS, embedder, dim=embedder.dim)
    results = search(conn, "délai de rétractation rupture conventionnelle", embedder, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.fiche_id == "F2"
    assert results[0].score >= results[1].score


def test_reindexing_replaces_previous_chunks(conn, embedder):
    index_chunks(conn, CHUNKS, embedder, dim=embedder.dim)
    index_chunks(conn, CHUNKS[:1], embedder, dim=embedder.dim)
    results = search(conn, "rupture", embedder, top_k=5)
    assert [r.chunk.fiche_id for r in results] == ["F1"]


def test_search_keeps_chunk_metadata(conn, embedder):
    index_chunks(conn, CHUNKS, embedder, dim=embedder.dim)
    top = search(conn, "apprenti congés", embedder, top_k=1)[0].chunk
    assert top == CHUNKS[0]
