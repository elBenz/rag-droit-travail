from langchain_core.documents import Document

from rag.chunking import Chunk
from rag.rerank import rerank, retrieve
from rag.store import Result, index_chunks


class ReverseReranker:
    """Faux reranker : inverse l'ordre et note du plus pertinent (1.0) au moins pertinent."""

    def compress_documents(self, documents, query):
        out = []
        for i, doc in enumerate(reversed(documents)):
            d = Document(doc.page_content, metadata=dict(doc.metadata))
            d.metadata["relevance_score"] = 1.0 - i / 10
            out.append(d)
        return out


def result(fiche_id, score):
    return Result(Chunk(fiche_id, "t", "u", f"texte {fiche_id}"), score)


def test_rerank_reorders_keeps_top_n_and_uses_reranker_scores():
    results = [result("F1", 0.9), result("F2", 0.8), result("F3", 0.7)]
    reranked = rerank("q", results, ReverseReranker(), top_n=2)
    assert [r.chunk.fiche_id for r in reranked] == ["F3", "F2"]
    assert [r.score for r in reranked] == [1.0, 0.9]


def test_rerank_of_nothing_is_nothing():
    assert rerank("q", [], ReverseReranker(), top_n=5) == []


def test_retrieve_reranks_a_wider_candidate_pool(conn, embedder):
    chunks = [Chunk(f"F{i}", "t", "u", f"congés payés salarié numéro {i}") for i in range(10)]
    index_chunks(conn, chunks, embedder, dim=embedder.dim)
    plain = retrieve(conn, "congés payés", embedder, top_k=2)
    reranked = retrieve(conn, "congés payés", embedder, ReverseReranker(), top_k=2, candidates=10)
    assert len(plain) == len(reranked) == 2
    # Le faux reranker remonte les derniers des 10 candidats, absents du top 2 de la recherche seule.
    assert not {r.chunk.fiche_id for r in plain} & {r.chunk.fiche_id for r in reranked}
