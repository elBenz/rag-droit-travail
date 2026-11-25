from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag import api
from rag.chunking import Chunk
from rag.store import index_chunks


def make_client(conn, embedder, response):
    index_chunks(conn, [Chunk("F1", "Congés payés", "https://u/F1", "congés payés 2,5 jours par mois")],
                 embedder, dim=embedder.dim)
    api.app.dependency_overrides = {
        api.get_conn: lambda: conn,
        api.get_embedder: lambda: embedder,
        api.get_reranker: lambda: None,
        api.get_llm: lambda: FakeListChatModel(responses=[response]),
    }
    return TestClient(api.app)


def test_ask_returns_answer_and_sources(conn, embedder):
    client = make_client(conn, embedder, "2,5 jours par mois [1].")
    res = client.post("/ask", json={"question": "Combien de jours de congés payés ?"})
    assert res.status_code == 200
    assert res.json() == {
        "answer": "2,5 jours par mois [1].",
        "sources": [{"id": "F1", "title": "Congés payés", "url": "https://u/F1"}],
    }


def test_blank_question_is_rejected(conn, embedder):
    client = make_client(conn, embedder, "")
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_top_k_is_bounded(conn, embedder):
    client = make_client(conn, embedder, "")
    assert client.post("/ask", json={"question": "q", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "q", "top_k": 50}).status_code == 422
