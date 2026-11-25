"""API HTTP. Lancement : uv run uvicorn rag.api:app --reload"""

from dataclasses import asdict
from functools import cache
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, StringConstraints

from rag import config
from rag.answer import answer
from rag.rerank import retrieve

app = FastAPI(title="RAG droit du travail")


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    top_k: int = Field(5, ge=1, le=20)


class SourceOut(BaseModel):
    id: str
    title: str
    url: str


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut]


def get_conn():
    with config.connect() as conn:
        yield conn


@cache
def get_embedder():
    return config.get_embedder()


@cache
def get_reranker():
    return config.get_reranker()


@cache
def get_llm():
    return config.get_llm()


@app.post("/ask", response_model=AnswerOut)
def ask(
    q: Question,
    conn=Depends(get_conn),
    embedder=Depends(get_embedder),
    reranker=Depends(get_reranker),
    llm=Depends(get_llm),
):
    results = retrieve(conn, q.question, embedder, reranker, top_k=q.top_k)
    return asdict(answer(q.question, results, llm))
