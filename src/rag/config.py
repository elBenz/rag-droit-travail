"""Clients réels (Voyage, Claude, PostgreSQL), configurés par le fichier .env."""

import os

import psycopg
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

load_dotenv()

EMBEDDING_MODEL = "voyage-3.5"
EMBEDDING_DIM = 1024
RERANK_MODEL = "rerank-2.5"
LLM_MODEL = "claude-opus-5-5"


def get_embedder() -> VoyageAIEmbeddings:
    # Voyage distingue documents et questions (input_type) : la bibliothèque le gère
    # via embed_documents / embed_query.
    return VoyageAIEmbeddings(model=EMBEDDING_MODEL, batch_size=128)


def get_reranker() -> VoyageAIRerank:
    return VoyageAIRerank(model=RERANK_MODEL, top_k=None)


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(model=LLM_MODEL, max_tokens=4096)


def connect() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"])
