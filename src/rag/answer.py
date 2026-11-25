"""Génération de la réponse par le LLM à partir des extraits retrouvés, avec citations."""

import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from rag.store import Result

REFUSAL = "Je ne trouve pas cette information dans les fiches."

SYSTEM = f"""Tu réponds à des questions sur le droit du travail français à partir d'extraits \
de fiches officielles Service-Public, numérotés [1], [2]…

Règles :
- Appuie-toi uniquement sur les extraits fournis, jamais sur tes connaissances propres.
- Après chaque affirmation, cite le ou les extraits qui la justifient, par exemple [2] ou [1][3].
- Si les extraits ne permettent pas de répondre, réponds exactement : « {REFUSAL} »
- Si la réponse dépend de la situation (type de contrat, convention collective…), dis-le.
- Réponds en français, de façon concise."""


@dataclass
class Source:
    id: str
    title: str
    url: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]


def build_prompt(question: str, results: list[Result]) -> str:
    extracts = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(results, start=1))
    return f"Extraits :\n\n{extracts}\n\nQuestion : {question}"


def cited_sources(text: str, results: list[Result]) -> list[Source]:
    """Fiches réellement citées dans la réponse, dédoublonnées, dans l'ordre d'apparition."""
    sources: dict[str, Source] = {}
    for n in re.findall(r"\[(\d+)\]", text):
        i = int(n) - 1
        if 0 <= i < len(results):
            c = results[i].chunk
            sources.setdefault(c.fiche_id, Source(c.fiche_id, c.title, c.url))
    return list(sources.values())


def answer(question: str, results: list[Result], llm: BaseChatModel) -> Answer:
    if not results:
        return Answer(REFUSAL, [])
    message = llm.invoke([("system", SYSTEM), ("human", build_prompt(question, results))])
    text = message.text.strip()
    if not text:  # ex. refus du modèle pour raison de sécurité : pas de réponse exploitable
        return Answer(REFUSAL, [])
    return Answer(text, cited_sources(text, results))
