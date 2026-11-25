"""Évaluation du pipeline sur eval/questions.jsonl.

Usage : uv run python -m rag.evaluate [top_k] [--rerank]

- Recherche : hit@k (une fiche attendue est dans les k extraits) et MRR (rang moyen inverse).
- Réponse : éléments attendus présents, et au moins une fiche attendue citée.
- Hors corpus : la réponse doit être le refus.
"""

import json
import sys
from pathlib import Path

from rag.answer import REFUSAL, answer
from rag.config import connect, get_embedder, get_llm, get_reranker
from rag.rerank import retrieve

QUESTIONS = Path(__file__).parents[2] / "eval" / "questions.jsonl"


def reciprocal_rank(retrieved_ids: list[str], expected: set[str]) -> float:
    for rank, fiche_id in enumerate(retrieved_ids, start=1):
        if fiche_id in expected:
            return 1 / rank
    return 0.0


def _normalize(text: str) -> str:
    return " ".join(text.replace(" ", " ").replace(" ", " ").lower().split())


def contains_all(text: str, expected: list[str]) -> bool:
    return all(_normalize(e) in _normalize(text) for e in expected)


def main(top_k: int = 5, use_reranker: bool = False) -> None:
    cases = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    embedder, llm = get_embedder(), get_llm()
    reranker = get_reranker() if use_reranker else None
    rr, hits, correct, cited, refusals = [], 0, 0, 0, 0
    in_corpus = [c for c in cases if c["fiches"]]
    with connect() as conn:
        for case in cases:
            results = retrieve(conn, case["question"], embedder, reranker, top_k=top_k)
            result = answer(case["question"], results, llm)
            expected = set(case["fiches"])
            if expected:
                r = reciprocal_rank([x.chunk.fiche_id for x in results], expected)
                rr.append(r)
                hits += r > 0
                ok = contains_all(result.answer, case["must_contain"])
                correct += ok
                cites = bool(expected & {s.id for s in result.sources})
                cited += cites
                status = f"rang {round(1 / r) if r else '—'} | réponse {'OK' if ok else 'KO'} | citation {'OK' if cites else 'KO'}"
            else:
                refused = REFUSAL in result.answer
                refusals += refused
                status = f"hors corpus | refus {'OK' if refused else 'KO'}"
            print(f"- {case['question']}\n    {status}")
    n, m = len(in_corpus), len(cases) - len(in_corpus)
    print(f"\nReranking : {'oui' if reranker else 'non'}")
    print(f"Recherche : hit@{top_k} = {hits}/{n}, MRR = {sum(rr) / n:.2f}")
    print(f"Réponse : éléments attendus {correct}/{n}, fiche attendue citée {cited}/{n}")
    print(f"Hors corpus : refus {refusals}/{m}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rerank"]
    main(int(args[0]) if args else 5, "--rerank" in sys.argv)
