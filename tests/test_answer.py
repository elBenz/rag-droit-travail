from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.answer import REFUSAL, answer, build_prompt
from rag.chunking import Chunk
from rag.store import Result

RESULTS = [
    Result(Chunk("F1", "Congés payés", "https://u/F1", "Congés payés\n2,5 jours par mois."), 0.9),
    Result(Chunk("F2", "Période d'essai", "https://u/F2", "Période d'essai\n2 mois pour un employé."), 0.8),
    Result(Chunk("F1", "Congés payés", "https://u/F1", "Congés payés > Apprenti\n5 semaines."), 0.7),
]


def test_prompt_numbers_each_extract():
    prompt = build_prompt("Combien de jours ?", RESULTS)
    assert "[1] Congés payés\n2,5 jours par mois." in prompt
    assert "[3] Congés payés > Apprenti\n5 semaines." in prompt
    assert "Combien de jours ?" in prompt


def test_sources_are_the_cited_fiches_deduplicated_in_order():
    llm = FakeListChatModel(responses=["2,5 jours par mois [3][1], soit 5 semaines [1]."])
    result = answer("Combien de jours ?", RESULTS, llm)
    assert result.answer == "2,5 jours par mois [3][1], soit 5 semaines [1]."
    assert [s.id for s in result.sources] == ["F1"]
    assert result.sources[0].url == "https://u/F1"


def test_out_of_range_citations_are_ignored():
    llm = FakeListChatModel(responses=["Voir [2] et [7]."])
    assert [s.id for s in answer("q", RESULTS, llm).sources] == ["F2"]


def test_refusal_has_no_sources():
    llm = FakeListChatModel(responses=[REFUSAL])
    assert answer("Quel temps fait-il ?", RESULTS, llm).sources == []


def test_no_extract_means_refusal_without_calling_llm():
    llm = FakeListChatModel(responses=[])  # lèverait une erreur si appelé
    result = answer("q", [], llm)
    assert (result.answer, result.sources) == (REFUSAL, [])


def test_empty_llm_output_becomes_refusal():
    llm = FakeListChatModel(responses=[""])
    assert answer("q", RESULTS, llm).answer == REFUSAL
