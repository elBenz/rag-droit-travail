from rag.chunking import chunk_fiche
from rag.parsing import Fiche, Section


def fiche(*sections):
    return Fiche(id="F1", title="Congés payés", url="https://u", sections=list(sections))


def test_one_chunk_per_section_prefixed_with_context():
    chunks = chunk_fiche(fiche(Section([], "Intro."), Section(["Qui ?", "Apprenti"], "5 semaines.")))
    assert [c.text for c in chunks] == [
        "Congés payés\nIntro.",
        "Congés payés > Qui ? > Apprenti\n5 semaines.",
    ]
    assert all((c.fiche_id, c.title, c.url) == ("F1", "Congés payés", "https://u") for c in chunks)


def test_long_section_is_split_with_context_on_every_piece():
    long_text = " ".join(f"Phrase numéro {i}." for i in range(200))
    chunks = chunk_fiche(fiche(Section(["Chapitre"], long_text)), max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(c.text.startswith("Congés payés > Chapitre\n") for c in chunks)
    assert all(len(c.text) <= 500 + len("Congés payés > Chapitre\n") for c in chunks)


def test_empty_sections_are_skipped():
    assert chunk_fiche(fiche(Section(["Vide"], "  "))) == []
