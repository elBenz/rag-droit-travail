from pathlib import Path

from rag.parsing import load_fiches, parse_fiche

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_metadata():
    fiche = parse_fiche(FIXTURES / "F9999.xml")
    assert fiche.id == "F9999"
    assert fiche.title == "Congés de test"
    assert fiche.url == "https://www.service-public.gouv.fr/particuliers/vosdroits/F9999"


def test_introduction_is_first_section_with_plain_text():
    intro = parse_fiche(FIXTURES / "F9999.xml").sections[0]
    assert intro.path == []
    assert intro.text == "Le salarié a droit à 2,5 jours par mois."


def test_cases_become_their_own_sections_with_title_path():
    sections = parse_fiche(FIXTURES / "F9999.xml").sections
    paths = [s.path for s in sections]
    assert ["Qui a droit aux congés ?", "Cas général"] in paths
    assert ["Qui a droit aux congés ?", "Apprenti"] in paths
    apprenti = next(s for s in sections if s.path[-1:] == ["Apprenti"])
    assert apprenti.text == "5 semaines par an.\n- Premier point\n- Second point"


def test_chapter_without_cases_keeps_all_its_text():
    chap = next(s for s in parse_fiche(FIXTURES / "F9999.xml").sections
                if s.path == ["Comment poser ses congés ?"])
    assert "L'employeur fixe l'ordre des départs." in chap.text
    assert "Un accord collectif peut prévoir autre chose." in chap.text


def test_load_fiches_keeps_only_travail_theme():
    assert [f.id for f in load_fiches(FIXTURES)] == ["F9999"]


def test_situations_are_sections_and_table_rows_stay_on_one_line():
    sections = parse_fiche(FIXTURES / "F9999.xml").sections
    table = next(s for s in sections if s.path == ["CDD", "Quel délai de prévenance ?"])
    assert table.text == "Délai\nPrésence | Délai\nMoins de 8 jours | 24 heures"
