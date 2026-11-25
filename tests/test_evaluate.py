from rag.evaluate import contains_all, reciprocal_rank


def test_reciprocal_rank_of_first_expected_fiche():
    assert reciprocal_rank(["F2", "F1", "F1"], {"F1"}) == 0.5
    assert reciprocal_rank(["F1"], {"F1", "F9"}) == 1.0
    assert reciprocal_rank(["F2", "F3"], {"F1"}) == 0.0


def test_contains_all_ignores_case_and_non_breaking_spaces():
    assert contains_all("Majoration de 25 % puis 50 %", ["25 %", "50 %"])
    assert contains_all("Le 1ER MAI", ["1er mai"])
    assert not contains_all("2 mois", ["4 mois"])
