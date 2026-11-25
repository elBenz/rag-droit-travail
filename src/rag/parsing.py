"""Lecture des fiches Service-Public (XML, schéma 3.4) en sections de texte brut."""

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

TRAVAIL_THEME = "N19806"
DC = "{http://purl.org/dc/elements/1.1/}"
# Éléments qui ouvrent une nouvelle section, identifiée par leur <Titre>.
SECTION_TAGS = {"Situation", "Chapitre", "SousChapitre", "Cas"}


@dataclass
class Section:
    path: list[str]  # titres du chapitre, puis du cas éventuel ; [] pour l'introduction
    text: str


@dataclass
class Fiche:
    id: str
    title: str
    url: str
    sections: list[Section]


def _clean(text: str) -> str:
    return " ".join(text.split())


def _title(el) -> str:
    titre = el.find("Titre")
    return _clean("".join(titre.itertext())) if titre is not None else ""


def _walk(el, path: list[str], sections: list[Section]) -> list[str]:
    """Renvoie les lignes de texte de `el` ; les sous-sections sont ajoutées à `sections`."""
    lines: list[str] = []
    for child in el:
        tag = child.tag
        if not isinstance(tag, str) or tag == "Titre":
            continue
        if tag in SECTION_TAGS:
            _add_section(child, path + [_title(child)], sections)
        elif tag == "Paragraphe":
            if text := _clean("".join(child.itertext())):
                lines.append(text)
        elif tag == "Tableau":
            if title := _title(child):
                lines.append(title)
            lines.extend(_walk(child, path, sections))
        elif tag == "Rangée":  # une ligne de tableau reste sur une ligne
            lines.append(" | ".join(" ".join(_walk(cell, path, sections)) for cell in child.iter("Cellule")))
        elif tag == "Item":
            lines.append("- " + " ".join(_walk(child, path, sections)))
        else:  # Liste, BlocCas, ASavoir, Attention, Tableau… : on descend
            lines.extend(_walk(child, path, sections))
    return lines


def _add_section(el, path: list[str], sections: list[Section]) -> None:
    position = len(sections)
    lines = _walk(el, path, sections)
    if lines:
        # Le texte propre à la section passe avant ses sous-sections.
        sections.insert(position, Section(path, "\n".join(lines)))


def parse_fiche(path: Path) -> Fiche:
    root = etree.parse(str(path)).getroot()
    sections: list[Section] = []
    intro = root.find("Introduction")
    if intro is not None:
        _add_section(intro, [], sections)
    # Corps de la fiche : <Texte> direct, ou une variante par situation (CDI, CDD…).
    for tag in ("Texte", "ListeSituations"):
        body = root.find(tag)
        if body is not None:
            _add_section(body, [], sections)
    return Fiche(
        id=root.get("ID"),
        title=_clean(root.findtext(f"{DC}title") or ""),
        url=root.get("spUrl"),
        sections=sections,
    )


def is_travail(path: Path) -> bool:
    root = etree.parse(str(path)).getroot()
    return any(t.get("ID") == TRAVAIL_THEME for t in root.findall("Theme"))


def load_fiches(directory: Path) -> list[Fiche]:
    paths = sorted(Path(directory).glob("F*.xml"))
    return [parse_fiche(p) for p in paths if is_travail(p)]
