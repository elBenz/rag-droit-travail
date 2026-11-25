"""Découpage des fiches en chunks autonomes : une section = un chunk, redécoupée si trop longue."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.parsing import Fiche


@dataclass
class Chunk:
    fiche_id: str
    title: str
    url: str
    text: str


def chunk_fiche(fiche: Fiche, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap, separators=["\n", ". ", " ", ""]
    )
    chunks = []
    for section in fiche.sections:
        if not section.text.strip():
            continue
        context = " > ".join([fiche.title, *section.path])
        for piece in splitter.split_text(section.text):
            chunks.append(Chunk(fiche.id, fiche.title, fiche.url, f"{context}\n{piece}"))
    return chunks
