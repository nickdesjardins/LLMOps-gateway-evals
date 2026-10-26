"""The knowledge base: Markdown documents in ``corpus/``, one chunk per section.

Each document starts with ``# Title``; each ``## Section`` is a chunk, split further at
paragraph boundaries when it runs past ``MAX_CHUNK_CHARS``. A chunk's id is
``<file stem>#<section slug>`` (``expense-policy#meals-and-entertainment``): stable across
runs, readable in a trace, and the label the golden dataset uses for the passages that hold
each answer.

The text that is embedded and shown to the model starts with the document title and the
section heading. A passage that says "up to 75 dollars a day" means nothing without them.

The corpus describes a fictional company (Cedar Ridge Naturals) and was written for this
demo. Its fingerprint, a hash of every chunk, is part of the cache namespace: editing a
document empties the cache rather than serving answers drawn from the old text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "corpus"
MAX_CHUNK_CHARS = 1_400
_HEADING = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def title(self) -> str:
        return f"{self.doc_title} > {self.section}"

    @property
    def indexed_text(self) -> str:
        """What is embedded and searched: the headings carry the context."""
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    chunks: tuple[Chunk, ...]


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def parse_document(doc_id: str, markdown: str) -> Document:
    headings = list(_HEADING.finditer(markdown))
    title = next((m.group(2) for m in headings if m.group(1) == "#"), doc_id)
    sections = [m for m in headings if m.group(1) == "##"]
    chunks: list[Chunk] = []
    for index, heading in enumerate(sections):
        end = sections[index + 1].start() if index + 1 < len(sections) else len(markdown)
        body = markdown[heading.end() : end].strip()
        if not body:
            continue
        section = heading.group(2)
        parts = _split_long(body)
        for number, part in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f"-{number}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slugify(section)}{suffix}",
                    doc_id=doc_id,
                    doc_title=title,
                    section=section,
                    text=part.strip(),
                )
            )
    return Document(doc_id=doc_id, title=title, chunks=tuple(chunks))


@dataclass(frozen=True)
class Corpus:
    documents: tuple[Document, ...]

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    def chunk(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode("utf-8") + b"\x00" + chunk.text.encode("utf-8") + b"\x00")
        return digest.hexdigest()[:16]


def load_corpus(directory: Path = CORPUS_DIR) -> Corpus:
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no Markdown documents in {directory}")
    documents = [parse_document(path.stem, path.read_text(encoding="utf-8")) for path in files]
    ids = [c.chunk_id for d in documents for c in d.chunks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate chunk ids in the corpus: {duplicates}")
    return Corpus(documents=tuple(documents))
