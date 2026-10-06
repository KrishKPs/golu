"""Split documents by their structure: one chunk per section.

- A section is the text under a heading, labelled with its heading path
  (e.g. "API reference > Store.put").
- Code blocks are never split.
- A section longer than `max_chars` is split between paragraphs/code blocks,
  never inside one. A single huge code block stays whole.

Optional front matter at the top of a markdown file sets metadata:
    ---
    library: quillstore
    version: 3.2
    source_url: https://example.com/docs/api
    ---
"""

import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 2400
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    section: str  # heading path, "A > B > C"
    index: int  # position within the document
    title: str = ""
    library: str | None = None
    version: str | None = None

    def embedding_text(self) -> str:
        """What we embed: the text plus where it sits, which helps short sections match."""
        prefix = " > ".join(p for p in (self.library, self.section) if p)
        return f"{prefix}\n\n{self.text}"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[end + 4 :].lstrip("\n")


def chunk_markdown(
    text: str,
    source: str,
    *,
    library: str | None = None,
    version: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    meta, body = parse_front_matter(text)
    source = meta.get("source_url", source)
    library = meta.get("library", library)
    version = meta.get("version", version)
    sections = _split_sections(body)
    title = meta.get("title") or next((path[0] for path, _ in sections if path), source)

    chunks: list[Chunk] = []
    for path, lines in sections:
        body_lines = lines[1:] if path else lines  # lines[0] is the heading itself
        if not "".join(body_lines).strip():
            continue  # a heading with nothing under it (its children carry its path)
        # Text before any heading is labelled with the document title.
        label = " > ".join(path) if path else title
        for piece in _pack(_blocks(lines), max_chars):
            chunks.append(Chunk(piece, source, label, len(chunks), title, library, version))
    return chunks


def chunk_plain_text(
    text: str,
    source: str,
    *,
    library: str | None = None,
    version: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """For files with no heading structure: paragraphs packed into chunks."""
    title = source.rsplit("/", 1)[-1]
    return [
        Chunk(piece, source, title, i, title, library, version)
        for i, piece in enumerate(_pack(_blocks(text.splitlines()), max_chars))
    ]


def _split_sections(body: str) -> list[tuple[list[str], list[str]]]:
    """[(heading path, lines including the heading line)]"""
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    stack: list[tuple[int, str]] = []  # (level, heading text)
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            sections.append(([h for _, h in stack], [line]))
        else:
            sections[-1][1].append(line)
    return sections


def _blocks(lines: list[str]) -> list[str]:
    """Paragraphs and whole code blocks."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            if not in_fence and current:  # a fence starts a new block
                blocks.append("\n".join(current))
                current = []
            in_fence = not in_fence
            current.append(line)
            if not in_fence:  # fence closed: the code block is complete
                blocks.append("\n".join(current))
                current = []
            continue
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    """Greedily join blocks into chunks of at most max_chars (a bigger block stays whole)."""
    pieces: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = block
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return pieces
