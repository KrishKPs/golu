from rag_server.chunking import chunk_markdown, chunk_plain_text, parse_front_matter

DOC = """---
library: demo
version: 1.2
---
# Guide

Intro text.

## Install

pip install demo

## Usage

### Basic

```python
# a comment that looks like a heading
x = 1
```
"""


def test_front_matter() -> None:
    meta, body = parse_front_matter(DOC)
    assert meta == {"library": "demo", "version": "1.2"}
    assert body.startswith("# Guide")


def test_sections_use_heading_paths_and_metadata() -> None:
    chunks = chunk_markdown(DOC, "guide.md")
    assert [c.section for c in chunks] == ["Guide", "Guide > Install", "Guide > Usage > Basic"]
    assert all(c.library == "demo" and c.version == "1.2" for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_heading_inside_code_block_is_not_a_section() -> None:
    chunks = chunk_markdown(DOC, "guide.md")
    assert "# a comment that looks like a heading" in chunks[-1].text


def test_long_section_split_between_blocks_code_kept_whole() -> None:
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```"
    text = "# Big\n\n" + "para one. " * 30 + "\n\n" + code + "\n\n" + "para two. " * 30
    chunks = chunk_markdown(text, "big.md", max_chars=500)
    assert len(chunks) == 3
    assert all(c.section == "Big" for c in chunks)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1 and code_chunks[0].text.rstrip().endswith("```")


def test_empty_heading_sections_skipped_and_source_url() -> None:
    text = "---\nsource_url: https://x.dev/a\n---\n# A\n## B\ncontent\n"
    chunks = chunk_markdown(text, "a.md")
    assert [(c.source, c.section) for c in chunks] == [("https://x.dev/a", "A > B")]


def test_plain_text() -> None:
    chunks = chunk_plain_text("one\n\ntwo", "notes/readme.txt", version="2")
    assert len(chunks) == 1 and chunks[0].section == "readme.txt" and chunks[0].version == "2"


def test_embedding_text_includes_location() -> None:
    chunk = chunk_markdown(DOC, "guide.md")[1]
    assert chunk.embedding_text().startswith("demo > Guide > Install")
