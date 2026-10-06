---
library: quillstore
version: 3.2
---
# Command-line tool

Installing quillstore adds a `quillstore` command.

## quillstore inspect

`quillstore inspect PATH` prints the store's version, codec, compression, and key count.

## quillstore dump

`quillstore dump PATH --format jsonl > out.jsonl` writes every key and value. Formats:
`jsonl` (default) and `csv`. Use `--prefix` to dump only matching keys.

## quillstore load

`quillstore load PATH < out.jsonl` imports a jsonl dump. Existing keys are overwritten
unless you pass `--skip-existing`.

## quillstore migrate

Converts a 2.x file to the 3.x format. See the migration guide.

## quillstore repair

`quillstore repair PATH` recovers a corrupted store. Same as `quillstore.repair()`.

## quillstore stats

`quillstore stats PATH` prints the same numbers as `Store.stats()`.
