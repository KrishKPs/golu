---
library: quillstore
version: 3.2
---
# Configuration

## Environment variables

Environment variables set defaults for `quillstore.open()`; explicit arguments win.

- `QUILL_CACHE_MB`: default `cache_mb`.
- `QUILL_SYNC`: default `sync` mode (`always`, `batch`, or `never`).
- `QUILL_LOG_LEVEL`: log level for the `quillstore` logger.

## Codecs

The `codec` argument controls how values are serialized:

- `"msgpack"` (default): compact and fast.
- `"json"`: human-readable files; values must be JSON-compatible.
- `"pickle"`: any Python object. Disabled by default because loading pickle data can run
  arbitrary code; enable it with `allow_pickle=True`.

A store remembers its codec. Opening it later with a different codec raises
`CodecMismatchError`.

## Compression

`compression` can be `"lz4"` (default), `"zstd"`, or `None`. zstd compresses better but
needs the `quillstore[zstd]` extra. Compression applies to values larger than 256 bytes.

## Logging

quillstore logs through the standard `logging` module under the logger name
`quillstore`. Slow operations (over 100 ms) are logged at WARNING level.
