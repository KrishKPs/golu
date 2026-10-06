---
library: quillstore
version: 3.2
---
# Getting started with quillstore

quillstore is an embedded key-value store for Python. Data lives in a single file on
disk; there is no server to run.

## Installation

```bash
pip install quillstore
```

quillstore 3.2 requires Python 3.10 or newer. The optional `zstd` compression codec
needs the extra: `pip install "quillstore[zstd]"`.

## Opening a store

Call `quillstore.open()` with a file path. The file is created if it does not exist.

```python
import quillstore

store = quillstore.open("data.qs")
```

Pass `create=False` to raise `StoreNotFoundError` instead of creating a missing file.

## Reading and writing

```python
store.put("user:1", {"name": "Ada"})
store.get("user:1")  # {'name': 'Ada'}
store.get("missing", default=0)  # 0
store.delete("user:1")  # True if the key existed
```

Keys are `str` or `bytes`. Values can be any msgpack-serializable value: `str`,
`bytes`, `int`, `float`, `bool`, `None`, lists, and dicts.

## Closing the store

Always close the store so buffered writes are flushed. The easiest way is a `with`
block, which closes it automatically:

```python
with quillstore.open("data.qs") as store:
    store.put("k", "v")
```

You can also call `store.close()` yourself. Using a closed store raises `StoreClosedError`.
