---
library: quillstore
version: 3.2
---
# Migrating from quillstore 2.x to 3.x

## Renamed APIs

| 2.x | 3.x |
| --- | --- |
| `QuillDB(path)` | `quillstore.open(path)` |
| `db.set(key, value, expire=60)` | `store.put(key, value, ttl_seconds=60)` |
| `db.iter_prefix(p)` | `store.scan(prefix=p)` |
| `db.atomic()` | `store.transaction()` |

## Transaction isolation changed

In 2.x transactions were serializable. In 3.x the default is snapshot isolation, which
allows more concurrency. Pass `isolation="serializable"` to keep the old behaviour.

## Upgrading data files

3.x reads 2.x files in read-only mode. Convert them once with the CLI:

```bash
quillstore migrate data.qs
```

The original file is kept as `data.qs.v2.bak`.
