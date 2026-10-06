---
library: quillstore
version: 3.2
---
# API reference

## quillstore.open

```python
quillstore.open(path, *, create=True, sync="batch", cache_mb=64, readonly=False,
                lock_timeout=5.0, codec="msgpack", compression="lz4") -> Store
```

Opens (or creates) a store file and returns a `Store`.

- `sync`: when writes are flushed to disk. `"always"` fsyncs every write (safest, slowest),
  `"batch"` fsyncs at most every 50 ms (default), `"never"` leaves it to the OS.
- `cache_mb`: size of the in-memory read cache in megabytes.
- `readonly`: open without write access; writes raise `ReadOnlyError`.
- `lock_timeout`: seconds to wait for another process's write lock before raising
  `StoreLockedError`.

## Store.put

```python
Store.put(key, value, *, ttl_seconds=None, if_absent=False) -> int
```

Stores `value` under `key` and returns the new revision number.

- `ttl_seconds`: the key expires and is deleted this many seconds after the write.
  `None` means it never expires.
- `if_absent`: only write if the key does not exist yet. Returns `0` without writing if
  the key already exists.

Keys longer than 1024 bytes raise `KeyTooLargeError`.

## Store.get

```python
Store.get(key, default=None) -> Any
```

Returns the value for `key`, or `default` if the key does not exist or has expired.

## Store.get_many

```python
Store.get_many(keys) -> dict
```

Fetches several keys in one call. Missing keys are left out of the returned dict.

## Store.delete

```python
Store.delete(key) -> bool
```

Deletes `key`. Returns `True` if it existed.

## Store.scan

```python
Store.scan(prefix="", *, start=None, limit=None, reverse=False) -> Iterator[tuple[key, value]]
```

Iterates over keys in sorted order. `prefix` limits results to keys starting with it;
`start` begins at the first key greater than or equal to `start`; `reverse=True` walks
backwards. Expired keys are skipped.

```python
for key, value in store.scan(prefix="user:", limit=10):
    print(key, value)
```

## Store.transaction

```python
with store.transaction(isolation="snapshot") as txn:
    balance = txn.get("acct:1")
    txn.put("acct:1", balance - 10)
```

All reads in a transaction see one consistent snapshot. Writes are applied atomically
when the `with` block exits without an exception, and discarded if it raises.
`isolation` can be `"snapshot"` (default) or `"serializable"`. If another writer changed
a key this transaction read, commit raises `ConflictError`.

## Store.watch

```python
sub = store.watch("user:", callback)
sub.cancel()
```

Calls `callback(key, value)` after every committed write to a key starting with the
prefix. `value` is `None` for deletes. The callback runs on a background thread.
Call `sub.cancel()` to stop watching.

## Store.compact

`Store.compact()` rewrites the file to reclaim space from deleted and expired keys. It
blocks writers while it runs.

## Store.stats

`Store.stats()` returns a dict with `keys`, `file_bytes`, `live_bytes`, `cache_hits`,
and `cache_misses`.
