---
library: quillstore
version: 3.2
---
# Errors

All quillstore exceptions inherit from `quillstore.QuillError`.

## KeyTooLargeError

Raised by `put` when a key is longer than 1024 bytes.

## ConflictError

Raised when a transaction commits after another writer changed a key it read. Retry the
whole transaction. `store.retry_transaction(fn, attempts=3)` calls `fn(txn)` inside a
new transaction and retries on `ConflictError` up to `attempts` times.

```python
def transfer(txn):
    txn.put("a", txn.get("a") - 1)
    txn.put("b", txn.get("b") + 1)


store.retry_transaction(transfer, attempts=5)
```

## StoreLockedError

Another process holds the store's write lock and did not release it within
`lock_timeout` seconds. Only one process can open a store for writing at a time;
any number can open it with `readonly=True`.

## CorruptionError

The file failed a checksum, usually after a crash with `sync="never"`. Run
`quillstore.repair(path)` (or the `quillstore repair` command), which keeps every intact
record and writes a report of what was lost.

## ReadOnlyError

A write was attempted on a store opened with `readonly=True`.

## StoreNotFoundError

`open()` was called with `create=False` and the file does not exist.

## StoreClosedError

The store was used after `close()`.

## CodecMismatchError

The store was opened with a different `codec` than the one it was created with.
