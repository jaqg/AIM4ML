"""Deterministic ordered parallel map (fork + copy-on-write).

Reuses the pattern from the v0 pipeline (03-stats/stats_qm40.py):
  - module-level globals set BEFORE the Pool is created, so forked workers
    inherit them via copy-on-write — no pickling of large `items` per task;
  - imap_unordered over chunked index lists;
  - results reordered by original index so output order == input order,
    identical for any worker count.
"""

import multiprocessing as mp

# Python 3.14 changed the Linux default start method to "forkserver", which
# forks workers from a pristine server process — module globals assigned AFTER
# import are NOT inherited (copy-on-write breaks). Pin to "fork" so
# _WORK_FN/_WORK_ITEMS set before Pool() are visible in workers via COW,
# avoiding per-task pickling of large `items`.
_CTX = mp.get_context("fork")

_WORK_FN = None
_WORK_ITEMS = None


def _run_chunk(indices):
    """Worker: apply _WORK_FN to a chunk of items, return (index, result)."""
    return [(i, _WORK_FN(_WORK_ITEMS[i])) for i in indices]


def _split_chunks(n, n_chunks):
    """Split range(n) into n_chunks balanced index lists (no empty chunks)."""
    if n_chunks <= 0:
        n_chunks = 1
    if n_chunks > n:
        n_chunks = n
    base, rem = divmod(n, n_chunks)
    chunks = []
    start = 0
    for k in range(n_chunks):
        size = base + (1 if k < rem else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def parallel_map(fn, items, n_workers=1, chunk_factor=20):
    """
    Map fn over items, returning results in input order.

    workers <= 1 : serial (no pool overhead).
    workers >  1 : forked process pool with imap_unordered over index chunks;
                   results reordered by index → deterministic output.

    fn must be module-level (picklable). items must be picklable OR inherited
    via fork copy-on-write (Linux default start method).
    """
    if n_workers <= 1:
        return [fn(x) for x in items]

    n = len(items)
    if n == 0:
        return []

    global _WORK_FN, _WORK_ITEMS
    _WORK_FN = fn
    _WORK_ITEMS = items

    chunks = _split_chunks(n, n_workers * chunk_factor)
    results = [None] * n
    with _CTX.Pool(processes=n_workers) as pool:
        for chunk_results in pool.imap_unordered(_run_chunk, chunks):
            for i, r in chunk_results:
                results[i] = r
    return results
