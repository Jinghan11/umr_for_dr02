"""Bounded, single-worker preparation of complete synchronized replay data.

The loader remains responsible for artifact and clock validation.  This module
only changes when it runs: the visible request has priority over queued adjacent
motions, and each artifact has at most one preparation in flight.  It never
changes the review queue or any review conclusion.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import heapq
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Hashable, Mapping, Sequence


DEFAULT_MAX_BYTES = 512 * 1024 ** 2


def artifact_key(item: Mapping[str, Any]) -> tuple:
    """Bind reuse to frozen identities *and* the currently present file versions.

    Stat signatures only invalidate a cached result.  They never establish data
    integrity: a cache miss still runs the loader's complete hash validation.
    """
    final_path = Path(item["trajectory"]).resolve()

    def signature(path):
        path = Path(path).resolve()
        try:
            value = path.stat()
        except OSError:
            return str(path), None
        return str(path), (value.st_dev, value.st_ino, value.st_size,
                           value.st_mtime_ns, value.st_ctime_ns)

    return (signature(final_path),
            signature(item.get("native_trajectory") or final_path.with_name("native_trajectory.npz")),
            signature(item.get("contact_report") or final_path.with_name("contact_report.json")),
            item.get("final_sha256"), item.get("native_sha256"),
            item.get("model_sha256"), item.get("report_sha256"))


def data_nbytes(value: Any) -> int:
    """Count array buffers and ordinary Python containers without double counting.

    ReplayData is dominated by its full-frame source vertices.  ndarray views
    sharing a base buffer count once; no array conversion or frame reduction is
    performed.  Python container overhead is an estimate from ``getsizeof``.
    """
    seen, buffers = set(), set()

    def visit(obj):
        identity = id(obj)
        if identity in seen:
            return 0
        seen.add(identity)
        nbytes = getattr(obj, "nbytes", None)
        if isinstance(nbytes, int):
            base = obj
            while getattr(base, "base", None) is not None:
                base = base.base
            buffer_id = id(base)
            if buffer_id in buffers:
                return 0
            buffers.add(buffer_id)
            return int(getattr(base, "nbytes", nbytes))
        size = sys.getsizeof(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return size + sum(visit(getattr(obj, field.name)) for field in fields(obj))
        if isinstance(obj, dict):
            return size + sum(visit(key) + visit(item) for key, item in obj.items())
        if isinstance(obj, (tuple, list, set, frozenset, deque)):
            return size + sum(visit(item) for item in obj)
        if hasattr(obj, "__dict__"):
            return size + visit(vars(obj))
        return size

    return visit(value)


@dataclass
class _Entry:
    key: Hashable
    item: Any
    future: Future
    priority: int
    sequence: int
    queued_at: float
    state: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    size: int = 0
    error: str | None = None
    foreground_used: bool = False
    prefetch_rank: int = 0


class ReplayBuffer:
    """Prepare adjacent motions ahead of review with a byte-bounded LRU.

    ``loader(item)`` must return a complete replay.  ``request`` returns a normal
    Future, including when the same preparation is already queued or running.
    A single current foreground result is protected from eviction; if that one
    motion exceeds the budget, it is retained alone and reported as oversized.
    Other references held by the scene or callers are outside this cache's
    ownership and cannot be freed here.
    """

    def __init__(self, loader: Callable[[Any], Any], max_bytes=DEFAULT_MAX_BYTES,
                 key: Callable[[Any], Hashable] | None = None):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.loader = loader
        self.max_bytes = max_bytes
        self.key = key or artifact_key
        self._condition = threading.Condition(threading.RLock())
        self._entries: dict[Hashable, _Entry] = {}
        self._lru: OrderedDict[Hashable, None] = OrderedDict()
        self._history: OrderedDict[Hashable, dict] = OrderedDict()
        self._timings = deque(maxlen=256)
        self._heap = []
        self._sequence = 0
        self._current = None
        self._resident_bytes = 0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="motion-replay-buffer", daemon=True)
        self._worker.start()

    def request(self, item, foreground=True, *, retry=False) -> Future:
        """Return the single shared future, promoting a queued visible request."""
        with self._condition:
            return self._request_locked(item, foreground=foreground, retry=retry, priority=10)

    def _request_locked(self, item, *, foreground, retry, priority):
        if self._closed:
            raise RuntimeError("ReplayBuffer is closed")
        identity = self.key(item)
        if foreground:
            self._current = identity
        entry = self._entries.get(identity)
        if entry is not None and retry and entry.state == "error":
            del self._entries[identity]
            entry = None
        if entry is not None:
            if foreground:
                entry.foreground_used = True
            else:
                entry.prefetch_rank = priority
            if entry.state == "ready":
                self._lru.move_to_end(identity)
            elif entry.state == "queued" and foreground and entry.priority != 0:
                self._sequence += 1
                entry.priority, entry.sequence = 0, self._sequence
                heapq.heappush(self._heap, (0, entry.sequence, identity))
                self._condition.notify()
            self._evict_locked()
            return entry.future
        self._sequence += 1
        entry = _Entry(identity, deepcopy(item), Future(), 0 if foreground else priority,
                       self._sequence, time.monotonic(),
                       foreground_used=foreground, prefetch_rank=priority)
        self._entries[identity] = entry
        heapq.heappush(self._heap, (entry.priority, entry.sequence, identity))
        self._evict_locked()
        self._condition.notify()
        return entry.future

    def prefetch(self, items: Sequence[Any], current_index: int, ahead=3, behind=2) -> list[Future]:
        """Queue the next three, then previous two; discard obsolete queued work.

        A running load is allowed to finish.  No more than one loader is active,
        and moving through a large queue does not enqueue that entire dataset.
        """
        if not 0 <= current_index < len(items):
            raise IndexError("current_index is outside the review queue")
        if ahead < 0 or behind < 0:
            raise ValueError("prefetch distances must be nonnegative")
        indices = list(range(current_index + 1, min(len(items), current_index + ahead + 1)))
        indices += list(range(current_index - 1, max(-1, current_index - behind - 1), -1))
        with self._condition:
            if self._closed:
                raise RuntimeError("ReplayBuffer is closed")
            desired = {self.key(items[index]) for index in indices}
            for identity, entry in list(self._entries.items()):
                if entry.state == "queued" and identity != self._current and identity not in desired:
                    entry.future.cancel()
                    self._remember_locked(entry, state="cancelled")
                    del self._entries[identity]
            return [self._request_locked(items[index], foreground=False, retry=False, priority=10 + rank)
                    for rank, index in enumerate(indices)]

    def _entry_status(self, entry, *, state=None):
        now = time.monotonic()
        return {"state": state or entry.state,
                "elapsed_s": None if entry.started_at is None else (entry.finished_at or now) - entry.started_at,
                "queue_wait_s": (entry.started_at or now) - entry.queued_at,
                "bytes": entry.size, "error": entry.error,
                "in_memory": (state or entry.state) == "ready"}

    def status(self, item) -> dict:
        """Return actual preparation timing and queued/loading/ready/error state."""
        identity = self.key(item)
        with self._condition:
            entry = self._entries.get(identity)
            if entry is not None:
                return self._entry_status(entry)
            return dict(self._history.get(identity, {"state": "not_requested", "elapsed_s": None,
                                                      "queue_wait_s": None, "bytes": 0,
                                                      "error": None, "in_memory": False}))

    def stats(self) -> dict:
        with self._condition:
            return {"max_bytes": self.max_bytes, "resident_bytes": self._resident_bytes,
                    "oversized_current": self._resident_bytes > self.max_bytes,
                    "ready": sum(e.state == "ready" for e in self._entries.values()),
                    "queued": sum(e.state == "queued" for e in self._entries.values()),
                    "loading": sum(e.state == "loading" for e in self._entries.values()),
                    "errors": sum(e.state == "error" for e in self._entries.values()),
                    "closed": self._closed, "timings": [dict(record) for record in self._timings]}

    def _remember_locked(self, entry, *, state=None):
        self._history[entry.key] = self._entry_status(entry, state=state)
        self._history.move_to_end(entry.key)
        while len(self._history) > 2048:
            self._history.popitem(last=False)

    def _evict_locked(self):
        while self._resident_bytes > self.max_bytes:
            victim = next((identity for identity in self._lru if identity != self._current), None)
            if victim is None:
                break
            # Background loads must not push the immediately-next prepared
            # motion out merely because a farther motion finished more recently.
            # Already consumed entries retain ordinary LRU order.  When its
            # oldest candidate is speculative, discard the farthest speculative
            # result instead; its verified disk cache is unaffected.
            if not self._entries[victim].foreground_used:
                speculative = [identity for identity in self._lru
                               if identity != self._current and not self._entries[identity].foreground_used]
                victim = max(speculative, key=lambda identity: self._entries[identity].prefetch_rank)
            self._lru.pop(victim)
            entry = self._entries.pop(victim)
            self._resident_bytes -= entry.size
            self._remember_locked(entry, state="evicted")

    def _run(self):
        while True:
            with self._condition:
                entry = None
                while entry is None:
                    if self._closed:
                        return
                    while self._heap:
                        priority, sequence, identity = heapq.heappop(self._heap)
                        candidate = self._entries.get(identity)
                        if (candidate is not None and candidate.state == "queued"
                                and (candidate.priority, candidate.sequence) == (priority, sequence)):
                            entry = candidate
                            break
                    if entry is None:
                        self._condition.wait()
                if not entry.future.set_running_or_notify_cancel():
                    del self._entries[entry.key]
                    self._remember_locked(entry, state="cancelled")
                    continue
                entry.state, entry.started_at = "loading", time.monotonic()
            try:
                result = self.loader(entry.item)
                size = data_nbytes(result)
            except BaseException as exc:
                with self._condition:
                    entry.state, entry.finished_at = "error", time.monotonic()
                    entry.error = f"{type(exc).__name__}: {exc}"
                    self._remember_locked(entry)
                    self._timings.append(self._entry_status(entry))
                # Do not keep a failed loader's entire frame (and any large
                # partially built source arrays) alive through the cached Future.
                entry.future.set_exception(exc.with_traceback(None))
            else:
                with self._condition:
                    entry.state, entry.finished_at, entry.size = "ready", time.monotonic(), size
                    self._lru[entry.key] = None
                    self._resident_bytes += size
                    self._remember_locked(entry)
                    self._timings.append(self._entry_status(entry))
                    self._evict_locked()
                # Complete outside the lock: consumers' callbacks may query this
                # buffer, attach a scene, or enqueue their own UI event.
                entry.future.set_result(result)
                del result
            # A worker's last Future/result must not secretly pin an evicted
            # full source mesh while the thread waits for the next request.
            del entry

    def close(self, wait=False):
        """Cancel queued work; an already running immutable load may finish."""
        with self._condition:
            self._closed = True
            for identity, entry in list(self._entries.items()):
                if entry.state == "queued":
                    entry.future.cancel()
                    self._remember_locked(entry, state="cancelled")
                    del self._entries[identity]
            self._heap.clear()
            self._condition.notify_all()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close(wait=True)
