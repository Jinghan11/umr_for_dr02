"""Full-data prefetch deduplication, priority, cancellation and byte limits."""
from concurrent.futures import CancelledError
from dataclasses import dataclass
import threading

import numpy as np
import pytest

from engineering.umr.human_review.replay_buffer import ReplayBuffer, artifact_key, data_nbytes


def item(identity):
    return {"trajectory": f"/frozen/{identity}/trajectory.npz", "final_sha256": identity,
            "review_id": f"human_{identity}"}


def test_requests_share_one_preparation_and_one_completed_future():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(row):
        calls.append(row["final_sha256"])
        entered.set()
        assert release.wait(5)
        return np.zeros((10, 20, 3), np.float32)

    with ReplayBuffer(load) as buffer:
        first = buffer.request(item("a"), foreground=False)
        assert entered.wait(5)
        assert buffer.status(item("a"))["state"] == "loading"
        # A new review identity still refers to the exact same frozen artifact.
        alias = {**item("a"), "review_id": "second_pass_of_same_artifact"}
        assert buffer.request(alias) is first
        release.set()
        replay = first.result(timeout=5)
        assert buffer.request(item("a")) is first
        assert buffer.request(item("a")).result() is replay
        assert calls == ["a"]
        status = buffer.status(item("a"))
        assert status["state"] == "ready" and status["bytes"] == replay.nbytes
        assert status["elapsed_s"] >= 0 and status["queue_wait_s"] >= 0
        assert buffer.stats()["loading"] == 0
        assert len(buffer.stats()["timings"]) == 1


def test_visible_request_is_promoted_ahead_of_queued_background_work():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(row):
        identity = row["final_sha256"]
        calls.append(identity)
        if identity == "a":
            entered.set()
            assert release.wait(5)
        return np.zeros(1, dtype=np.uint8)

    rows = [item(identity) for identity in "abcde"]
    with ReplayBuffer(load) as buffer:
        first = buffer.request(rows[0])
        assert entered.wait(5)
        futures = buffer.prefetch(rows, 0)
        assert buffer.status(rows[1])["state"] == "queued"
        promoted = buffer.request(rows[3])
        assert promoted is futures[2]
        release.set()
        first.result(timeout=5)
        for future in futures:
            future.result(timeout=5)
        assert calls == ["a", "d", "b", "c"]


def test_prefetch_only_nearby_items_and_cancels_obsolete_pending_work():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(row):
        identity = row["final_sha256"]
        calls.append(identity)
        if identity == "0":
            entered.set()
            assert release.wait(5)
        return np.zeros(1, dtype=np.uint8)

    rows = [item(str(index)) for index in range(200)]
    with ReplayBuffer(load) as buffer:
        first = buffer.request(rows[0])
        assert entered.wait(5)
        obsolete = buffer.prefetch(rows, 0)
        target = buffer.request(rows[100])
        adjacent = buffer.prefetch(rows, 100)
        assert len(adjacent) == 5
        assert all(future.cancelled() for future in obsolete)
        with pytest.raises(CancelledError):
            obsolete[0].result()
        assert buffer.status(rows[1])["state"] == "cancelled"
        assert buffer.stats()["queued"] == 6
        release.set()
        first.result(timeout=5)
        target.result(timeout=5)
        for future in adjacent:
            future.result(timeout=5)
        assert calls == ["0", "100", "101", "102", "103", "99", "98"]


def test_lru_is_limited_by_real_array_bytes_and_keeps_current_result():
    calls = []

    def load(row):
        calls.append(row["final_sha256"])
        return np.zeros(128, dtype=np.uint8)

    with ReplayBuffer(load, max_bytes=256) as buffer:
        a = buffer.request(item("a"))
        a.result(timeout=5)
        b = buffer.request(item("b"))
        b.result(timeout=5)
        assert buffer.request(item("a")) is a  # touch a, b is oldest
        buffer.request(item("c")).result(timeout=5)
        assert buffer.stats()["resident_bytes"] == 256
        assert buffer.status(item("b"))["state"] == "evicted"
        assert buffer.status(item("a"))["state"] == "ready"
        assert buffer.status(item("c"))["state"] == "ready"
        buffer.request(item("b")).result(timeout=5)
        assert calls == ["a", "b", "c", "b"]


def test_oversized_current_is_retained_alone_without_hiding_memory_use():
    with ReplayBuffer(lambda _: np.zeros(1024, np.uint8), max_bytes=512) as buffer:
        current = buffer.request(item("a"))
        current.result(timeout=5)
        buffer.request(item("b"), foreground=False).result(timeout=5)
        stats = buffer.stats()
        assert stats["resident_bytes"] == 1024 and stats["oversized_current"] is True
        assert stats["ready"] == 1
        assert buffer.status(item("a"))["in_memory"] is True
        assert buffer.status(item("b"))["state"] == "evicted"
        assert buffer.request(item("a")) is current


def test_farther_prefetch_does_not_evict_the_immediately_next_ready_motion():
    rows = [item(identity) for identity in "abcde"]
    with ReplayBuffer(lambda _: np.zeros(128, np.uint8), max_bytes=256) as buffer:
        buffer.request(rows[0]).result(timeout=5)
        adjacent = buffer.prefetch(rows, 0)
        for future in adjacent:
            future.result(timeout=5)
        assert buffer.stats()["resident_bytes"] == 256
        assert buffer.status(rows[0])["state"] == "ready"
        assert buffer.status(rows[1])["state"] == "ready"
        assert buffer.status(rows[2])["state"] == "evicted"
        assert buffer.status(rows[3])["state"] == "evicted"
        assert buffer.request(rows[1]) is adjacent[0]


def test_consumed_old_motion_is_evicted_to_make_room_for_next_prefetch():
    rows = [item(identity) for identity in "abcd"]
    with ReplayBuffer(lambda _: np.zeros(128, np.uint8), max_bytes=256) as buffer:
        buffer.request(rows[0]).result(timeout=5)
        buffer.request(rows[1]).result(timeout=5)
        for future in buffer.prefetch(rows, 1, ahead=2, behind=0):
            future.result(timeout=5)
        assert buffer.status(rows[0])["state"] == "evicted"
        assert buffer.status(rows[1])["state"] == "ready"
        assert buffer.status(rows[2])["state"] == "ready"


def test_loader_failures_are_visible_and_explicit_retry_does_one_new_load():
    calls = []

    def load(row):
        calls.append(row)
        if len(calls) == 1:
            raise ValueError("source checksum mismatch")
        return np.zeros(8, dtype=np.uint8)

    with ReplayBuffer(load) as buffer:
        failed = buffer.request(item("a"))
        with pytest.raises(ValueError, match="source checksum mismatch"):
            failed.result(timeout=5)
        assert buffer.request(item("a")) is failed
        status = buffer.status(item("a"))
        assert status["state"] == "error" and "checksum mismatch" in status["error"]
        assert status["bytes"] == 0 and status["elapsed_s"] >= 0
        buffer.request(item("a"), retry=True).result(timeout=5)
        assert len(calls) == 2 and buffer.status(item("a"))["state"] == "ready"


def test_full_source_mesh_and_shared_array_views_are_counted_without_reduction():
    @dataclass
    class Data:
        source_vertices: object
        duplicate_view: object
        metadata: object

    vertices = np.ones((101, 18056, 3), np.float32)
    data = Data(vertices, vertices[3:50], {"clock": "complete"})
    size = data_nbytes(data)
    assert vertices.nbytes <= size < vertices.nbytes + 4096
    with ReplayBuffer(lambda _: data, max_bytes=size + 1) as buffer:
        replay = buffer.request(item("a")).result(timeout=5)
        assert replay.source_vertices is vertices
        assert replay.source_vertices.shape == (101, 18056, 3)
        assert buffer.stats()["resident_bytes"] == size


def test_source_identity_and_artifact_versions_are_part_of_cache_key():
    before, after = item("a"), {**item("a"), "final_sha256": "new bytes"}
    assert artifact_key(before) != artifact_key(after)
    assert artifact_key(before) == artifact_key({**before, "review_id": "different review"})


@pytest.mark.parametrize("changed", ["trajectory", "native_trajectory", "contact_report"])
def test_changed_file_bytes_do_not_reuse_a_stale_memory_future(tmp_path, changed):
    row = {"trajectory": str(tmp_path / "trajectory.npz"),
           "native_trajectory": str(tmp_path / "native_trajectory.npz"),
           "contact_report": str(tmp_path / "contact_report.json"),
           "final_sha256": "unchanged frozen fingerprint"}
    for field in ("trajectory", "native_trajectory", "contact_report"):
        from pathlib import Path
        Path(row[field]).write_bytes(b"original")
    calls = []
    with ReplayBuffer(lambda item: calls.append(item) or np.zeros(1, np.uint8)) as buffer:
        first = buffer.request(row)
        first.result(timeout=5)
        Path(row[changed]).write_bytes(b"changed bytes, manifest fingerprint unchanged")
        second = buffer.request(row)
        second.result(timeout=5)
        assert first is not second and len(calls) == 2


def test_close_cancels_waiting_work_and_rejects_new_work():
    entered, release = threading.Event(), threading.Event()

    def load(_):
        entered.set()
        assert release.wait(5)
        return np.zeros(4, dtype=np.uint8)

    buffer = ReplayBuffer(load)
    running = buffer.request(item("a"))
    assert entered.wait(5)
    pending = buffer.request(item("b"), foreground=False)
    buffer.close()
    assert pending.cancelled() and buffer.stats()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        buffer.request(item("c"))
    with pytest.raises(RuntimeError, match="closed"):
        buffer.prefetch([item("a")], 0)
    release.set()
    running.result(timeout=5)
    buffer.close(wait=True)


def test_invalid_limits_and_ranges_are_rejected():
    for limit in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ReplayBuffer(lambda _: None, max_bytes=limit)
    with ReplayBuffer(lambda _: None) as buffer:
        with pytest.raises(IndexError):
            buffer.prefetch([], 0)
        with pytest.raises(ValueError):
            buffer.prefetch([item("a")], 0, ahead=-1)
