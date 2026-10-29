"""Persistent source protocol tests; no real training/review data is changed."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time

import pytest

from engineering.umr.human_review import source_worker as worker


@pytest.fixture
def fake_worker(tmp_path):
    script = tmp_path / "fake source process.py"
    script.write_text(r'''
import json
import os
from pathlib import Path
import sys
import time
count = 0
for line in sys.stdin:
    request = json.loads(line)
    spec = json.loads(Path(request["path"]).read_text())
    mode = spec.get("mode", "ok")
    if mode == "crash_once":
        marker = Path(spec["marker"])
        if not marker.exists():
            marker.write_text("first worker died")
            os._exit(17)
    if mode == "timeout":
        time.sleep(30)
    if mode == "wrong_id":
        request["id"] = "wrong"
    count += 1
    response = {"id": request["id"], "status": "error" if mode == "error" else "ok",
                "error": "intentional source error", "worker_pid": os.getpid(), "count": count,
                "source_path": request["path"],
                "environment": {key: os.environ.get(key) for key in
                    ("UMR_SOMA_DEVICE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "CUDA_VISIBLE_DEVICES")}}
    print(json.dumps(response), flush=True)
''')
    return [sys.executable, str(script)]


def spec(tmp_path, name="request.json", **values):
    path = tmp_path / name
    path.write_text(json.dumps(values))
    return path


def test_worker_reuses_process_and_cpu_environment(fake_worker, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preserved-test-setting")
    connection = worker.SourceWorker(command=fake_worker)
    try:
        first = connection.build(spec(tmp_path, "first.json"))
        second = connection.build(spec(tmp_path, "second.json"))
        assert first["worker_pid"] == second["worker_pid"]
        assert [first["count"], second["count"]] == [1, 2]
        assert first["environment"] == {"UMR_SOMA_DEVICE": "cpu", "OMP_NUM_THREADS": "1",
                                        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                        "CUDA_VISIBLE_DEVICES": "preserved-test-setting"}
    finally:
        connection.close()
    assert connection._process is None


def test_concurrent_builds_cannot_consume_each_others_response(fake_worker, tmp_path):
    paths = [spec(tmp_path, f"request {number}.json") for number in range(8)]
    connection = worker.SourceWorker(command=fake_worker)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(connection.build, paths))
        assert [response["source_path"] for response in responses] == [str(path) for path in paths]
        assert len({response["worker_pid"] for response in responses}) == 1
        assert sorted(response["count"] for response in responses) == list(range(1, 9))
    finally:
        connection.close()


def test_build_error_is_local_and_next_request_uses_same_worker(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker)
    try:
        first = connection.build(spec(tmp_path, "first.json"))
        with pytest.raises(worker.SourceBuildError, match="intentional source error"):
            connection.build(spec(tmp_path, "broken.json", mode="error"))
        last = connection.build(spec(tmp_path, "last.json"))
        assert first["worker_pid"] == last["worker_pid"]
        assert last["count"] == 3
    finally:
        connection.close()


def test_worker_restarts_after_its_process_dies(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker)
    try:
        first = connection.build(spec(tmp_path, "first.json"))
        connection._process.kill()
        connection._process.wait(timeout=2.)
        second = connection.build(spec(tmp_path, "second.json"))
        assert first["worker_pid"] != second["worker_pid"]
        assert second["count"] == 1
    finally:
        connection.close()


def test_worker_retries_one_interrupted_atomic_build(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker)
    try:
        result = connection.build(spec(tmp_path, mode="crash_once", marker=str(tmp_path / "crash marker")))
        assert result["status"] == "ok"
        assert result["count"] == 1
    finally:
        connection.close()


def test_unrelated_result_is_never_accepted(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker)
    try:
        with pytest.raises(worker.SourceBuildError, match="unrelated response"):
            connection.build(spec(tmp_path, mode="wrong_id"))
        assert connection._process is None
    finally:
        connection.close()


def test_timeout_kills_only_own_worker_and_can_recover(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker, timeout_s=.2)
    try:
        started = time.monotonic()
        with pytest.raises(worker.SourceBuildError, match="exceeded"):
            connection.build(spec(tmp_path, mode="timeout"))
        assert time.monotonic() - started < 5.
        assert connection._process is None
        assert connection.build(spec(tmp_path, "next.json"))["status"] == "ok"
    finally:
        connection.close()


def test_missing_request_does_not_start_process(fake_worker, tmp_path):
    connection = worker.SourceWorker(command=fake_worker)
    with pytest.raises(FileNotFoundError):
        connection.build(tmp_path / "missing.json")
    assert connection._process is None


def test_serve_redirects_builder_logs_and_recovers_after_build_failure(tmp_path, capsys):
    first, broken, last = [spec(tmp_path, name) for name in ("first.json", "broken.json", "last.json")]
    calls = []

    def builder(path):
        calls.append(path)
        print("heavy source dependency log")
        if path == broken:
            raise ValueError("wrong source identity")

    requests = io.StringIO("".join(json.dumps({"id": str(index), "path": str(path)}) + "\n"
                                   for index, path in enumerate((first, broken, last))))
    output = io.StringIO()
    worker._serve(requests, output, builder=builder)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["id"] for response in responses] == ["0", "1", "2"]
    assert [response["status"] for response in responses] == ["ok", "error", "ok"]
    assert responses[1]["error"] == "ValueError: wrong source identity"
    assert calls == [first, broken, last]
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.count("heavy source dependency log") == 3
