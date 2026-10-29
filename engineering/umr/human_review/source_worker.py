"""Reuse the read-only SOMA source model across replay preparations.

The viewer sends paths to frozen source-build requests to one private child.
Only the child imports Torch/SOMA.  Its existing, parameter-keyed SOMA layer
cache stays alive between clips; every clip still uses replay_data's complete
source reconstruction and validation.  This module changes no motion geometry.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]


class SourceBuildError(RuntimeError):
    """A source reconstruction failed without corrupting later requests."""


class SourceWorker:
    """A serialized JSON-lines connection to a persistent CPU source process.

    ``command`` is useful for protocol tests.  Normal callers use the module's
    ``build`` function, so foreground and prefetch requests share one model.
    """

    def __init__(self, *, command=None, timeout_s=900.):
        python = os.environ.get("UMR_REPLAY_SOMA_PYTHON", sys.executable)
        self.command = list(command) if command is not None else [python, str(Path(__file__)), "--serve"]
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0.:
            raise ValueError("Source worker timeout must be positive")
        self._lock = threading.Lock()
        self._process = None
        self._responses = None

    def _start_locked(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_locked()
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", UMR_SOMA_DEVICE="cpu")
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, encoding="utf-8", bufsize=1, env=environment, cwd=ROOT)
        responses = queue.Queue()
        self._process, self._responses = process, responses

        def read_responses():
            try:
                for line in process.stdout:
                    responses.put(line)
            except (OSError, ValueError) as error:
                responses.put(error)
            finally:
                responses.put(None)

        threading.Thread(target=read_responses, name="soma-source-responses", daemon=True).start()

    def _stop_locked(self):
        process, self._process = self._process, None
        self._responses = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def build(self, spec_path: Path) -> dict:
        """Build one complete source; automatically restart an exited worker.

        An interrupted request may safely be repeated: replay_data publishes
        completed mesh and metadata files atomically.  A reported build error
        is returned to the caller and never consumed as a later clip's result.
        """
        spec_path = Path(spec_path).resolve(strict=True)
        with self._lock:
            for attempt in range(2):
                self._start_locked()
                request_id = uuid.uuid4().hex
                try:
                    self._process.stdin.write(json.dumps({"id": request_id, "path": str(spec_path)}) + "\n")
                    self._process.stdin.flush()
                    raw = self._responses.get(timeout=self.timeout_s)
                    if raw is None or isinstance(raw, Exception):
                        raise ConnectionError("SOMA source process exited before returning its result")
                    response = json.loads(raw)
                    if not isinstance(response, dict) or response.get("id") != request_id:
                        raise ConnectionError("SOMA source process returned an unrelated response")
                    if response.get("status") == "error":
                        raise SourceBuildError(response.get("error", "Unknown SOMA source build error"))
                    if response.get("status") != "ok":
                        raise ConnectionError("SOMA source process returned an invalid status")
                    return response
                except queue.Empty as error:
                    self._stop_locked()
                    raise SourceBuildError(f"SOMA source build exceeded {self.timeout_s:g} seconds: {spec_path}") from error
                except (BrokenPipeError, OSError, ConnectionError, json.JSONDecodeError) as error:
                    self._stop_locked()
                    if attempt:
                        raise SourceBuildError(f"SOMA source worker failed: {error}") from error
        raise AssertionError("Unreachable source worker state")

    def close(self):
        with self._lock:
            self._stop_locked()


_WORKER = SourceWorker()


def build(spec_path: Path) -> dict:
    return _WORKER.build(spec_path)


def close():
    _WORKER.close()


atexit.register(close)


def _serve(input_stream, output_stream, builder=None):
    if builder is None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from engineering.umr.human_review.replay_data import _build_source
        builder = _build_source
    for line in input_stream:
        request_id = None
        started = time.monotonic()
        try:
            request = json.loads(line)
            request_id = request["id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("A nonempty request ID is required")
            path = Path(request["path"]).resolve(strict=True)
            with contextlib.redirect_stdout(sys.stderr):
                builder(path)
            response = {"id": request_id, "status": "ok", "duration_s": time.monotonic() - started,
                        "worker_pid": os.getpid()}
            # Diagnostics expose reuse without changing the upstream layer or
            # its cache keys, allowing real cold/warm timings to be audited.
            soma = sys.modules.get("soma_source")
            layer = getattr(soma, "_soma_x_layer", None)
            if layer is not None and hasattr(layer, "cache_info"):
                response["layer_cache"] = layer.cache_info()._asdict()
        except Exception as error:
            response = {"id": request_id, "status": "error", "error": f"{type(error).__name__}: {error}",
                        "duration_s": time.monotonic() - started, "worker_pid": os.getpid()}
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.parse_args()
    # Keep a dedicated protocol descriptor before redirecting fd 1.  Some
    # native SOMA dependencies print outside Python's redirect_stdout context;
    # those logs must also go to stderr rather than corrupting JSON-lines.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with protocol:
        _serve(sys.stdin, protocol)


if __name__ == "__main__":
    main()
