"""Dispatch lifecycle tests with a local fake CLI and isolated synthetic packets.

No model is invoked and no real review session or production file is touched.
"""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from engineering.umr.human_review import dispatch_agent as dispatch
from test_agent_review import _packet, _response


@pytest.fixture
def job(tmp_path):
    directory = _packet(tmp_path)
    # This is deliberately an honest UNKNOWN/HOLD response, not a fabricated
    # successful optimization. The real importer still validates the response.
    response = _response(directory)
    (directory / "FAKE_RESPONSE.json").write_text(json.dumps(response))
    return directory


@pytest.fixture
def fake_cli(tmp_path):
    executable = tmp_path / "fake codex cli.py"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import hashlib
import json
import os
from pathlib import Path
import sys
import time

arguments = sys.argv[1:]
assert arguments[:6] == ["exec", "--sandbox", "read-only", "--ephemeral", "--json", "--output-schema"]
assert arguments[-1] == "-"
schema_path = Path(arguments[arguments.index("--output-schema") + 1])
result_path = Path(arguments[arguments.index("--output-last-message") + 1])
assert json.loads(schema_path.read_text())["type"] == "object"
prompt = sys.stdin.read()
assert prompt == (schema_path.parent / "PROMPT.md").read_text()
print(json.dumps({"fake_cli_only": True, "arguments": arguments,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest()}), flush=True)
print("fake CLI diagnostic; no model was called", file=sys.stderr, flush=True)
mode = os.environ.get("UMR_TEST_DISPATCH_MODE", "success")
if mode == "missing":
    raise SystemExit(0)
response = json.loads((schema_path.parent / "FAKE_RESPONSE.json").read_text())
if mode == "wrong_identity":
    response["job_sha256"] = "0" * 64
result_path.write_text("{invalid JSON" if mode == "malformed" else json.dumps(response))
if mode == "nonzero":
    raise SystemExit(7)
if mode == "timeout":
    time.sleep(30)
''')
    executable.chmod(0o755)
    return str(executable)


def status_record(record):
    attempt = Path(record["attempt"])
    assert attempt.is_dir()
    assert (attempt / "events.jsonl").is_file()
    assert (attempt / "stderr.log").is_file()
    assert json.loads((attempt / "DISPATCH_STATUS.json").read_text()) == record
    return attempt


def test_command_is_read_only_ephemeral_and_uses_the_exact_frozen_schema(job, fake_cli):
    response = job / "unused response.json"
    command = dispatch.command_for(job, response, executable=fake_cli)
    assert command == [fake_cli, "exec", "--sandbox", "read-only", "--ephemeral", "--json",
                       "--output-schema", str(job / "RESPONSE_SCHEMA.json"),
                       "--output-last-message", str(response), "-"]
    assert not response.exists()
    schema = json.loads((job / "RESPONSE_SCHEMA.json").read_text())
    assert schema == dispatch.agent_module().response_schema()
    assert schema["additionalProperties"] is False


def test_prepare_only_checks_the_packet_without_starting_or_creating_an_attempt(job, fake_cli, monkeypatch):
    process = Mock(side_effect=AssertionError("prepare-only must not launch any process"))
    monkeypatch.setattr(dispatch.subprocess, "Popen", process)
    original = {path.name: path.read_bytes() for path in job.iterdir() if path.is_file()}
    record = dispatch.dispatch(job, executable=fake_cli, prepare_only=True)
    assert record["status"] == "ready_not_executed"
    assert record["agent_started"] is False
    assert not record.get("completed", False)
    assert record["sandbox"] == "read-only"
    assert not (job / "attempts").exists()
    assert {path.name: path.read_bytes() for path in job.iterdir() if path.is_file()} == original
    process.assert_not_called()


@pytest.mark.parametrize("damage", ["missing_schema", "changed_prompt", "changed_registered_trajectory"])
def test_packet_failure_does_not_launch_a_process_or_leave_a_false_attempt(job, fake_cli, monkeypatch, damage):
    process = Mock(side_effect=AssertionError("an invalid packet must not launch an agent"))
    monkeypatch.setattr(dispatch.subprocess, "Popen", process)
    if damage == "missing_schema":
        (job / "RESPONSE_SCHEMA.json").unlink()
    elif damage == "changed_prompt":
        prompt = job / "PROMPT.md"
        prompt.chmod(0o644)
        prompt.write_text("tampered fixture prompt")
    else:
        packet = json.loads((job / "AGENT_REVIEW_JOB.json").read_text())
        Path(packet["motions"][0]["after"]["trajectory"]).write_bytes(b"changed registered fixture")
    with pytest.raises(ValueError, match="incomplete|identity mismatch|changed"):
        dispatch.dispatch(job, executable=fake_cli)
    assert not (job / "attempts").exists()
    process.assert_not_called()


def test_success_uses_real_validation_and_preserves_honest_unknown_for_human_review(job, fake_cli):
    record = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=10)
    assert record["status"] == "validated"
    assert record["returncode"] == 0 and record["completed"] is True
    assert record["validation"]["completed"] is True
    assert record["validation"]["optimization_accepted"] is False
    assert record["validation"]["human_review_still_required"] is True
    attempt = status_record(record)
    assert (attempt / "response.json").is_file()
    event = json.loads((attempt / "events.jsonl").read_text())
    assert event["fake_cli_only"] is True
    assert event["prompt_sha256"] == dispatch.sha(job / "PROMPT.md")
    assert event["schema_sha256"] == record["schema_sha256"]
    assert record["job_sha256"] == dispatch.sha(job / "AGENT_REVIEW_JOB.json")
    assert "no model was called" in (attempt / "stderr.log").read_text()
    report = Path(record["validation"]["report_path"])
    assert report.is_file()
    assert hashlib.sha256(report.read_bytes()).hexdigest() == record["validation"]["report_sha256"]


@pytest.mark.parametrize("mode,expected", [("nonzero", "agent_failed"), ("timeout", "agent_timeout"),
                                            ("missing", "missing_structured_response")])
def test_process_failures_never_complete_or_import_even_if_response_already_exists(job, fake_cli, monkeypatch, mode, expected):
    monkeypatch.setenv("UMR_TEST_DISPATCH_MODE", mode)
    validator = Mock(side_effect=AssertionError("failed or incomplete process must not import a response"))
    monkeypatch.setattr(dispatch.agent_module(), "validate_response", validator)
    record = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=.2 if mode == "timeout" else 10)
    assert record["status"] == expected
    assert record["completed"] is False
    attempt = status_record(record)
    if mode in {"nonzero", "timeout"}:
        assert (attempt / "response.json").is_file()  # File presence is never a completion signal.
    else:
        assert not (attempt / "response.json").exists()
    if mode == "nonzero":
        assert record["returncode"] == 7
    if mode == "timeout":
        assert record["timeout_seconds"] == .2
    validator.assert_not_called()
    assert not list(job.glob("VALIDATION_REPORT_*.json"))


@pytest.mark.parametrize("mode", ["malformed", "wrong_identity"])
def test_zero_exit_does_not_override_failed_response_validation(job, fake_cli, monkeypatch, mode):
    monkeypatch.setenv("UMR_TEST_DISPATCH_MODE", mode)
    record = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=10)
    assert record["returncode"] == 0
    assert record["status"] == "response_rejected"
    assert record["completed"] is False
    assert record["validation"]["errors"]
    assert record["validation"]["optimization_accepted"] is False
    status_record(record)


def test_validation_exception_is_a_failed_attempt_with_both_logs_retained(job, fake_cli, monkeypatch):
    validator = Mock(side_effect=ValueError("fixture validation failure"))
    monkeypatch.setattr(dispatch.agent_module(), "validate_response", validator)
    record = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=10)
    assert record["status"] == "response_rejected" and record["completed"] is False
    assert "fixture validation failure" in record["error"]
    assert validator.call_count == 1
    attempt = status_record(record)
    assert (attempt / "events.jsonl").read_text()
    assert (attempt / "stderr.log").read_text()


def test_repeated_dispatch_keeps_distinct_attempt_logs_and_immutable_validation_history(job, fake_cli, monkeypatch):
    first = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=10)
    first_attempt = status_record(first)
    first_files = {path.name: path.read_bytes() for path in first_attempt.iterdir()}
    first_report = Path(first["validation"]["report_path"])
    first_report_bytes = first_report.read_bytes()
    monkeypatch.setenv("UMR_TEST_DISPATCH_MODE", "nonzero")
    second = dispatch.dispatch(job, executable=fake_cli, timeout_seconds=10)
    second_attempt = status_record(second)
    assert first_attempt != second_attempt
    assert set((job / "attempts").iterdir()) == {first_attempt, second_attempt}
    assert {path.name: path.read_bytes() for path in first_attempt.iterdir()} == first_files
    assert first_report.read_bytes() == first_report_bytes
    assert first["completed"] is True and second["completed"] is False
    assert len(list(job.glob("VALIDATION_REPORT_*.json"))) == 1


def test_invalid_timeout_and_unavailable_cli_never_start_an_attempt(job, monkeypatch):
    process = Mock(side_effect=AssertionError("invalid inputs must not launch an agent"))
    monkeypatch.setattr(dispatch.subprocess, "Popen", process)
    with pytest.raises(ValueError, match="positive"):
        dispatch.dispatch(job, timeout_seconds=0)
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="unavailable"):
        dispatch.dispatch(job)
    assert not (job / "attempts").exists()
    process.assert_not_called()


def test_process_start_failure_preserves_attempt_status(job, fake_cli, monkeypatch):
    monkeypatch.setattr(dispatch.subprocess, 'Popen', Mock(side_effect=OSError('fixture cannot start')))
    record = dispatch.dispatch(job, executable=fake_cli)
    assert record['status'] == 'agent_start_failed' and not record['completed'] and not record['agent_started']
    status_record(record)


def test_session_dispatch_renders_all_packages_and_binds_visuals_before_calling_agent(tmp_path, monkeypatch):
    from types import SimpleNamespace
    events = []
    packages = []
    for index in range(2):
        path = tmp_path / str(index); path.mkdir()
        (path / 'AGENT_REVIEW_JOB.json').write_text(json.dumps({'motions': [{'review_id': str(index)}]}))
        packages.append({'job_dir': str(path)})
    module = SimpleNamespace(
        create_jobs=lambda *args, **kwargs: packages,
        create_job=lambda *args, **kwargs: events.append(('bind', kwargs)) or {'job_dir': str(tmp_path / 'ready')},
        _adaptive_module=lambda: SimpleNamespace(_state=lambda _: {'cycle': 1}),
        validate_session_review=lambda *args: {'completed': True, 'next_review_count': 20})
    monkeypatch.setattr(dispatch, 'agent_module', lambda: module)
    def render(path, output):
        events.append(('render', path.name)); return {'visual_manifest_path': str(output / 'visual_manifest.json')}
    monkeypatch.setitem(sys.modules, 'engineering.umr.human_review.render_agent_evidence', SimpleNamespace(render_job=render))
    monkeypatch.setattr(dispatch, 'dispatch', lambda *args, **kwargs: events.append(('agent', args[0])) or {'completed': True})
    record = dispatch.dispatch_session(tmp_path)
    assert record['completed'] and len(record['batches']) == 2
    assert [item[0] for item in events] == ['render', 'bind', 'agent', 'render', 'bind', 'agent']
    assert events[1][1]['review_ids'] == ['0'] and events[4][1]['review_ids'] == ['1']
    assert events[1][1]['visual_manifest'].endswith('/0/visuals/visual_manifest.json')
