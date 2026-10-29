#!/usr/bin/env python3
"""Run a bounded read-only AI comparison on a registered repair evidence packet."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[3]


def agent_module():
    try:
        from . import agent_review
    except ImportError:
        import agent_review
    return agent_review


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packet_directory(created):
    if isinstance(created, (str, Path)):
        return Path(created).resolve()
    for key in ["job_dir", "output", "directory"]:
        if key in created:
            return Path(created[key]).resolve()
    raise ValueError("Agent packet builder did not return a job directory")


def verify_packet(job):
    job = Path(job).resolve()
    for name in ["AGENT_REVIEW_JOB.json", "PROMPT.md", "RESPONSE_SCHEMA.json"]:
        if not (job / name).is_file():
            raise ValueError(f"Agent packet is incomplete: {name}")
    checker = getattr(agent_module(), "validate_job", None)
    if checker is not None:
        checker(job)
    checksums = job / "SHA256SUMS"
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            expected, name = line.split("  ", 1)
            if Path(name).name != name or sha(job / name) != expected:
                raise ValueError(f"Frozen AI evidence packet changed: {name}")
    # This schema is also enforced by the external CLI. Import-side identity
    # and semantic validation remains mandatory after the agent returns.
    schema = json.loads((job / "RESPONSE_SCHEMA.json").read_text())
    if schema.get("type") != "object":
        raise ValueError("Agent response schema must define an object")
    return job


def command_for(job, result, executable="codex"):
    binary = shutil.which(executable)
    if binary is None:
        raise ValueError("Codex CLI is unavailable; the evidence packet can still be reviewed and imported separately")
    return [binary, "exec", "--sandbox", "read-only", "--ephemeral", "--json",
            "--output-schema", str(job / "RESPONSE_SCHEMA.json"),
            "--output-last-message", str(result), "-"]


def dispatch(job, *, timeout_seconds=1800, executable="codex", prepare_only=False):
    if timeout_seconds <= 0:
        raise ValueError("Agent timeout must be positive")
    job = verify_packet(job)
    attempt = job / "attempts" / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    result = attempt / "response.json"
    argv = command_for(job, result, executable)
    if prepare_only:
        return {"status": "ready_not_executed", "job_dir": str(job), "command": argv,
                "sandbox": "read-only", "agent_started": False}
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"job_dir": str(job), "attempt": str(attempt), "command": argv,
              "job_sha256": sha(job / "AGENT_REVIEW_JOB.json"),
              "prompt_sha256": sha(job / "PROMPT.md"), "schema_sha256": sha(job / "RESPONSE_SCHEMA.json"),
              "sandbox": "read-only", "agent_started": True, "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    with (attempt / "events.jsonl").open("x") as events, (attempt / "stderr.log").open("x") as errors:
        process = None
        try:
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                       cwd=ROOT, text=True, start_new_session=True)
            process.communicate((job / "PROMPT.md").read_text(), timeout=timeout_seconds)
            record["returncode"] = process.returncode
            if process.returncode:
                record.update(status="agent_failed", completed=False)
            elif not result.is_file():
                record.update(status="missing_structured_response", completed=False)
            else:
                validation = agent_module().validate_response(job, result)
                record.update(status="validated" if validation.get("completed") else "response_rejected",
                              completed=bool(validation.get("completed")), validation=validation)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record.update(status="agent_timeout", completed=False, timeout_seconds=timeout_seconds)
        except Exception as exc:
            record.update(status="response_rejected" if process is not None else "agent_start_failed",
                          completed=False, agent_started=process is not None, error=str(exc))
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (attempt / "DISPATCH_STATUS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def dispatch_session(session, *, batch_size=10, timeout_seconds=1800, executable="codex", progress=None):
    """Resume AI work in bounded packages, generating true four-way visuals."""
    try:
        from .render_agent_evidence import render_job
    except ImportError:
        from render_agent_evidence import render_job
    module = agent_module()
    jobs = module.create_jobs(session, batch_size=batch_size, resume=True)
    results = []
    for index, created in enumerate(jobs, 1):
        numeric = packet_directory(created)
        if progress:
            progress(f"正在生成第 {index} / {len(jobs)} 批的全部事件多视角证据…")
        rendered = render_job(numeric, numeric / "visuals")
        visual_manifest = rendered["visual_manifest_path"]
        ids = [row["review_id"] for row in json.loads((numeric / "AGENT_REVIEW_JOB.json").read_text())["motions"]]
        ready = packet_directory(module.create_job(session, visual_manifest=visual_manifest, review_ids=ids))
        if progress:
            progress(f"正在对照第 {index} / {len(jobs)} 批；未查看事件会保留为待复核…")
        result = dispatch(ready, timeout_seconds=timeout_seconds, executable=executable)
        results.append(result)
        if not result.get("completed"):
            return {"completed": False, "status": result["status"], "attempt": result.get("attempt"), "batches": results}
    cycle = module._adaptive_module()._state(Path(session).resolve())["cycle"]
    gate = module.validate_session_review(session, cycle)
    return {**gate, "status": "validated" if gate["completed"] else "response_rejected", "batches": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--session", type=Path, help="Build a packet from the current registered repair cycle")
    selector.add_argument("--job", type=Path, help="Use an existing frozen comparison packet")
    parser.add_argument("--prepare-only", action="store_true", help="Validate packet and show invocation without calling an agent")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=10, help="每批AI对照动作数，所有动作仍须覆盖")
    args = parser.parse_args()
    try:
        if args.session and not args.prepare_only:
            result = dispatch_session(args.session, batch_size=args.batch_size, timeout_seconds=args.timeout_seconds)
        else:
            job = packet_directory(agent_module().create_job(args.session)) if args.session else args.job
            result = dispatch(job, timeout_seconds=args.timeout_seconds, prepare_only=args.prepare_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("completed") is False:
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"dispatch-agent: {exc}\n")


if __name__ == "__main__":
    main()
