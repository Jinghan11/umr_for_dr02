"""Validate the adapted UMR source without relying on Git metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


# Hash executable source, configuration and provenance on every validation. Mesh
# and sample-data bytes remain recorded in the manifest, but only their presence
# and size are checked here to avoid repeatedly reading the entire asset tree.
_SOURCE_SUFFIXES = frozenset({
    ".py", ".json", ".txt", ".xml", ".urdf", ".yaml", ".yml", ".toml",
    ".js", ".mjs", ".html", ".md",
})
_SOURCE_FILENAMES = frozenset({"LICENSE", ".gitignore", ".gitattributes"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError(f"Invalid vendored UMR manifest path: {value!r}")
    return Path(*path.parts)


def validate_vendored_source(upstream: Path, expected_commit: str) -> str:
    """Check the recorded base commit and adapted source identity.

    SOURCE.json pins the complete manifest, which in turn records the original
    upstream tracked-file set after the local adaptation. This works in a plain
    source directory, including one nested inside a different Git repository.
    """
    upstream = Path(upstream)
    method = upstream.parent
    source = json.loads((method / "SOURCE.json").read_text())
    if source["commit"] != expected_commit:
        raise ValueError(f"Vendored UMR base commit changed: {source['commit']}")
    provenance = source["vendor"]
    manifest_path = method / _relative_path(provenance["manifest"])
    if _sha256(manifest_path) != provenance["manifest_sha256"]:
        raise ValueError("Vendored UMR manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported vendored UMR manifest version")
    if manifest["base_commit"] != expected_commit:
        raise ValueError(f"Vendored UMR manifest base commit changed: {manifest['base_commit']}")
    if manifest["source_url"] != source["source_url"]:
        raise ValueError("Vendored UMR source URL differs from provenance")
    patch = manifest["adaptation_patch"]
    if _sha256(method / _relative_path(patch["path"])) != patch["sha256"]:
        raise ValueError("Vendored UMR adaptation patch SHA256 mismatch")
    files = manifest["files"]
    if not files or len(files) != manifest["file_count"] or "LICENSE" not in files:
        raise ValueError("Vendored UMR manifest has an incomplete tracked-file set")
    if sum(PurePosixPath(name).suffix == ".py" for name in files) != manifest["python_file_count"]:
        raise ValueError("Vendored UMR Python source count differs from manifest")
    for name, record in files.items():
        path = upstream / _relative_path(name)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Vendored UMR file missing or changed: {name}")
        if (path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_FILENAMES) and _sha256(path) != record["sha256"]:
            raise ValueError(f"Vendored UMR source SHA256 mismatch: {name}")
    return manifest["base_commit"]
