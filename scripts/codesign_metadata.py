#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_METADATA_FIELDS = [
    "timestamp",
    "git_commit",
    "git_status",
    "cli_args",
    "baseline_yaml_sha256",
    "patch_matrix",
    "hw_yaml_content_sha256",
    "hw_yaml_diff_from_baseline",
    "per_cell",
    "ramulator2_config_sha256",
    "gem5_binary_sha256",
    "togsim_binary_sha256",
    "pytorchsim_git_commit",
    "env_vars_snapshot",
    "versions",
    "determinism_smoke_test",
    "artifact_paths",
]
FORBIDDEN_CLAIM_VALUES = {"placeholder", "modeled", "pending_measurement", "pending"}
REQUIRED_PER_CELL_FIELDS = [
    "hw_id",
    "mapping_id",
    "status",
    "simulator_cmdline",
    "stdout_path",
    "stderr_path",
    "subprocess_pid",
    "start_time",
    "end_time",
]


class MetadataSchemaError(ValueError):
    pass


class ClaimBearingDataError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "not_captured"


def git_status(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else "not_captured"


def capture_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["torch", "transformers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_captured"
    return versions


def env_snapshot(env: dict[str, str] | None = None) -> dict[str, str]:
    source = env or os.environ
    return {
        "PATH": source.get("PATH", ""),
        "PYTHONPATH": source.get("PYTHONPATH", ""),
        "LD_LIBRARY_PATH": source.get("LD_LIBRARY_PATH", ""),
    }


def _path_sha256(path: Path | str) -> str:
    candidate = Path(path)
    return sha256_file(candidate) if candidate.exists() and candidate.is_file() else "not_captured"


def _hw_yaml_diff_from_baseline(hw_summary: dict[str, Any]) -> dict[str, Any]:
    if "hw_yaml_diff_from_baseline" in hw_summary:
        return dict(hw_summary["hw_yaml_diff_from_baseline"])
    if "patch_matrix" in hw_summary:
        return dict(hw_summary["patch_matrix"])
    return {}


def build_metadata_manifest(
    *,
    repo_root: Path,
    cli_args: dict[str, Any],
    hw_summary: dict[str, Any],
    sweep_summary: dict[str, Any],
    determinism_smoke_test: dict[str, Any],
    artifact_paths: dict[str, Any],
    binary_paths: dict[str, Path | str],
    env: dict[str, str] | None = None,
    versions: dict[str, str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    per_cell = [dict(row) for row in sweep_summary.get("fit_availability", {}).get("cells", [])]
    baseline_yaml = Path(hw_summary.get("baseline_yaml", ""))
    baseline_sha = hw_summary.get("baseline_yaml_sha256")
    if not baseline_sha and baseline_yaml.exists():
        baseline_sha = sha256_file(baseline_yaml)
    manifest = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "git_status": git_status(repo_root),
        "cli_args": cli_args,
        "baseline_yaml_sha256": baseline_sha or "not_captured",
        "patch_matrix": dict(hw_summary.get("patch_matrix", {})),
        "hw_yaml_content_sha256": dict(hw_summary.get("hw_yaml_content_sha256", {})),
        "hw_yaml_diff_from_baseline": _hw_yaml_diff_from_baseline(hw_summary),
        "per_cell": per_cell,
        "ramulator2_config_sha256": _path_sha256(binary_paths.get("ramulator2_config", "")),
        "gem5_binary_sha256": _path_sha256(binary_paths.get("gem5_binary", "")),
        "togsim_binary_sha256": _path_sha256(binary_paths.get("togsim_binary", "")),
        "pytorchsim_git_commit": git_commit(repo_root),
        "env_vars_snapshot": env_snapshot(env),
        "versions": versions or capture_versions(),
        "determinism_smoke_test": determinism_smoke_test,
        "artifact_paths": artifact_paths,
    }
    lint_metadata(manifest)
    return manifest


def lint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise MetadataSchemaError("metadata missing required fields: " + ", ".join(missing))
    for idx, row in enumerate(metadata.get("per_cell", [])):
        row_missing = [field for field in REQUIRED_PER_CELL_FIELDS if field not in row]
        if row_missing:
            raise MetadataSchemaError(
                f"per_cell[{idx}] missing required fields: " + ", ".join(row_missing)
            )
    return {"ok": True, "checked_fields": REQUIRED_METADATA_FIELDS}


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def lint_claim_bearing(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("claim_bearing", False):
        return {"ok": True, "claim_bearing": False}
    bad = [
        str(value)
        for value in _walk_values(payload)
        if isinstance(value, str) and value in FORBIDDEN_CLAIM_VALUES
    ]
    if bad:
        raise ClaimBearingDataError("claim-bearing payload contains forbidden values: " + ", ".join(sorted(set(bad))))
    return {"ok": True, "claim_bearing": True}
