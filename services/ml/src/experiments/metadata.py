"""Small file-based helpers for versioned experiment tracking.

This module intentionally stays lightweight. It is not an experiment tracking
platform; it only standardizes the repeatable parts this repository needs:

- validate and resolve experiment IDs,
- write machine-readable metadata JSON,
- preserve existing artifacts by copying instead of moving,
- maintain the Markdown experiment registry table.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.generator.config import REPO_ROOT, SERVICE_ROOT

EXPERIMENT_ID_PATTERN = re.compile(r"^exp_\d{3}_[a-z0-9_]+$")

DOCS_EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
CONFIGS_EXPERIMENTS_DIR = SERVICE_ROOT / "configs" / "experiments"
ARTIFACT_EXPERIMENTS_DIR = REPO_ROOT / "data" / "artifacts" / "experiments"

METADATA_FILENAME = "experiment_metadata.json"
REGISTRY_START = "<!-- experiment-registry:start -->"
REGISTRY_END = "<!-- experiment-registry:end -->"


@dataclass(frozen=True)
class RegistryEntry:
    """One row in the human-readable experiment registry."""

    experiment_id: str
    title: str
    status: str
    schema_version: str
    dataset_config: str
    primary_target: str
    artifact_dir: str
    doc_path: str
    conclusion: str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_experiment_id(experiment_id: str) -> str:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id must match `exp_###_short_name`, "
            f"got: {experiment_id!r}"
        )
    return experiment_id


def resolve_experiment_artifact_dir(
    experiment_id: str,
    *,
    root: Path | None = None,
) -> Path:
    validate_experiment_id(experiment_id)
    base = root or ARTIFACT_EXPERIMENTS_DIR
    return (base / experiment_id).resolve()


def ensure_new_experiment_dir(path: Path, *, allow_existing: bool = False) -> Path:
    """Create an output directory while protecting completed experiment records."""

    existing_entries = list(path.iterdir()) if path.exists() else []
    if existing_entries and not allow_existing:
        raise FileExistsError(
            f"Experiment output directory is not empty: {path}. "
            "Use a new experiment ID or rerun explicitly with overwrite enabled."
        )
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / METADATA_FILENAME
    if metadata_path.exists() and not allow_existing:
        raise FileExistsError(
            f"Experiment output already has {METADATA_FILENAME}: {metadata_path}. "
            "Use a new experiment ID or rerun explicitly with overwrite enabled."
        )
    return path


def write_experiment_metadata(
    metadata: dict[str, Any],
    *,
    output_dir: Path,
    filename: str = METADATA_FILENAME,
) -> Path:
    validate_experiment_id(str(metadata.get("experiment_id", "")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / filename
    metadata_path.write_text(
        json.dumps(_to_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def preserve_artifact_copy(source: Path, destination: Path) -> Path:
    """Copy an existing artifact without mutating the source.

    If the destination already exists, it must be byte-identical. This lets
    preservation steps be idempotent while still catching accidental drift.
    """

    if not source.exists():
        raise FileNotFoundError(f"Cannot preserve missing artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(
                f"Destination artifact exists with different content: {destination}"
            )
        return destination
    shutil.copy2(source, destination)
    return destination


def upsert_registry_entry(registry_path: Path, entry: RegistryEntry) -> Path:
    """Insert or replace one registry entry in a Markdown table."""

    validate_experiment_id(entry.experiment_id)
    entries = _load_registry_entries(registry_path)
    entries[entry.experiment_id] = entry
    rendered_table = _render_registry_table(
        [entries[key] for key in sorted(entries)]
    )

    if registry_path.exists():
        text = registry_path.read_text(encoding="utf-8")
    else:
        text = _default_registry_text()

    if REGISTRY_START not in text or REGISTRY_END not in text:
        text = text.rstrip() + "\n\n" + REGISTRY_START + "\n" + REGISTRY_END + "\n"

    before = text.split(REGISTRY_START, maxsplit=1)[0]
    after = text.split(REGISTRY_END, maxsplit=1)[1]
    updated = before + REGISTRY_START + "\n" + rendered_table + REGISTRY_END + after
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(updated, encoding="utf-8")
    return registry_path


def relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _default_registry_text() -> str:
    return (
        "# Experiment Registry\n\n"
        "This registry is the canonical index of versioned experiment records.\n\n"
        f"{REGISTRY_START}\n{REGISTRY_END}\n"
    )


def _load_registry_entries(registry_path: Path) -> dict[str, RegistryEntry]:
    if not registry_path.exists():
        return {}
    text = registry_path.read_text(encoding="utf-8")
    if REGISTRY_START not in text or REGISTRY_END not in text:
        return {}
    table = text.split(REGISTRY_START, maxsplit=1)[1].split(REGISTRY_END, maxsplit=1)[0]
    entries: dict[str, RegistryEntry] = {}
    for line in table.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| ID ") or stripped.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 9:
            continue
        experiment_id = _extract_experiment_id(cells[0])
        if experiment_id is None:
            continue
        entries[experiment_id] = RegistryEntry(
            experiment_id=experiment_id,
            title=cells[1],
            status=cells[2],
            schema_version=cells[3],
            dataset_config=cells[4],
            primary_target=cells[5],
            artifact_dir=_extract_markdown_target(cells[6]),
            doc_path=_extract_markdown_target(cells[7]),
            conclusion=cells[8],
        )
    return entries


def _render_registry_table(entries: list[RegistryEntry]) -> str:
    lines = [
        (
            "| ID | Title | Status | Schema | Dataset / config | Primary target | "
            "Artifacts | Doc | Conclusion |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        artifact_label = Path(entry.artifact_dir).name
        doc_label = Path(entry.doc_path).name
        lines.append(
            "| {id} | {title} | {status} | {schema} | {dataset} | {target} | "
            "[{artifact_label}]({artifact}) | [{doc_label}]({doc}) | {conclusion} |".format(
                id=entry.experiment_id,
                title=_escape_cell(entry.title),
                status=_escape_cell(entry.status),
                schema=_escape_cell(entry.schema_version),
                dataset=_escape_cell(entry.dataset_config),
                target=_escape_cell(entry.primary_target),
                artifact_label=_escape_cell(artifact_label),
                artifact=entry.artifact_dir,
                doc_label=_escape_cell(doc_label),
                doc=entry.doc_path,
                conclusion=_escape_cell(entry.conclusion),
            )
        )
    return "\n".join(lines) + "\n"


def _extract_experiment_id(cell: str) -> str | None:
    match = EXPERIMENT_ID_PATTERN.search(cell)
    return match.group(0) if match else None


def _extract_markdown_target(cell: str) -> str:
    match = re.search(r"\]\(([^)]+)\)", cell)
    return match.group(1) if match else cell


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    return value
