from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOPIC_TITLES = [
    "Programming Foundations and Tooling",
    "Variables, Types, and Expressions",
    "Conditionals and Boolean Logic",
    "Loops and Iteration",
    "Functions and Modular Thinking",
    "Collections and Nested Data",
    "File Handling and Exceptions",
    "Object-Oriented Basics",
    "Algorithmic Practice and Debugging",
    "Review, Integration, and Final Project Prep",
]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _extended_topic_titles(titles: list[str], num_weeks: int) -> list[str]:
    """Return enough topic titles for generated courses longer than the base template."""

    extended = list(titles)
    while len(extended) < num_weeks:
        week_number = len(extended) + 1
        extended.append(f"Extended Practice Week {week_number}")
    return extended


class CourseConfig(BaseModel):
    course_id: str = "course_prog_101"
    course_code: str = "CS101"
    course_name: str = "Introduction to Programming"
    term_label: str = "Fall 2026"
    course_description: str = (
        "Synthetic introductory programming course for Student Digital Twin research."
    )
    topic_titles: list[str] = Field(default_factory=lambda: list(DEFAULT_TOPIC_TITLES))


class OutputConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("data/artifacts")


class GeneratorConfig(BaseModel):
    schema_version: str = "1.2"
    trajectory_tuning: str = "v1_3_refined"
    num_students: int = 120
    num_weeks: int = 10
    num_groups: int = 3
    assignments_per_week: int = 2
    sessions_per_week: int = 2
    seed: int = 42
    pass_mark: float = 50.0
    withdrawal_rate: float = 0.05
    start_date: date = date(2026, 9, 1)
    course: CourseConfig = Field(default_factory=CourseConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    trajectory_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "stable_high": 0.26,
            "improving": 0.22,
            "declining": 0.22,
            "consistently_at_risk": 0.30,
        }
    )

    @model_validator(mode="after")
    def validate_generation_bounds(self) -> "GeneratorConfig":
        if self.num_students <= 0:
            raise ValueError("num_students must be positive")
        if self.num_weeks <= 0:
            raise ValueError("num_weeks must be positive")
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if self.assignments_per_week <= 0:
            raise ValueError("assignments_per_week must be positive")
        if self.sessions_per_week <= 0:
            raise ValueError("sessions_per_week must be positive")
        if not 0 <= self.withdrawal_rate < 1:
            raise ValueError("withdrawal_rate must be in [0, 1)")
        if self.pass_mark < 0 or self.pass_mark > 100:
            raise ValueError("pass_mark must be in [0, 100]")
        if len(self.course.topic_titles) < self.num_weeks:
            raise ValueError("course.topic_titles must include at least num_weeks entries")
        if not self.trajectory_weights:
            raise ValueError("trajectory_weights must not be empty")
        if any(weight <= 0 for weight in self.trajectory_weights.values()):
            raise ValueError("trajectory_weights must be positive")
        if self.trajectory_tuning not in {"v1_2_baseline", "v1_3_refined"}:
            raise ValueError("trajectory_tuning must be one of: v1_2_baseline, v1_3_refined")
        return self

    @property
    def raw_output_dir(self) -> Path:
        return _resolve_repo_path(self.outputs.raw_dir)

    @property
    def processed_output_dir(self) -> Path:
        return _resolve_repo_path(self.outputs.processed_dir)

    @property
    def artifacts_output_dir(self) -> Path:
        return _resolve_repo_path(self.outputs.artifacts_dir)

    @property
    def contract_path(self) -> Path:
        return REPO_ROOT / "packages" / "contracts" / "schema_versions" / "schema_v1.2.yaml"


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, SERVICE_ROOT / path, REPO_ROOT / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config path: {config_path}")


def load_generator_config(
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    output_root: str | Path | None = None,
    num_students_override: int | None = None,
    num_weeks_override: int | None = None,
    num_groups_override: int | None = None,
    assignments_per_week_override: int | None = None,
    sessions_per_week_override: int | None = None,
) -> tuple[GeneratorConfig, Path]:
    resolved_config_path = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}

    if seed_override is not None:
        raw_payload["seed"] = seed_override
    if num_students_override is not None:
        raw_payload["num_students"] = num_students_override
    if num_weeks_override is not None:
        raw_payload["num_weeks"] = num_weeks_override
    if num_groups_override is not None:
        raw_payload["num_groups"] = num_groups_override
    if assignments_per_week_override is not None:
        raw_payload["assignments_per_week"] = assignments_per_week_override
    if sessions_per_week_override is not None:
        raw_payload["sessions_per_week"] = sessions_per_week_override

    num_weeks = int(
        raw_payload.get("num_weeks", GeneratorConfig.model_fields["num_weeks"].default)
    )
    course_payload = dict(raw_payload.get("course") or {})
    topic_titles = course_payload.get("topic_titles") or list(DEFAULT_TOPIC_TITLES)
    course_payload["topic_titles"] = _extended_topic_titles(list(topic_titles), num_weeks)
    raw_payload["course"] = course_payload

    if output_root is not None:
        root = Path(output_root)
        raw_payload["outputs"] = {
            "raw_dir": root / "raw",
            "processed_dir": root / "processed",
            "artifacts_dir": root / "artifacts",
        }

    config = GeneratorConfig.model_validate(raw_payload)
    return config, resolved_config_path
