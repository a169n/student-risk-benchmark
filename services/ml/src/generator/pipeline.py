from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.generator.activity import generate_weekly_activity_record
from src.generator.attendance import generate_attendance_records
from src.generator.config import GeneratorConfig
from src.generator.courses import generate_assignments, generate_course, generate_course_topics
from src.generator.final_results import generate_final_results
from src.generator.snapshots import build_student_twin_snapshots
from src.generator.students import StudentProfile, compute_effective_state, generate_student_profiles
from src.generator.submissions import generate_submission_records
from src.validation.quality import ValidationSuite
from src.validation.realism import RealismAudit, write_reports


RAW_TABLES = [
    "students",
    "courses",
    "course_topics",
    "assignments",
    "attendance",
    "submissions",
    "weekly_activity",
    "final_results",
]
PROCESSED_TABLES = ["student_twin_snapshots"]


@dataclass
class PipelineSummary:
    config_path: Path | None
    seed: int
    raw_output_dir: Path
    processed_output_dir: Path
    realism_report_dir: Path
    row_counts: dict[str, int]
    risk_distribution: dict[str, int]
    withdrawal_count: int


@dataclass
class PipelineResult:
    datasets: dict[str, pd.DataFrame]
    summary: PipelineSummary


def _topic_row_for_week(course_topics_df: pd.DataFrame, week_number: int) -> dict[str, object]:
    topic_row = course_topics_df.loc[course_topics_df["week_number"] == week_number].iloc[0]
    return topic_row.to_dict()


def _assignments_for_week(assignments_df: pd.DataFrame, topic_id: str) -> pd.DataFrame:
    return assignments_df.loc[assignments_df["topic_id"] == topic_id].copy()


def _sort_dataset(name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    sort_keys = {
        "students": ["student_id"],
        "courses": ["course_id"],
        "course_topics": ["week_number"],
        "assignments": ["assignment_id"],
        "attendance": ["attendance_id"],
        "submissions": ["submission_id"],
        "weekly_activity": ["weekly_activity_id"],
        "final_results": ["final_result_id"],
        "student_twin_snapshots": ["snapshot_id"],
    }
    return dataframe.sort_values(sort_keys[name]).reset_index(drop=True)


def _write_outputs(
    datasets: dict[str, pd.DataFrame],
    *,
    config: GeneratorConfig,
    validation_suite: ValidationSuite,
    skip_parquet: bool,
) -> None:
    config.raw_output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, dataframe in datasets.items():
        ordered_columns = [field["name"] for field in validation_suite.table_specs[table_name]["fields"]]
        output_df = dataframe[ordered_columns]
        target_dir = config.raw_output_dir if table_name in RAW_TABLES else config.processed_output_dir
        output_df.to_csv(target_dir / f"{table_name}.csv", index=False)

        if table_name == "student_twin_snapshots" and not skip_parquet:
            output_df.to_parquet(target_dir / f"{table_name}.parquet", index=False)


def _generate_weekly_raw_data(
    profiles: list[StudentProfile],
    *,
    course_topics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attendance_records: list[dict[str, object]] = []
    activity_records: list[dict[str, object]] = []
    submission_records: list[dict[str, object]] = []

    for profile in profiles:
        for week_number in range(1, config.num_weeks + 1):
            topic_row = _topic_row_for_week(course_topics_df, week_number)
            effective_state = compute_effective_state(
                profile,
                week_number=week_number,
                total_weeks=config.num_weeks,
                trajectory_tuning=config.trajectory_tuning,
                rng=rng,
            )
            weekly_attendance_records, attendance_rate = generate_attendance_records(
                profile,
                topic_id=str(topic_row["topic_id"]),
                week_number=week_number,
                topic_difficulty=float(topic_row["topic_difficulty"]),
                effective_state=effective_state,
                config=config,
                rng=rng,
            )
            attendance_records.extend(weekly_attendance_records)

            activity_record = generate_weekly_activity_record(
                profile,
                topic_id=str(topic_row["topic_id"]),
                week_number=week_number,
                topic_difficulty=float(topic_row["topic_difficulty"]),
                effective_state=effective_state,
                attendance_rate=attendance_rate,
                rng=rng,
            )
            activity_records.append(activity_record)

            weekly_assignments = _assignments_for_week(assignments_df, str(topic_row["topic_id"]))
            submission_records.extend(
                generate_submission_records(
                    profile,
                    assignments_for_week=weekly_assignments,
                    week_number=week_number,
                    topic_difficulty=float(topic_row["topic_difficulty"]),
                    effective_state=effective_state,
                    attendance_rate=attendance_rate,
                    activity_score=float(activity_record["activity_score"]),
                    rng=rng,
                )
            )

    return (
        pd.DataFrame(attendance_records),
        pd.DataFrame(activity_records),
        pd.DataFrame(submission_records),
    )


def run_generation_pipeline(
    config: GeneratorConfig,
    *,
    config_path: Path | None = None,
    skip_parquet: bool = False,
) -> PipelineResult:
    rng = np.random.default_rng(config.seed)

    courses_df = generate_course(config)
    course_topics_df = generate_course_topics(config, rng)
    assignments_df = generate_assignments(config, course_topics_df)
    students_df, profiles = generate_student_profiles(config, rng)
    attendance_df, weekly_activity_df, submissions_df = _generate_weekly_raw_data(
        profiles,
        course_topics_df=course_topics_df,
        assignments_df=assignments_df,
        config=config,
        rng=rng,
    )
    final_results_df = generate_final_results(
        profiles,
        attendance_df=attendance_df,
        assignments_df=assignments_df,
        submissions_df=submissions_df,
        config=config,
    )
    snapshots_df = build_student_twin_snapshots(
        students_df=students_df,
        course_topics_df=course_topics_df,
        assignments_df=assignments_df,
        attendance_df=attendance_df,
        submissions_df=submissions_df,
        weekly_activity_df=weekly_activity_df,
        config=config,
    )

    datasets = {
        "students": _sort_dataset("students", students_df),
        "courses": _sort_dataset("courses", courses_df),
        "course_topics": _sort_dataset("course_topics", course_topics_df),
        "assignments": _sort_dataset("assignments", assignments_df),
        "attendance": _sort_dataset("attendance", attendance_df),
        "submissions": _sort_dataset("submissions", submissions_df),
        "weekly_activity": _sort_dataset("weekly_activity", weekly_activity_df),
        "final_results": _sort_dataset("final_results", final_results_df),
        "student_twin_snapshots": _sort_dataset("student_twin_snapshots", snapshots_df),
    }

    validation_suite = ValidationSuite(config.contract_path, config)
    validation_suite.assert_valid(datasets)
    _write_outputs(datasets, config=config, validation_suite=validation_suite, skip_parquet=skip_parquet)
    realism_report_dir = write_reports(
        RealismAudit().run(datasets),
        config.artifacts_output_dir / "reports",
    )

    summary = PipelineSummary(
        config_path=config_path,
        seed=config.seed,
        raw_output_dir=config.raw_output_dir,
        processed_output_dir=config.processed_output_dir,
        realism_report_dir=realism_report_dir,
        row_counts={name: len(frame) for name, frame in datasets.items()},
        risk_distribution={
            risk_level: int(
                datasets["student_twin_snapshots"]["risk_level"].value_counts().to_dict().get(risk_level, 0)
            )
            for risk_level in ["low", "medium", "high"]
        },
        withdrawal_count=int((datasets["final_results"]["completion_status"] == "withdrawn").sum()),
    )
    return PipelineResult(datasets=datasets, summary=summary)
