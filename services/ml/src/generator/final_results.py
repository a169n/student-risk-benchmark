from __future__ import annotations

import pandas as pd

from src.generator.config import GeneratorConfig
from src.generator.students import StudentProfile
from src.generator.utils import clamp


def _series_mean_or_default(series: pd.Series, default: float) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return default
    return float(non_null.mean())


def generate_final_results(
    profiles: list[StudentProfile],
    *,
    attendance_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    config: GeneratorConfig,
) -> pd.DataFrame:
    # End-of-course outcomes are the experimental targets for supervised ML.
    # Keep them separate from weekly heuristic labels to avoid target conflation.
    assignment_context = assignments_df[["assignment_id", "assignment_type", "max_score", "is_required"]].copy()
    submission_context = submissions_df.merge(assignment_context, on="assignment_id", how="left")
    attendance_context = attendance_df.copy()

    records: list[dict[str, object]] = []
    for profile in profiles:
        if profile.enrollment_status == "withdrawn":
            completion_status = "withdrawn"
            final_grade = None
            passed = None
            completed_weeks = (profile.withdrawal_week - 1) if profile.withdrawal_week else config.num_weeks - 1
        else:
            completion_status = "completed"
            student_submissions = submission_context[submission_context["student_id"] == profile.student_id].copy()
            student_submissions["normalized_score"] = (
                student_submissions["score"] / student_submissions["max_score"] * 100.0
            )

            assignment_avg = _series_mean_or_default(
                student_submissions.loc[
                    (student_submissions["assignment_type"] == "assignment")
                    & student_submissions["normalized_score"].notna(),
                    "normalized_score",
                ],
                0.0,
            )
            quiz_avg = _series_mean_or_default(
                student_submissions.loc[
                    (student_submissions["assignment_type"] == "quiz")
                    & student_submissions["normalized_score"].notna(),
                    "normalized_score",
                ],
                assignment_avg,
            )

            student_attendance = attendance_context[attendance_context["student_id"] == profile.student_id].copy()
            attendance_values = student_attendance["attendance_status"].map(
                {"present": 1.0, "late": 0.85, "absent": 0.0, "excused": pd.NA}
            )
            valid_attendance = attendance_values.dropna()
            attendance_rate = float(valid_attendance.mean()) if not valid_attendance.empty else 0.0

            due_required = student_submissions[
                (student_submissions["is_required"]) & (student_submissions["submission_status"] != "excused")
            ]
            on_time_rate = float(due_required["is_on_time"].eq(True).mean()) if not due_required.empty else 0.0

            final_grade = round(
                clamp(
                    0.55 * assignment_avg
                    + 0.25 * quiz_avg
                    + 0.10 * attendance_rate * 100.0
                    + 0.10 * on_time_rate * 100.0,
                    0.0,
                    100.0,
                ),
                2,
            )
            passed = bool(final_grade >= config.pass_mark)
            completed_weeks = config.num_weeks

        records.append(
            {
                "final_result_id": f"final_{profile.student_id}",
                "student_id": profile.student_id,
                "course_id": profile.course_id,
                "completion_status": completion_status,
                "final_grade": final_grade,
                "passed": passed,
                "completed_weeks": completed_weeks,
            }
        )

    return pd.DataFrame(records)
