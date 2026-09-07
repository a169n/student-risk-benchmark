from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.generator.config import GeneratorConfig
from src.generator.utils import clamp, normalized_ratio, recent_trend

# Provisional risk-score thresholds (v1.2 calibration).
# These define the mapping from continuous risk_score to categorical risk_level.
# Recalibrated from 0.40/0.70 (v1.1) to produce a more realistic distribution:
#   low: risk_score < RISK_THRESHOLD_LOW
#   medium: RISK_THRESHOLD_LOW <= risk_score < RISK_THRESHOLD_HIGH
#   high: risk_score >= RISK_THRESHOLD_HIGH
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_HIGH = 0.55
RISK_SCORE_SCALE = 0.915
RISK_SCORE_OFFSET = -0.04


def _mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _score_trend_from_weekly_scores(student_scores: pd.DataFrame) -> float:
    weekly_values = (
        student_scores.groupby("week_number", sort=True)["normalized_score"].mean().dropna().tolist()
    )
    return round(recent_trend(weekly_values, scale=100.0), 4)


def _trend_from_series(student_series: pd.Series, *, scale: float) -> float:
    return round(recent_trend(student_series.dropna().tolist(), scale=scale), 4)


def build_student_twin_snapshots(
    *,
    students_df: pd.DataFrame,
    course_topics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    weekly_activity_df: pd.DataFrame,
    config: GeneratorConfig,
) -> pd.DataFrame:
    topic_lookup = course_topics_df.set_index("week_number").to_dict("index")
    assignment_context = assignments_df.merge(
        course_topics_df[["topic_id", "week_number", "topic_difficulty"]],
        on="topic_id",
        how="left",
    )
    submission_context = submissions_df.merge(
        assignment_context[
            [
                "assignment_id",
                "week_number",
                "assignment_type",
                "max_score",
                "is_required",
                "topic_id",
                "topic_difficulty",
            ]
        ],
        on="assignment_id",
        how="left",
    )
    submission_context["normalized_score"] = (
        submission_context["score"] / submission_context["max_score"] * 100.0
    )

    attendance_context = attendance_df.copy()
    attendance_context["attendance_value"] = attendance_context["attendance_status"].map(
        {"present": 1.0, "late": 0.85, "absent": 0.0, "excused": pd.NA}
    )

    snapshot_records: list[dict[str, object]] = []
    for student in students_df.to_dict("records"):
        student_id = str(student["student_id"])
        student_submissions = submission_context[submission_context["student_id"] == student_id].copy()
        student_attendance = attendance_context[attendance_context["student_id"] == student_id].copy()
        student_activity = weekly_activity_df[weekly_activity_df["student_id"] == student_id].copy()

        for week_number in range(1, config.num_weeks + 1):
            topic_context = topic_lookup[week_number]
            submission_slice = student_submissions[student_submissions["week_number"] <= week_number].copy()
            attendance_slice = student_attendance[student_attendance["week_number"] <= week_number].copy()
            activity_slice = student_activity[student_activity["week_number"] <= week_number].copy()

            assignment_scores = submission_slice.loc[
                (submission_slice["assignment_type"] == "assignment")
                & submission_slice["normalized_score"].notna(),
                "normalized_score",
            ]
            quiz_scores = submission_slice.loc[
                (submission_slice["assignment_type"] == "quiz") & submission_slice["normalized_score"].notna(),
                "normalized_score",
            ]
            avg_assignment_score = _mean_or_none(assignment_scores)
            avg_quiz_score = _mean_or_none(quiz_scores)
            has_assignment_score_to_date = avg_assignment_score is not None
            has_quiz_score_to_date = avg_quiz_score is not None

            valid_attendance = attendance_slice["attendance_value"].dropna()
            attendance_rate = float(valid_attendance.mean() or 0.0) if not valid_attendance.empty else 0.0

            due_required = submission_slice[
                (submission_slice["is_required"]) & (submission_slice["submission_status"] != "excused")
            ]
            due_required_count = len(due_required)
            on_time_count = float(due_required["is_on_time"].eq(True).sum()) if due_required_count else 0.0
            on_time_rate = normalized_ratio(on_time_count, float(due_required_count))
            missed_assignments = int((due_required["submission_status"] == "missing").sum())
            late_submissions = int((submission_slice["submission_status"] == "late").sum())
            avg_attempt_count = _mean_or_none(due_required["attempt_count"].astype(float))

            activity_score_to_date = (
                float(activity_slice["activity_score"].mean()) if not activity_slice.empty else 0.0
            )
            time_spent_to_date = (
                float(activity_slice["time_on_platform_minutes"].sum()) if not activity_slice.empty else 0.0
            )

            score_trend = _score_trend_from_weekly_scores(student_submissions[student_submissions["week_number"] <= week_number])
            activity_trend = _trend_from_series(
                activity_slice.sort_values("week_number")["activity_score"], scale=100.0
            )
            weekly_attendance_rates = (
                student_attendance[student_attendance["week_number"] <= week_number]
                .groupby("week_number", sort=True)["attendance_value"]
                .mean()
            )
            attendance_trend = _trend_from_series(weekly_attendance_rates, scale=1.0)

            current_topic_scores = submission_slice.loc[
                (submission_slice["topic_id"] == topic_context["topic_id"])
                & submission_slice["normalized_score"].notna(),
                "normalized_score",
            ]
            current_topic_mastery = _mean_or_none(current_topic_scores)
            if current_topic_mastery is None:
                fallback_score = avg_quiz_score if avg_quiz_score is not None else avg_assignment_score
                current_topic_mastery = round(float(fallback_score or 50.0), 4)

            topic_mastery = (
                submission_slice.loc[submission_slice["normalized_score"].notna(), ["topic_id", "normalized_score"]]
                .groupby("topic_id", sort=True)["normalized_score"]
                .mean()
            )
            overall_mastery = (
                round(float(topic_mastery.mean()), 4) if not topic_mastery.empty else round(float(current_topic_mastery), 4)
            )

            assignment_norm = (avg_assignment_score or overall_mastery or 50.0) / 100.0
            quiz_norm = (avg_quiz_score if avg_quiz_score is not None else avg_assignment_score or overall_mastery or 50.0) / 100.0
            overall_mastery_norm = (overall_mastery or avg_assignment_score or 50.0) / 100.0
            positive_score_trend = max(score_trend, 0.0)
            late_penalty = normalized_ratio(float(late_submissions), float(max(due_required_count, 1)))
            missed_penalty = normalized_ratio(float(missed_assignments), float(max(due_required_count, 1)))

            engagement_index = 100.0 * (
                0.40 * attendance_rate
                + 0.30 * (activity_score_to_date / 100.0)
                + 0.30 * on_time_rate
            )
            performance_index = 100.0 * (
                0.45 * assignment_norm
                + 0.25 * quiz_norm
                + 0.20 * overall_mastery_norm
                + 0.10 * positive_score_trend
            )
            discipline_index = 100.0 * (
                0.50 * on_time_rate + 0.30 * (1.0 - late_penalty) + 0.20 * (1.0 - missed_penalty)
            )

            negative_trend_penalty = (
                max(-score_trend, 0.0) + max(-activity_trend, 0.0) + max(-attendance_trend, 0.0)
            ) / 3.0
            raw_risk_score = clamp(
                0.30 * (1.0 - performance_index / 100.0)
                + 0.25 * (1.0 - engagement_index / 100.0)
                + 0.20 * (1.0 - discipline_index / 100.0)
                + 0.15 * missed_penalty
                + 0.10 * negative_trend_penalty,
                0.0,
                1.0,
            )
            # risk_level is a teacher-facing heuristic label, not the ML ground truth.
            # The scale/offset terms preserve monotonicity while improving the snapshot
            # distribution for realism checks introduced in v1.2.
            risk_score = clamp(
                raw_risk_score * RISK_SCORE_SCALE + RISK_SCORE_OFFSET,
                0.0,
                1.0,
            )
            if risk_score < RISK_THRESHOLD_LOW:
                risk_level = "low"
            elif risk_score < RISK_THRESHOLD_HIGH:
                risk_level = "medium"
            else:
                risk_level = "high"

            # This remains a snapshot-time heuristic estimate. The realized ML target
            # lives in final_results.final_grade after course completion.
            predicted_final_grade = clamp(
                0.50 * performance_index
                + 0.20 * overall_mastery
                + 0.15 * discipline_index
                + 0.15 * engagement_index,
                0.0,
                100.0,
            )
            snapshot_date = config.start_date + timedelta(weeks=week_number - 1, days=6)

            snapshot_records.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week_number:02d}",
                    "student_id": student_id,
                    "course_id": student["course_id"],
                    "week_number": week_number,
                    "snapshot_date": snapshot_date,
                    "attendance_rate_to_date": round(attendance_rate, 4),
                    "avg_assignment_score_to_date": None if avg_assignment_score is None else round(avg_assignment_score, 2),
                    "avg_quiz_score_to_date": None if avg_quiz_score is None else round(avg_quiz_score, 2),
                    "has_assignment_score_to_date": has_assignment_score_to_date,
                    "has_quiz_score_to_date": has_quiz_score_to_date,
                    "on_time_submission_rate_to_date": round(on_time_rate, 4),
                    "missed_assignments_to_date": missed_assignments,
                    "late_submissions_to_date": late_submissions,
                    "avg_attempt_count_to_date": None if avg_attempt_count is None else round(avg_attempt_count, 4),
                    "activity_score_to_date": round(activity_score_to_date, 2),
                    "time_spent_to_date": round(time_spent_to_date, 2),
                    "score_trend_3w": score_trend,
                    "activity_trend_3w": activity_trend,
                    "attendance_trend_3w": attendance_trend,
                    "current_topic_mastery": round(float(current_topic_mastery), 2),
                    "overall_mastery": round(float(overall_mastery), 2),
                    "engagement_index": round(engagement_index, 2),
                    "performance_index": round(performance_index, 2),
                    "discipline_index": round(discipline_index, 2),
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "predicted_final_grade": round(predicted_final_grade, 2),
                }
            )

    return pd.DataFrame(snapshot_records)
