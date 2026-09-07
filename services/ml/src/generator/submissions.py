from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.generator.students import StudentProfile
from src.generator.utils import clamp


def generate_submission_records(
    profile: StudentProfile,
    *,
    assignments_for_week: pd.DataFrame,
    week_number: int,
    topic_difficulty: float,
    effective_state: dict[str, float | bool],
    attendance_rate: float,
    activity_score: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    baseline = float(effective_state["baseline"])
    motivation = float(effective_state["motivation"])
    discipline = float(effective_state["discipline"])

    for assignment in assignments_for_week.to_dict("records"):
        assignment_type = str(assignment["assignment_type"])
        due_at = pd.Timestamp(assignment["due_at"])
        difficulty = clamp(
            topic_difficulty + (0.04 if assignment_type == "quiz" else 0.02 if assignment_type == "lab" else 0.0),
            0.0,
            1.0,
        )

        if bool(effective_state["is_withdrawn_phase"]):
            submission_status = str(
                rng.choice(["missing", "excused", "late"], p=np.array([0.86, 0.08, 0.06]))
            )
        else:
            missing_probability = clamp(
                0.02 + (1 - discipline) * 0.25 + (1 - motivation) * 0.10 + difficulty * 0.05,
                0.01,
                0.82,
            )
            late_probability = clamp(
                0.05 + (1 - discipline) * 0.35 + (1 - motivation) * 0.08,
                0.03,
                0.65,
            )
            excused_probability = 0.01
            submitted_probability = max(
                0.0, 1.0 - missing_probability - late_probability - excused_probability
            )
            total = submitted_probability + late_probability + missing_probability + excused_probability
            submission_status = str(
                rng.choice(
                    ["submitted", "late", "missing", "excused"],
                    p=np.array(
                        [
                            submitted_probability / total,
                            late_probability / total,
                            missing_probability / total,
                            excused_probability / total,
                        ]
                    ),
                )
            )

        score = None
        submitted_at = None
        is_on_time: bool | None = None
        attempt_count = 0

        if submission_status in {"submitted", "late"}:
            base_score = clamp(
                0.38
                + 0.42 * baseline
                + 0.08 * motivation
                + 0.07 * discipline
                + 0.05 * attendance_rate
                + 0.04 * (activity_score / 100.0)
                - 0.18 * difficulty
                + float(rng.normal(0.0, 0.06)),
                0.05,
                0.99,
            )
            attempt_count = 1 + int(rng.random() < (0.12 + 0.20 * motivation + 0.10 * (1 - base_score)))
            if attempt_count == 2 and rng.random() < (0.05 + 0.08 * motivation):
                attempt_count += 1

            score_ratio = base_score + 0.03 * (attempt_count - 1)
            if submission_status == "late":
                score_ratio -= 0.05
                submitted_at = due_at + timedelta(hours=float(rng.uniform(3, 72)))
                is_on_time = False
            else:
                submitted_at = due_at - timedelta(hours=float(rng.uniform(3, 36)))
                is_on_time = True

            score = round(float(assignment["max_score"]) * clamp(score_ratio, 0.0, 1.0), 2)
        elif submission_status == "missing":
            is_on_time = False

        records.append(
            {
                "submission_id": f"sub_{profile.student_id}_{assignment['assignment_id']}",
                "student_id": profile.student_id,
                "assignment_id": assignment["assignment_id"],
                "submitted_at": submitted_at,
                "submission_status": submission_status,
                "score": score,
                "is_on_time": is_on_time,
                "attempt_count": attempt_count,
            }
        )

    return records
