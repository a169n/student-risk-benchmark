from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.generator.config import GeneratorConfig
from src.generator.students import StudentProfile
from src.generator.utils import clamp


def generate_attendance_records(
    profile: StudentProfile,
    *,
    topic_id: str,
    week_number: int,
    topic_difficulty: float,
    effective_state: dict[str, float | bool],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], float]:
    week_start = datetime.combine(config.start_date + timedelta(weeks=week_number - 1), datetime.min.time())
    records: list[dict[str, object]] = []

    if bool(effective_state["is_withdrawn_phase"]):
        probabilities = {
            "present": 0.02,
            "late": 0.04,
            "absent": 0.86,
            "excused": 0.08,
        }
    else:
        baseline = float(effective_state["baseline"])
        motivation = float(effective_state["motivation"])
        discipline = float(effective_state["discipline"])
        attendance_strength = clamp(
            0.18 + 0.50 * motivation + 0.18 * discipline + 0.14 * baseline - 0.12 * topic_difficulty,
            0.05,
            0.97,
        )
        late_probability = clamp(0.04 + (1 - discipline) * 0.18, 0.02, 0.25)
        excused_probability = clamp(0.01 + topic_difficulty * 0.03, 0.01, 0.08)
        present_probability = clamp(attendance_strength - late_probability * 0.5, 0.05, 0.94)
        absent_probability = max(0.0, 1.0 - present_probability - late_probability - excused_probability)
        total = present_probability + late_probability + absent_probability + excused_probability
        probabilities = {
            "present": present_probability / total,
            "late": late_probability / total,
            "absent": absent_probability / total,
            "excused": excused_probability / total,
        }

    attendance_values: list[float] = []
    statuses = ["present", "late", "absent", "excused"]
    weights = [probabilities[status] for status in statuses]

    for session_number in range(1, config.sessions_per_week + 1):
        session_type = "lecture" if session_number == 1 else "lab" if session_number == 2 else "tutorial"
        session_datetime = week_start + timedelta(days=min(5, (session_number - 1) * 2))
        attendance_status = str(rng.choice(statuses, p=np.array(weights, dtype=float)))

        records.append(
            {
                "attendance_id": f"att_{profile.student_id}_w{week_number:02d}_s{session_number:02d}",
                "student_id": profile.student_id,
                "course_id": profile.course_id,
                "topic_id": topic_id,
                "week_number": week_number,
                "session_date": session_datetime.date(),
                "session_type": session_type,
                "attendance_status": attendance_status,
            }
        )

        if attendance_status == "present":
            attendance_values.append(1.0)
        elif attendance_status == "late":
            attendance_values.append(0.85)
        elif attendance_status == "absent":
            attendance_values.append(0.0)

    attendance_rate = sum(attendance_values) / len(attendance_values) if attendance_values else 0.0
    return records, attendance_rate
