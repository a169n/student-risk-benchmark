from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generator.config import GeneratorConfig
from src.generator.utils import clamp, stable_probability_weights


@dataclass(frozen=True)
class StudentProfile:
    student_id: str
    student_code: str
    course_id: str
    cohort_label: str
    enrollment_status: str
    baseline_level: float
    motivation_level: float
    discipline_level: float
    trajectory_type: str
    withdrawal_week: int | None


TRAJECTORY_RANGES: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "stable_high": ((0.74, 0.95), (0.72, 0.95), (0.70, 0.93)),
    "improving": ((0.34, 0.58), (0.38, 0.68), (0.40, 0.70)),
    "declining": ((0.60, 0.83), (0.56, 0.81), (0.52, 0.78)),
    "consistently_at_risk": ((0.10, 0.30), (0.08, 0.32), (0.10, 0.36)),
}

TRAJECTORY_RANGES_BY_TUNING: dict[
    str,
    dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
] = {
    "v1_2_baseline": TRAJECTORY_RANGES,
    "v1_3_refined": {
        "stable_high": ((0.74, 0.95), (0.72, 0.95), (0.70, 0.93)),
        "improving": ((0.38, 0.62), (0.42, 0.72), (0.44, 0.74)),
        "declining": ((0.56, 0.78), (0.50, 0.74), (0.46, 0.70)),
        "consistently_at_risk": ((0.10, 0.30), (0.08, 0.32), (0.10, 0.36)),
    },
}


def _sample_parameter_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    return float(rng.uniform(value_range[0], value_range[1]))


def _sample_hidden_parameters(
    rng: np.random.Generator, trajectory_type: str, *, trajectory_tuning: str
) -> tuple[float, float, float]:
    trajectory_ranges = TRAJECTORY_RANGES_BY_TUNING[trajectory_tuning]
    baseline_range, motivation_range, discipline_range = trajectory_ranges[trajectory_type]
    return (
        _sample_parameter_range(rng, baseline_range),
        _sample_parameter_range(rng, motivation_range),
        _sample_parameter_range(rng, discipline_range),
    )


def generate_student_profiles(
    config: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[StudentProfile]]:
    trajectory_labels, trajectory_probabilities = stable_probability_weights(config.trajectory_weights)
    trajectory_choices = rng.choice(
        trajectory_labels,
        size=config.num_students,
        p=np.array(trajectory_probabilities, dtype=float),
        replace=True,
    )

    provisional_profiles: list[StudentProfile] = []
    withdrawal_scores: list[float] = []
    for index, trajectory_type in enumerate(trajectory_choices, start=1):
        baseline, motivation, discipline = _sample_hidden_parameters(
            rng,
            str(trajectory_type),
            trajectory_tuning=config.trajectory_tuning,
        )
        student_id = f"student_{index:03d}"
        student_code = f"STU-{index:03d}"
        cohort_label = f"G{((index - 1) % config.num_groups) + 1}"
        risk_propensity = (
            0.46 * (1 - baseline)
            + 0.30 * (1 - motivation)
            + 0.24 * (1 - discipline)
            + (0.10 if trajectory_type == "consistently_at_risk" else 0.0)
            + (
                0.06
                if trajectory_type == "declining" and config.trajectory_tuning == "v1_3_refined"
                else 0.04
                if trajectory_type == "declining"
                else 0.0
            )
        )
        withdrawal_scores.append(risk_propensity + float(rng.uniform(0.0, 0.08)))
        provisional_profiles.append(
            StudentProfile(
                student_id=student_id,
                student_code=student_code,
                course_id=config.course.course_id,
                cohort_label=cohort_label,
                enrollment_status="completed",
                baseline_level=baseline,
                motivation_level=motivation,
                discipline_level=discipline,
                trajectory_type=str(trajectory_type),
                withdrawal_week=None,
            )
        )

    withdrawal_count = int(round(config.num_students * config.withdrawal_rate))
    withdrawal_indices: set[int] = set()
    if withdrawal_count > 0:
        ranked_indices = np.argsort(np.array(withdrawal_scores))[::-1]
        withdrawal_indices = set(int(index) for index in ranked_indices[:withdrawal_count])

    student_records: list[dict[str, object]] = []
    final_profiles: list[StudentProfile] = []
    for index, profile in enumerate(provisional_profiles):
        withdrawal_week = None
        enrollment_status = "completed"
        if index in withdrawal_indices:
            min_withdraw_week = min(config.num_weeks - 1, max(4, config.num_weeks // 2))
            max_withdraw_week = max(min_withdraw_week, config.num_weeks - 1)
            withdrawal_week = int(rng.integers(min_withdraw_week, max_withdraw_week + 1))
            enrollment_status = "withdrawn"

        final_profile = StudentProfile(
            student_id=profile.student_id,
            student_code=profile.student_code,
            course_id=profile.course_id,
            cohort_label=profile.cohort_label,
            enrollment_status=enrollment_status,
            baseline_level=profile.baseline_level,
            motivation_level=profile.motivation_level,
            discipline_level=profile.discipline_level,
            trajectory_type=profile.trajectory_type,
            withdrawal_week=withdrawal_week,
        )
        final_profiles.append(final_profile)
        student_records.append(
            {
                "student_id": final_profile.student_id,
                "student_code": final_profile.student_code,
                "course_id": final_profile.course_id,
                "cohort_label": final_profile.cohort_label,
                "enrollment_status": final_profile.enrollment_status,
                "baseline_level": round(final_profile.baseline_level, 4),
                "motivation_level": round(final_profile.motivation_level, 4),
                "discipline_level": round(final_profile.discipline_level, 4),
                "trajectory_type": final_profile.trajectory_type,
            }
        )

    students_df = pd.DataFrame(student_records)
    return students_df, final_profiles


def compute_effective_state(
    profile: StudentProfile,
    *,
    week_number: int,
    total_weeks: int,
    trajectory_tuning: str,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    progress = 0.0 if total_weeks <= 1 else (week_number - 1) / (total_weeks - 1)

    if trajectory_tuning == "v1_3_refined":
        if profile.trajectory_type == "stable_high":
            deltas = (0.10 - 0.01 * progress, 0.11 - 0.01 * progress, 0.09 - 0.01 * progress)
        elif profile.trajectory_type == "declining":
            decline = max(0.0, (week_number - 2) / max(total_weeks - 2, 1))
            deltas = (0.02 - 0.42 * decline, -0.01 - 0.38 * decline, -0.02 - 0.34 * decline)
        elif profile.trajectory_type == "improving":
            recovery = max(0.0, (week_number - 3) / max(total_weeks - 3, 1))
            deltas = (-0.04 + 0.36 * recovery, -0.02 + 0.32 * recovery, -0.01 + 0.28 * recovery)
        else:
            deltas = (-0.26 - 0.08 * progress, -0.24 - 0.07 * progress, -0.22 - 0.08 * progress)
    else:
        if profile.trajectory_type == "stable_high":
            deltas = (0.10 - 0.02 * progress, 0.10 - 0.01 * progress, 0.08 - 0.01 * progress)
        elif profile.trajectory_type == "declining":
            decline = max(0.0, (week_number - 3) / max(total_weeks - 3, 1))
            deltas = (0.05 - 0.34 * decline, 0.03 - 0.28 * decline, 0.02 - 0.24 * decline)
        elif profile.trajectory_type == "improving":
            recovery = max(0.0, (week_number - 4) / max(total_weeks - 4, 1))
            deltas = (-0.08 + 0.30 * recovery, -0.05 + 0.26 * recovery, -0.04 + 0.22 * recovery)
        else:
            deltas = (-0.26 - 0.08 * progress, -0.24 - 0.07 * progress, -0.22 - 0.08 * progress)

    baseline = clamp(profile.baseline_level + deltas[0] + float(rng.normal(0.0, 0.025)), 0.02, 0.99)
    motivation = clamp(
        profile.motivation_level + deltas[1] + float(rng.normal(0.0, 0.03)), 0.02, 0.99
    )
    discipline = clamp(
        profile.discipline_level + deltas[2] + float(rng.normal(0.0, 0.025)), 0.02, 0.99
    )

    is_withdrawn_phase = bool(profile.withdrawal_week and week_number > profile.withdrawal_week)
    if is_withdrawn_phase:
        baseline = clamp(baseline * 0.45, 0.02, 0.65)
        motivation = clamp(motivation * 0.12, 0.01, 0.24)
        discipline = clamp(discipline * 0.15, 0.01, 0.28)

    return {
        "baseline": round(baseline, 4),
        "motivation": round(motivation, 4),
        "discipline": round(discipline, 4),
        "is_withdrawn_phase": is_withdrawn_phase,
    }
