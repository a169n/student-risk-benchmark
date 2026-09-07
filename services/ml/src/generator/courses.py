from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.generator.config import GeneratorConfig
from src.generator.utils import clamp


def generate_course(config: GeneratorConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "course_id": config.course.course_id,
                "course_code": config.course.course_code,
                "course_name": config.course.course_name,
                "term_label": config.course.term_label,
                "duration_weeks": config.num_weeks,
                "grading_policy_pass_mark": float(config.pass_mark),
                "course_description": config.course.course_description,
            }
        ]
    )


def generate_course_topics(config: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    topic_records: list[dict[str, object]] = []
    difficulty_baseline = np.linspace(0.35, 0.75, config.num_weeks)
    for week_number in range(1, config.num_weeks + 1):
        if week_number == config.num_weeks:
            topic_type = "review"
        elif week_number % 4 == 0:
            topic_type = "lab"
        elif week_number % 5 == 0:
            topic_type = "assessment_week"
        else:
            topic_type = "lecture"

        topic_records.append(
            {
                "topic_id": f"topic_w{week_number:02d}",
                "course_id": config.course.course_id,
                "week_number": week_number,
                "topic_title": config.course.topic_titles[week_number - 1],
                "topic_type": topic_type,
                "planned_assignment_count": config.assignments_per_week,
                "topic_difficulty": round(
                    clamp(difficulty_baseline[week_number - 1] + float(rng.normal(0.0, 0.05)), 0.15, 0.95),
                    4,
                ),
            }
        )
    return pd.DataFrame(topic_records)


def _assignment_types_for_week(assignments_per_week: int) -> list[str]:
    if assignments_per_week <= 1:
        return ["assignment"]
    assignment_types = ["assignment", "quiz"]
    while len(assignment_types) < assignments_per_week:
        assignment_types.append("lab")
    return assignment_types


def generate_assignments(config: GeneratorConfig, topics_df: pd.DataFrame) -> pd.DataFrame:
    assignment_records: list[dict[str, object]] = []
    assignment_types = _assignment_types_for_week(config.assignments_per_week)
    total_assignments = config.num_weeks * config.assignments_per_week
    weight_percent = round(100.0 / total_assignments, 4)

    for topic in topics_df.to_dict("records"):
        week_number = int(topic["week_number"])
        week_start = datetime.combine(config.start_date + timedelta(weeks=week_number - 1), datetime.min.time())

        for index, assignment_type in enumerate(assignment_types, start=1):
            assignment_id = f"asg_w{week_number:02d}_{index:02d}"
            due_at = week_start + timedelta(days=min(5, 2 + index), hours=23, minutes=59)
            assignment_records.append(
                {
                    "assignment_id": assignment_id,
                    "course_id": config.course.course_id,
                    "topic_id": topic["topic_id"],
                    "assignment_type": assignment_type,
                    "title": f"{assignment_type.title()} W{week_number:02d}",
                    "max_score": 100.0,
                    "due_at": pd.Timestamp(due_at),
                    "weight_percent": weight_percent,
                    "is_required": True,
                }
            )

    return pd.DataFrame(assignment_records)
