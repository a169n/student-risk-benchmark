from __future__ import annotations

import numpy as np

from src.generator.students import StudentProfile
from src.generator.utils import clamp


def generate_weekly_activity_record(
    profile: StudentProfile,
    *,
    topic_id: str,
    week_number: int,
    topic_difficulty: float,
    effective_state: dict[str, float | bool],
    attendance_rate: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    baseline = float(effective_state["baseline"])
    motivation = float(effective_state["motivation"])
    discipline = float(effective_state["discipline"])

    if bool(effective_state["is_withdrawn_phase"]):
        login_count = int(rng.integers(0, 2))
        active_days = int(rng.integers(0, 2))
        content_views = int(rng.integers(0, 4))
        practice_events = int(rng.integers(0, 3))
        forum_posts = int(rng.integers(0, 2))
        time_on_platform_minutes = float(rng.uniform(0, 25))
    else:
        base_engagement = clamp(
            0.34 * motivation + 0.20 * discipline + 0.18 * baseline + 0.28 * attendance_rate,
            0.02,
            0.99,
        )
        login_count = max(0, int(round(1 + base_engagement * 8 + rng.normal(0.0, 1.0))))
        active_days = int(clamp(round(base_engagement * 6 + rng.normal(0.0, 0.8)), 0, 7))
        content_views = max(
            0,
            int(round(4 + base_engagement * 18 + topic_difficulty * 4 + rng.normal(0.0, 2.2))),
        )
        practice_events = max(
            0,
            int(round(1 + (0.55 * baseline + 0.25 * motivation + 0.20 * discipline) * 12 + rng.normal(0.0, 1.8))),
        )
        forum_posts = max(0, int(round(base_engagement * 2 + rng.normal(0.0, 0.7))))
        time_on_platform_minutes = float(
            clamp(20 + base_engagement * 220 + rng.normal(0.0, 18.0), 5.0, 320.0)
        )

    login_norm = clamp(login_count / 12.0, 0.0, 1.0)
    active_norm = clamp(active_days / 7.0, 0.0, 1.0)
    content_norm = clamp(content_views / 30.0, 0.0, 1.0)
    practice_norm = clamp(practice_events / 15.0, 0.0, 1.0)
    time_norm = clamp(time_on_platform_minutes / 300.0, 0.0, 1.0)
    activity_score = 100 * (
        0.20 * login_norm
        + 0.20 * active_norm
        + 0.20 * content_norm
        + 0.20 * practice_norm
        + 0.20 * time_norm
    )

    return {
        "weekly_activity_id": f"act_{profile.student_id}_w{week_number:02d}",
        "student_id": profile.student_id,
        "course_id": profile.course_id,
        "topic_id": topic_id,
        "week_number": week_number,
        "login_count": login_count,
        "active_days": active_days,
        "content_views": content_views,
        "practice_events": practice_events,
        "forum_posts": forum_posts,
        "time_on_platform_minutes": round(time_on_platform_minutes, 2),
        "activity_score": round(activity_score, 2),
    }
