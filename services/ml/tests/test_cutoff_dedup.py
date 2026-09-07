"""A student must appear at most once in a cohort.

Parallel sections of one course map to the same cohort key, so a student
enrolled in two of them arrived twice and was trained and evaluated on twice.
It was 12 rows across three KU Leuven cohorts, which is small enough that only
a row count caught it, which is why this test exists.
"""

import pandas as pd

from src.experiments.transfer_benchmark import cutoff_rows


def _weekly(student_ids: list[str], n_weeks: int = 12) -> pd.DataFrame:
    rows = [
        {
            "student_id": sid,
            "week_number": w,
            "n_weeks": n_weeks,
            "passed": 1,
            "cum_clicks": float(w),
            "cum_active_days": float(w),
            "cum_content_clicks": float(w),
            "cum_social": 0.0,
            "cur_clicks": 1.0,
            "active_weeks": float(w),
            "weeks_since_active": 0.0,
        }
        for sid in student_ids
        for w in range(1, n_weeks + 1)
    ]
    return pd.DataFrame(rows)


def test_duplicated_enrolment_yields_one_row():
    canon = _weekly(["a", "b", "b", "c"])
    rows = cutoff_rows(canon, 0.33)
    assert len(rows) == 3
    assert rows["student_id"].is_unique


def test_distinct_students_are_all_kept():
    canon = _weekly(["a", "b", "c"])
    rows = cutoff_rows(canon, 0.33)
    assert sorted(rows["student_id"]) == ["a", "b", "c"]
    # week 4 = round(0.33 * 12)
    assert set(rows["week_number"]) == {4}
