import numpy as np
import pandas as pd

from src.experiments import transfer_benchmark as tb


def _weekly():
    # two students, 4-week course; s1 active weeks 1,2,4; s2 active only week 1
    return pd.DataFrame(
        {
            "student_id": ["s1"] * 4 + ["s2"] * 4,
            "week_number": [1, 2, 3, 4] * 2,
            "n_weeks": [4] * 8,
            "passed": [1] * 4 + [0] * 4,
            "current_clicks": [5, 3, 0, 2, 4, 0, 0, 0],
            "current_active_days": [2, 1, 0, 1, 1, 0, 0, 0],
            "current_content_clicks": [3, 1, 0, 1, 1, 0, 0, 0],
            "current_social_clicks": [0, 1, 0, 0, 0, 0, 0, 0],
            "institution": ["A"] * 8,
            "module": ["m"] * 8,
            "cohort_id": ["c"] * 8,
        }
    )


def test_canonical_features_are_leakage_safe_cumsums_and_recency():
    c = tb.canonical_from_weekly(_weekly())
    s1 = c[c.student_id == "s1"].set_index("week_number")
    s2 = c[c.student_id == "s2"].set_index("week_number")
    assert s1.loc[4, "cum_clicks"] == 10 and s1.loc[2, "cum_clicks"] == 8
    assert list(s1["active_weeks"]) == [1, 2, 2, 3]
    assert list(s1["weeks_since_active"]) == [0, 0, 1, 0]
    assert list(s2["weeks_since_active"]) == [0, 1, 2, 3]


def test_cutoff_and_percentile_representation():
    c = tb.canonical_from_weekly(_weekly())
    rows = tb.cutoff_rows(c, 0.5)  # week 2 of 4
    assert list(rows["week_number"]) == [2, 2]
    x = rows[list(tb.CANON)].astype(float)
    pct = tb.represent(x, "percentile")
    assert pct.shape == (2, len(tb.CANON))
    assert np.all((pct >= 0) & (pct <= 1))
    # constant column -> identical percentile for both rows, z-score -> zeros not NaN
    z = tb.represent(x, "zscore")
    assert not np.isnan(z).any()
