# -*- coding: utf-8 -*-
"""Student-clustered bootstrap confidence intervals for the OULAD BBB 2013J
mastery-vs-LMS comparison (exp_009), computed by faithfully re-running the
existing modeling pipeline.

It is ADDITIVE and NON-DESTRUCTIVE: it reads the frozen exp_009 snapshot and
results, recomputes the gradient-boosting regression cells with prediction
capture, runs a student-clustered bootstrap on the RMSE difference, and writes a
NEW artifact `bootstrap_ci.json` next to the frozen results. It never overwrites
the frozen exp_009 result files.

It also records the reproduced point estimates next to the stored ones and the
runtime environment, because the temporal-forward cell is an out-of-time
extrapolation whose exact magnitude is environment-sensitive.

Run:
    services/ml/.venv/Scripts/python.exe services/ml/scripts/bbb_temporal_bootstrap_ci.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = REPO_ROOT / "services" / "ml"
sys.path.insert(0, str(ML_ROOT))

from src.experiments.run_public_benchmark_oulad import (  # noqa: E402
    load_public_benchmark_config,
    _build_split,
)
from src.experiments.preprocessing import (  # noqa: E402
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.models import build_regression_model  # noqa: E402

EXP_DIR = REPO_ROOT / "data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j"
CONFIG = ML_ROOT / "configs/experiments/exp_009_oulad_ablation_bbb2013j.yaml"
SNAPSHOT = EXP_DIR / "oulad_weekly_snapshots.csv"
RESULTS = EXP_DIR / "exp_009_oulad_ablation_bbb2013j_results.json"
OUT = EXP_DIR / "bootstrap_ci.json"

BASELINE = "B_lms_oulad"
CANDIDATE = "B_lms_plus_mastery_oulad"
MODEL = "gradient_boosting"
B = 5000
SEED = 42


def stored_rmse(split: str, feature_set: str) -> float | None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for r in data["rows"]:
        if (
            r["task"] == "regression"
            and r["split_strategy"] == split
            and r["model"] == MODEL
            and r["feature_set"] == feature_set
        ):
            return float(r["metrics"]["rmse"])
    return None


def fit_cell(modeling_frame, feature_set, config, split):
    matrix = build_modeling_matrix(modeling_frame, feature_set)
    part = _build_split(matrix, split, config)
    train = select_rows(matrix, part.train_mask)
    test = select_rows(matrix, part.test_mask)
    imp = fit_imputer_on_training(train.features)
    x_tr, x_te = imp.transform(train.features), imp.transform(test.features)
    est = build_regression_model(MODEL, seed=config.seed)
    est.fit(x_tr, train.regression_target.to_numpy())
    y_te = test.regression_target.to_numpy()
    y_pred = est.predict(x_te)
    rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    return y_te, y_pred, rmse, test.groups.to_numpy()


def cluster_bootstrap(err_b, err_m, groups):
    uniq = np.unique(groups)
    idx_by_student = {s: np.where(groups == s)[0] for s in uniq}
    rng = np.random.default_rng(SEED)
    rb = np.empty(B); rm = np.empty(B); dd = np.empty(B)
    for i in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_student[s] for s in chosen])
        rb[i] = np.sqrt(np.mean(err_b[idx] ** 2))
        rm[i] = np.sqrt(np.mean(err_m[idx] ** 2))
        dd[i] = rm[i] - rb[i]
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "rmse_baseline_ci95": pct(rb),
        "rmse_candidate_ci95": pct(rm),
        "delta_candidate_minus_baseline_ci95": pct(dd),
        "p_delta_ge_0_one_sided": float(np.mean(dd >= 0)),
        "n_students": int(len(uniq)),
    }


def run_split(modeling_frame, config, split):
    fsets = {f.name: f for f in config.build_feature_sets()}
    yb, pb, rb, gb = fit_cell(modeling_frame, fsets[BASELINE], config, split)
    ym, pm, rm, gm = fit_cell(modeling_frame, fsets[CANDIDATE], config, split)
    assert np.array_equal(yb, ym) and np.array_equal(gb, gm), "test rows not aligned"
    boot = cluster_bootstrap(pb - yb, pm - yb, gb)
    sb, sc = stored_rmse(split, BASELINE), stored_rmse(split, CANDIDATE)
    return {
        "split": split,
        "n_test_rows": int(len(yb)),
        "reproduced": {
            "rmse_baseline": rb,
            "rmse_candidate": rm,
            "delta_candidate_minus_baseline": rm - rb,
        },
        "stored": {
            "rmse_baseline": sb,
            "rmse_candidate": sc,
            "delta_candidate_minus_baseline": (None if sb is None or sc is None else sc - sb),
        },
        "student_clustered_bootstrap": boot,
    }


def main() -> None:
    config, _ = load_public_benchmark_config(str(CONFIG))
    snap = pd.read_csv(SNAPSHOT)
    mf = snap.copy()
    mf["final_grade"] = pd.to_numeric(mf[config.oulad.targets.primary], errors="coerce")
    mf["passed"] = (
        pd.to_numeric(mf[config.oulad.targets.secondary], errors="coerce")
        .fillna(0).astype(int).astype(bool)
    )
    mf = mf.loc[mf["final_grade"].notna()].copy()

    payload = {
        "artifact_type": "student_clustered_bootstrap_ci",
        "source_experiment": "exp_009_oulad_ablation_bbb2013j",
        "cohort": "OULAD BBB 2013J",
        "model": MODEL,
        "baseline_feature_set": BASELINE,
        "candidate_feature_set": CANDIDATE,
        "bootstrap_resamples": B,
        "bootstrap_unit": "held-out test student (cluster)",
        "seed": SEED,
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "note": (
            "NON-DESTRUCTIVE: recomputed via the frozen pipeline; does not overwrite "
            "exp_009 result files. The temporal_forward cell is an out-of-time "
            "extrapolation whose exact RMSE magnitude is environment-sensitive, so "
            "reproduced point estimates may differ from the stored ones; the bootstrap "
            "CIs apply to the reproduced (this-environment) estimates."
        ),
        "splits": [
            run_split(mf, config, "temporal_forward"),
            run_split(mf, config, "student_group"),
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env = payload["environment"]
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"env: python {env['python']}, sklearn {env['sklearn']}, {env['platform']}")
    for s in payload["splits"]:
        rep, st, bo = s["reproduced"], s["stored"], s["student_clustered_bootstrap"]
        ci = bo["delta_candidate_minus_baseline_ci95"]
        print(f"\n[{s['split']}]  n_test={s['n_test_rows']}, n_students={bo['n_students']}")
        print(f"  reproduced: B_lms={rep['rmse_baseline']:.3f}  mastery={rep['rmse_candidate']:.3f}  delta={rep['delta_candidate_minus_baseline']:+.3f}")
        print(f"  stored    : B_lms={st['rmse_baseline']:.3f}  mastery={st['rmse_candidate']:.3f}  delta={st['delta_candidate_minus_baseline']:+.3f}")
        print(f"  delta 95% CI (clustered) = [{ci[0]:+.3f}, {ci[1]:+.3f}]  P(delta>=0)={bo['p_delta_ge_0_one_sided']:.4f}")


if __name__ == "__main__":
    main()
