"""exp_014 — transfer ladder for early-warning models across cohorts and institutions.

Question: does an engagement-only early-warning model trained on one cohort
still work on another cohort, another course, another institution — and does a
cohort-relative feature representation (unsupervised percentile / z-score
within the target cohort) make it portable?

Everything here works on ONE canonical frame contract shared by the three
institution loaders (OULAD, KU Leuven, UKZN):

    institution, cohort_id, module, student_id, week_number, n_weeks, passed,
    current_clicks, current_active_days, current_content_clicks, current_social_clicks

From the four per-week counters the module derives the seven canonical
engagement features every institution can supply (``CANON``), takes one row per
student at a relative cutoff (week = round(f * n_weeks)), applies a
representation, and evaluates every ordered (source -> target) cohort pair with
a fixed model. Models are fitted ONCE per (source, representation, model) and
reused for all targets, so the full ladder is cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.experiments.models import build_classification_model
from src.experiments.stability import jaccard_topk, kendall_tau

CANON: tuple[str, ...] = (
    "cum_clicks",
    "cum_active_days",
    "cum_content_clicks",
    "cum_social",
    "cur_clicks",
    "active_weeks",
    "weeks_since_active",
)
REPRESENTATIONS: tuple[str, ...] = ("raw", "zscore", "percentile")
MODELS: tuple[str, ...] = ("gradient_boosting", "logistic_regression", "random_forest")
# The explanation-portability claim is about the deployed model, and permutation
# importance is the O(cohorts^2) bottleneck, so rankings are computed for the
# primary model only. The other families are AUC robustness checks; their
# explanation columns come back as NaN.
EXPLAIN_MODEL: str = "gradient_boosting"
DISTANCE_ORDER: tuple[str, ...] = (
    "D0_within_cohort",
    "D1_same_module",
    "D2_other_module",
    "D3_other_institution",
)


# ---------------------------------------------------------------------------
# Canonical frame construction
# ---------------------------------------------------------------------------


def canonical_from_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the seven CANON features from the four per-week counters (leakage-safe cumsums)."""
    f = frame.sort_values(["student_id", "week_number"]).reset_index(drop=True)
    g = f.groupby("student_id", sort=False)
    f["cum_clicks"] = g["current_clicks"].cumsum()
    f["cum_active_days"] = g["current_active_days"].cumsum()
    f["cum_content_clicks"] = g["current_content_clicks"].cumsum()
    f["cum_social"] = g["current_social_clicks"].cumsum()
    f["cur_clicks"] = f["current_clicks"]
    active = (f["current_clicks"] > 0).astype(int)
    f["active_weeks"] = active.groupby(f["student_id"], sort=False).cumsum()
    last_active = f["week_number"].where(active == 1)
    last_active = last_active.groupby(f["student_id"], sort=False).ffill()
    f["weeks_since_active"] = (f["week_number"] - last_active.fillna(0)).astype(float)
    return f


def cutoff_rows(canon: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """One row per student at week = max(1, round(fraction * n_weeks)) of its own course.

    The dedup is not cosmetic. A student enrolled in two parallel sections of the
    same course arrives twice, and both copies then land in training and in
    evaluation of the same cohort. It is 12 rows of 45,170 here, all in KU Leuven
    Accountancy, but the guard belongs on the shared path rather than in one
    adapter, because any source with sections can do this.
    """
    target_week = np.maximum(1, np.rint(fraction * canon["n_weeks"]).astype(int))
    rows = canon.loc[canon["week_number"] == target_week]
    return rows.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)


def represent(x: pd.DataFrame, representation: str) -> np.ndarray:
    """Unsupervised, within-cohort feature representation. Uses no labels."""
    if representation == "raw":
        return x.to_numpy(dtype=float)
    if representation == "zscore":
        sd = x.std(ddof=0).replace(0, 1.0)
        return ((x - x.mean()) / sd).to_numpy(dtype=float)
    if representation == "percentile":
        return x.rank(pct=True, method="average").to_numpy(dtype=float)
    raise ValueError(representation)


# ---------------------------------------------------------------------------
# Cohort registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cohort:
    cohort_id: str
    institution: str
    module: str
    X_raw: pd.DataFrame  # one row per student, CANON columns
    y: np.ndarray

    def X(self, representation: str) -> np.ndarray:
        return represent(self.X_raw, representation)

    @property
    def n(self) -> int:
        return int(len(self.y))


def make_cohorts(
    canon_frames: dict[str, pd.DataFrame], fraction: float, min_students: int = 50, min_minority: int = 15
) -> list[Cohort]:
    """Keep cohorts with enough students AND enough examples of the rarer class (AUC is unstable below ~15)."""
    cohorts: list[Cohort] = []
    for cid, frame in canon_frames.items():
        rows = cutoff_rows(frame, fraction)
        y = rows["passed"].to_numpy(dtype=int)
        if len(y) < min_students or min(int(y.sum()), int(len(y) - y.sum())) < min_minority:
            continue
        cohorts.append(
            Cohort(
                cohort_id=cid,
                institution=str(rows["institution"].iloc[0]),
                module=str(rows["module"].iloc[0]),
                X_raw=rows[list(CANON)].astype(float),
                y=y,
            )
        )
    return cohorts


def distance_class(source: Cohort, target: Cohort) -> str:
    if source.cohort_id == target.cohort_id:
        return "D0_within_cohort"
    if source.institution != target.institution:
        return "D3_other_institution"
    if source.module == target.module:
        return "D1_same_module"
    return "D2_other_module"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


FLAG_RATE = 0.20


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Discrimination, calibration and decision quality, kept separate.

    Reporting these together is deliberate. In the external-validation
    literature a transported model routinely keeps its discrimination while its
    calibration fails, and one number cannot show that.

    - ``auc`` — discrimination, threshold-free.
    - ``f1_fail`` — F1 on the class an early-warning system actually alerts on
      (``passed == 0``). ``f1_pass`` is retained only because a naive
      implementation reports it by default; it is the majority class in most
      cohorts and is not a deployment metric.
    - ``recall_at_flag20`` / ``precision_at_flag20`` — the realistic decision
      rule: flag the ``FLAG_RATE`` lowest-scoring students, whatever the
      cohort's prevalence, and ask how many true failures that catches. This
      removes the fixed-threshold artefact without hiding the calibration
      problem, because the flag budget is what an institution actually sets.
    - ``brier`` and ``calibration_in_large`` (mean predicted minus observed
      failure rate) — calibration proper.
    - ``f1_fail_baseline`` — F1 of always predicting failure, so every F1 is
      readable against the trivial rule that beats many published models.
    """
    y = np.asarray(y)
    fail = 1 - y  # the alerting class
    risk = 1.0 - np.asarray(p)  # predicted probability of failing
    predicted_fail = (risk > 0.5).astype(int)

    n_flag = max(1, int(round(FLAG_RATE * len(y))))
    flagged = np.zeros(len(y), dtype=int)
    flagged[np.argsort(-risk)[:n_flag]] = 1
    caught = int((flagged & fail).sum())
    n_fail = int(fail.sum())
    prevalence = float(fail.mean())

    return {
        "auc": float(roc_auc_score(y, p)),
        "f1_pass": float(f1_score(y, (np.asarray(p) >= 0.5).astype(int), zero_division=0)),
        "f1_fail": float(f1_score(fail, predicted_fail, zero_division=0)),
        "f1_fail_baseline": float(2 * prevalence / (1 + prevalence)) if prevalence > 0 else 0.0,
        "recall_at_flag20": float(caught / n_fail) if n_fail else float("nan"),
        "precision_at_flag20": float(caught / n_flag),
        "lift_at_flag20": float((caught / n_flag) / prevalence) if prevalence > 0 else float("nan"),
        "brier": float(np.mean((risk - fail) ** 2)),
        "calibration_in_large": float(risk.mean() - prevalence),
    }


# Permutation importance is computed once per (source, target) pair, so it
# dominates runtime on a 45-cohort ladder. Both knobs below only affect the
# EXPLANATION ranking; every reported AUC / F1 still uses the full target set.
IMPORTANCE_REPEATS = 3
IMPORTANCE_MAX_ROWS = 1500


def _ranking(model, X: np.ndarray, y: np.ndarray, seed: int) -> list[str]:
    """Best-first feature ranking by permutation importance (AUC scoring).

    Large targets are subsampled (stratified by label) to keep the O(cohorts^2)
    ranking pass tractable; the ranking is a rank statistic over 7 features, so
    it is stable well below the full cohort size.
    """
    if len(y) > IMPORTANCE_MAX_ROWS:
        rng = np.random.default_rng(seed)
        idx = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(y == cls),
                    size=max(1, round(IMPORTANCE_MAX_ROWS * (y == cls).mean())),
                    replace=False,
                )
                for cls in (0, 1)
            ]
        )
        X, y = X[idx], y[idx]
    imp = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=IMPORTANCE_REPEATS, random_state=seed
    )
    order = np.argsort(-imp.importances_mean)
    return [CANON[i] for i in order]


def run_ladder(
    cohorts: list[Cohort], *, seed: int = 42, models: tuple[str, ...] = MODELS, verbose: bool = True
) -> pd.DataFrame:
    """Every ordered (source -> target) pair x representation x model. Fit once per source.

    An O(cohorts^2) pass over 60+ cohorts takes tens of minutes, so each
    (representation, model) block reports when it finishes.
    """
    import time

    rows: list[dict] = []
    target_rankings: dict[tuple[str, str, str], list[str]] = {}
    blocks = len(REPRESENTATIONS) * len(models)
    done = 0
    for rep in REPRESENTATIONS:
        for mname in models:
            started = time.time()
            explain = mname == EXPLAIN_MODEL
            fitted = {}
            for c in cohorts:
                m = build_classification_model(mname, seed=seed).fit(c.X(rep), c.y)
                fitted[c.cohort_id] = m
                if explain:
                    target_rankings[(c.cohort_id, rep, mname)] = _ranking(m, c.X(rep), c.y, seed)
            for s in cohorts:
                for t in cohorts:
                    d = distance_class(s, t)
                    if d == "D0_within_cohort":
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                        p = cross_val_predict(
                            build_classification_model(mname, seed=seed), t.X(rep), t.y, cv=cv, method="predict_proba"
                        )[:, 1]
                        tau = 1.0 if explain else float("nan")
                        jac = 1.0 if explain else float("nan")
                    else:
                        p = fitted[s.cohort_id].predict_proba(t.X(rep))[:, 1]
                        if explain:
                            r_src = _ranking(fitted[s.cohort_id], t.X(rep), t.y, seed)
                            r_tgt = target_rankings[(t.cohort_id, rep, mname)]
                            tau = kendall_tau(r_src, r_tgt)
                            jac = jaccard_topk(r_src, r_tgt, 3)
                        else:
                            tau = float("nan")
                            jac = float("nan")
                    rows.append(
                        {
                            "source": s.cohort_id,
                            "target": t.cohort_id,
                            "source_institution": s.institution,
                            "target_institution": t.institution,
                            "distance": d,
                            "representation": rep,
                            "model": mname,
                            "n_target": t.n,
                            **_scores(t.y, p),
                            "explanation_tau": tau,
                            "explanation_jaccard_top3": jac,
                        }
                    )
            done += 1
            if verbose:
                print(
                    f"  ladder block {done}/{blocks}: {mname} / {rep} "
                    f"({len(rows)} rows, {time.time() - started:.0f}s)",
                    flush=True,
                )
    return pd.DataFrame(rows)


def run_pooled(cohorts: list[Cohort], *, seed: int = 42, models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """Two pooled sources per target: leave-one-cohort-out within its institution, and all other institutions."""
    rows: list[dict] = []
    for rep in REPRESENTATIONS:
        for mname in models:
            for t in cohorts:
                pools = {
                    "P_in_leave_one_cohort_out": [c for c in cohorts if c.institution == t.institution and c is not t],
                    "P_out_other_institutions": [c for c in cohorts if c.institution != t.institution],
                }
                for pname, members in pools.items():
                    if not members:
                        continue
                    X = np.vstack([c.X(rep) for c in members])
                    y = np.concatenate([c.y for c in members])
                    m = build_classification_model(mname, seed=seed).fit(X, y)
                    p = m.predict_proba(t.X(rep))[:, 1]
                    rows.append(
                        {
                            "target": t.cohort_id,
                            "target_institution": t.institution,
                            "pool": pname,
                            "n_source_cohorts": len(members),
                            "representation": rep,
                            "model": mname,
                            "n_target": t.n,
                            **_scores(t.y, p),
                        }
                    )
    return pd.DataFrame(rows)


FEWSHOT_SOURCE_CAP = 8_000


def _cap_rows(X: np.ndarray, y: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Label-stratified subsample of a pooled training set, for runtime only."""
    if cap <= 0 or len(y) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == cls), size=max(1, round(cap * (y == cls).mean())), replace=False)
            for cls in (0, 1)
        ]
    )
    return X[idx], y[idx]


def run_fewshot(
    cohorts: list[Cohort],
    *,
    ks: tuple[int, ...] = (0, 25, 50, 100),
    representation: str = "percentile",
    model: str = "gradient_boosting",
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    source_cap: int = FEWSHOT_SOURCE_CAP,
) -> pd.DataFrame:
    """Other-institution pooled source + k labelled target students; evaluate on the rest of the target.

    The pooled source runs to ~40,000 rows across four institutions, and this
    grid refits it 64 targets x 5 seeds x 4 values of k times. The source is
    therefore label-stratified subsampled to ``source_cap`` rows per seed, which
    is far above the sample size at which this curve is estimable and keeps the
    stage from dominating the run. The curve compares k against k = 0 under the
    identical source, so the cap cannot manufacture the effect.
    """
    rows: list[dict] = []
    for t in cohorts:
        others = [c for c in cohorts if c.institution != t.institution]
        Xs_full = np.vstack([c.X(representation) for c in others])
        ys_full = np.concatenate([c.y for c in others])
        Xt = t.X(representation)
        for seed in seeds:
            Xs, ys = _cap_rows(Xs_full, ys_full, source_cap, seed)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(t.n)
            for k in ks:
                if k >= t.n - 30:
                    continue
                idx_k, idx_eval = perm[:k], perm[k:]
                if len(set(t.y[idx_eval])) < 2:
                    continue
                X = np.vstack([Xs, Xt[idx_k]]) if k else Xs
                y = np.concatenate([ys, t.y[idx_k]]) if k else ys
                m = build_classification_model(model, seed=seed).fit(X, y)
                p = m.predict_proba(Xt[idx_eval])[:, 1]
                rows.append({"target": t.cohort_id, "target_institution": t.institution, "k": k, "seed": seed, **_scores(t.y[idx_eval], p)})
    return pd.DataFrame(rows)


def aggregate(pairs: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Mean AUC / tau per (distance, representation, model) with a bootstrap-over-pairs 95% CI."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for (d, rep, m), g in pairs.groupby(["distance", "representation", "model"]):
        auc = g["auc"].to_numpy()
        boots = np.array([rng.choice(auc, size=len(auc), replace=True).mean() for _ in range(n_boot)])
        tau = pd.to_numeric(g["explanation_tau"], errors="coerce").dropna()
        out.append(
            {
                "distance": d,
                "representation": rep,
                "model": m,
                "n_pairs": int(len(g)),
                "auc_mean": float(auc.mean()),
                "auc_ci_low": float(np.percentile(boots, 2.5)),
                "auc_ci_high": float(np.percentile(boots, 97.5)),
                "f1_fail_mean": float(g["f1_fail"].mean()),
                "recall_flag20_mean": float(g["recall_at_flag20"].mean()),
                "brier_mean": float(g["brier"].mean()),
                "tau_mean": float(tau.mean()) if len(tau) else float("nan"),
                "jaccard_top3_mean": float(g["explanation_jaccard_top3"].mean()),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "distance", "representation"]).reset_index(drop=True)


def write_outputs(out_dir: Path, **frames: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def cohort_summary(cohorts: list[Cohort]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort_id": c.cohort_id,
            "institution": c.institution,
            "module": c.module,
            "n_students": c.n,
            "n_failed": int(c.n - c.y.sum()),
            "pass_rate": float(c.y.mean()),
            "mean_cum_clicks": float(c.X_raw["cum_clicks"].mean()),
        }
        for c in cohorts
    )


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
