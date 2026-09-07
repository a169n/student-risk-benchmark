# exp_009_oulad_ablation_bbb2013j: Second-cohort full A/B/C ablation on OULAD (BBB 2013J)

## Objective

Replication test of the exp_006 full nested A/B/C ablation on a second, different OULAD module-presentation (BBB 2013J) to determine whether the mixed-to-null finding from DDD 2013J — no twin feature block beats the LMS baseline by >1.0 RMSE on either split under a fixed model family — is course-specific or replicates across courses. BBB 2013J is a richer presentation with 2237 students and 11 dated+scored positive-weight assessments plus an exam, providing a stronger stress-test.


## Hypothesis

If the mixed-to-null result from exp_006 is robust and not an artefact of DDD 2013J course structure, the same pattern should replicate on BBB 2013J: no twin feature block (trends, mastery, indices, C_twin) should beat B_lms_oulad by >1.0 RMSE on either split under the fixed gradient_boosting model. If the result is course-specific, at least one twin block should show a clear improvement on BBB 2013J. A replication of the null is the expected and scientifically informative outcome.


## Dataset / config used

- Public benchmark dataset: Open University Learning Analytics Dataset (OULAD)
- Experiment config: `services/ml/configs/experiments/exp_009_oulad_ablation_bbb2013j.yaml`
- Raw directory: `datasets/oulad`
- Processed benchmark snapshots: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/oulad_weekly_snapshots.csv`
- Output directory: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j`
- Course filter: `{'code_module': 'BBB', 'code_presentation': '2013J'}`
- Course-filter rationale: BBB 2013J is a second, different OULAD module-presentation used to test whether the DDD 2013J mixed-to-null result from exp_006 is course-specific or replicates; it has a larger cohort (2237 students) and richer assessment structure (11 dated+scored positive-weight assessments plus exam) than DDD.

- Snapshot weeks used: `4..39`

## Exact OULAD files used

- `assessments.csv` from `datasets/oulad/assessments.csv`
- `courses.csv` from `datasets/oulad/courses.csv`
- `studentInfo.csv` from `datasets/oulad/studentInfo.csv`
- `studentRegistration.csv` from `datasets/oulad/studentRegistration.csv`
- `studentVle.csv` from `datasets/oulad/studentVle.csv`
- `vle.csv` from `datasets/oulad/vle.csv`
- `studentAssessment.csv` from `datasets/oulad/studentAssessment.csv`

## Targets

- Primary regression target: `final_weighted_score`
- Secondary classification target: `passed_observed`
- Excluded supervised target(s): `['risk_level']`

`final_weighted_score` is derived from OULAD `assessments.weight` and `studentAssessment.score` as sum(score_or_zero * assessment_weight) / sum(assessment_weight) over positive-weight assessments in the selected module-presentation that have score records in `studentAssessment`. Missing student submissions for those assessments contribute zero to the numerator. `passed_observed` is derived from `studentInfo.final_result`, with Pass and Distinction mapped to 1 and Fail and Withdrawn mapped to 0.


Target distribution is computed at the student-course-presentation level.

| target summary | value |
| --- | ---: |
| final_weighted_score count | 2237 |
| final_weighted_score mean | 41.189 |
| final_weighted_score std | 32.829 |
| final_weighted_score min | 0.000 |
| final_weighted_score median | 49.780 |
| final_weighted_score max | 96.460 |
| passed_observed counts | {'0.0': 1165, '1.0': 1072} |

## Feature sets

### `A_simple_oulad`

Minimal OULAD academic baseline: cumulative assessment score signals only.

- Columns: `week_number`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`
- Indicator columns: `has_assessment_score_to_date`

### `B_lms_oulad`

Strong OULAD LMS-style baseline analogue using cumulative assessment performance, submission discipline, VLE activity intensity and category signals, course-week progression, and registration state.


- Columns: `week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`
- Indicator columns: `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `B_lms_plus_trends_oulad`

OULAD LMS baseline plus short-horizon trend analogues (assessment-score and clicks week-over-week).

- Columns: `week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `assessment_score_trend_to_date`, `clicks_trend_to_date`
- Indicator columns: `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `B_lms_plus_mastery_oulad`

OULAD LMS baseline analogue plus a lean mastery-like block derived from dated assessment structure: due-to-date weighted mastery, current assessment-cluster mastery, assessment-type mastery aggregates, and assessment-coverage context.


- Columns: `week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `overall_mastery_proxy`, `current_assessment_cluster_mastery`, `tma_mastery_to_date`, `cma_mastery_to_date`, `exam_mastery_to_date`, `mastery_assessment_coverage_to_date`
- Indicator columns: `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`, `has_current_assessment_cluster`, `has_due_assessment_to_date`

### `B_lms_plus_indices_oulad`

OULAD LMS baseline plus composite engagement/performance/discipline indices.

- Columns: `week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `engagement_index_oulad`, `performance_index_oulad`, `discipline_index_oulad`
- Indicator columns: `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `C_twin_oulad`

Full OULAD twin analogue: LMS baseline + trends + mastery block + composite indices.

- Columns: `week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `assessment_score_trend_to_date`, `clicks_trend_to_date`, `overall_mastery_proxy`, `current_assessment_cluster_mastery`, `tma_mastery_to_date`, `cma_mastery_to_date`, `exam_mastery_to_date`, `mastery_assessment_coverage_to_date`, `engagement_index_oulad`, `performance_index_oulad`, `discipline_index_oulad`
- Indicator columns: `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`, `has_current_assessment_cluster`, `has_due_assessment_to_date`

## Split strategies and models

- Primary split: `student_group`
- Secondary split: `temporal_forward`
- Student-group split: `test_size=0.25`, `seed=42`
- Temporal-forward split: `train_weeks=20`, `student_test_size=0.25`, `student_seed=42`
- Regression models: `linear_regression`, `random_forest`, `gradient_boosting`
- Classification models: `logistic_regression`, `random_forest`, `gradient_boosting`

## Row counts

| item | count |
| --- | ---: |
| assessments | 12 |
| courses | 1 |
| student_info | 2237 |
| student_registration | 2237 |
| vle | 321 |
| student_assessment | 14375 |
| student_vle | 452638 |
| snapshots | 80532 |
| students | 2237 |

## Headline regression results

| split | feature set | best model | RMSE | MAE | R^2 | delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| temporal_forward | B_lms_plus_mastery_oulad | gradient_boosting | 5.284 | 3.426 | 0.974 | -1.026 |
| temporal_forward | C_twin_oulad | gradient_boosting | 5.546 | 3.619 | 0.971 | -0.765 |
| temporal_forward | B_lms_oulad | gradient_boosting | 6.311 | 4.124 | 0.963 | +0.000 |
| temporal_forward | B_lms_plus_trends_oulad | gradient_boosting | 6.546 | 4.321 | 0.960 | +0.235 |
| temporal_forward | B_lms_plus_indices_oulad | gradient_boosting | 6.554 | 4.215 | 0.960 | +0.243 |
| temporal_forward | A_simple_oulad | gradient_boosting | 10.230 | 6.656 | 0.903 | +3.919 |
| student_group | C_twin_oulad | gradient_boosting | 10.610 | 5.605 | 0.895 | -0.087 |
| student_group | B_lms_plus_mastery_oulad | gradient_boosting | 10.620 | 5.623 | 0.895 | -0.077 |
| student_group | B_lms_plus_indices_oulad | gradient_boosting | 10.650 | 5.741 | 0.895 | -0.047 |
| student_group | B_lms_plus_trends_oulad | gradient_boosting | 10.695 | 5.758 | 0.894 | -0.002 |
| student_group | B_lms_oulad | gradient_boosting | 10.697 | 5.770 | 0.894 | +0.000 |
| student_group | A_simple_oulad | gradient_boosting | 12.387 | 7.647 | 0.857 | +1.690 |

## Fixed-model regression results (model = `gradient_boosting`)

Same model across all feature sets and both splits, to neutralize best-model-per-cell (model-flip) artifacts. Headline split: `temporal_forward`.

| split | feature set | model | RMSE | MAE | R^2 | delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| temporal_forward | B_lms_plus_mastery_oulad | gradient_boosting | 5.284 | 3.426 | 0.974 | -1.026 |
| temporal_forward | C_twin_oulad | gradient_boosting | 5.546 | 3.619 | 0.971 | -0.765 |
| temporal_forward | B_lms_oulad | gradient_boosting | 6.311 | 4.124 | 0.963 | +0.000 |
| temporal_forward | B_lms_plus_trends_oulad | gradient_boosting | 6.546 | 4.321 | 0.960 | +0.235 |
| temporal_forward | B_lms_plus_indices_oulad | gradient_boosting | 6.554 | 4.215 | 0.960 | +0.243 |
| temporal_forward | A_simple_oulad | gradient_boosting | 10.230 | 6.656 | 0.903 | +3.919 |
| student_group | C_twin_oulad | gradient_boosting | 10.610 | 5.605 | 0.895 | -0.087 |
| student_group | B_lms_plus_mastery_oulad | gradient_boosting | 10.620 | 5.623 | 0.895 | -0.077 |
| student_group | B_lms_plus_indices_oulad | gradient_boosting | 10.650 | 5.741 | 0.895 | -0.047 |
| student_group | B_lms_plus_trends_oulad | gradient_boosting | 10.695 | 5.758 | 0.894 | -0.002 |
| student_group | B_lms_oulad | gradient_boosting | 10.697 | 5.770 | 0.894 | +0.000 |
| student_group | A_simple_oulad | gradient_boosting | 12.387 | 7.647 | 0.857 | +1.690 |

## Secondary classification results

| split | feature set | best model | F1 | accuracy | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| student_group | A_simple_oulad | gradient_boosting | 0.866 | 0.862 | 0.931 |
| student_group | B_lms_oulad | gradient_boosting | 0.881 | 0.879 | 0.942 |
| student_group | B_lms_plus_indices_oulad | gradient_boosting | 0.880 | 0.878 | 0.943 |
| student_group | B_lms_plus_mastery_oulad | logistic_regression | 0.881 | 0.880 | 0.945 |
| student_group | B_lms_plus_trends_oulad | gradient_boosting | 0.881 | 0.880 | 0.943 |
| student_group | C_twin_oulad | gradient_boosting | 0.881 | 0.880 | 0.943 |
| temporal_forward | A_simple_oulad | logistic_regression | 0.876 | 0.883 | 0.955 |
| temporal_forward | B_lms_oulad | logistic_regression | 0.928 | 0.928 | 0.968 |
| temporal_forward | B_lms_plus_indices_oulad | logistic_regression | 0.928 | 0.927 | 0.968 |
| temporal_forward | B_lms_plus_mastery_oulad | logistic_regression | 0.929 | 0.928 | 0.969 |
| temporal_forward | B_lms_plus_trends_oulad | logistic_regression | 0.929 | 0.928 | 0.968 |
| temporal_forward | C_twin_oulad | logistic_regression | 0.928 | 0.928 | 0.970 |

## Interpretation

The OULAD benchmark is directionally consistent with the synthetic carry-forward decision: the lean mastery analogue improves over the OULAD LMS baseline on the primary grouped split. This supports the representation-transfer logic with caveats, not full external validity.

- Outcome: `supports_with_caveats`
- Primary delta RMSE: `-0.765`
- Short conclusion: `C_twin_oulad` improved over `B_lms_oulad` on OULAD.

This is a representation-transfer stress test, not a claim that one dataset is better than another and not a claim of full external validity.

## Artifact paths

- Results CSV: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/exp_009_oulad_ablation_bbb2013j_results.csv`
- Results JSON: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/exp_009_oulad_ablation_bbb2013j_results.json`
- Summary Markdown: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/exp_009_oulad_ablation_bbb2013j_summary.md`
- Processed snapshots CSV: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/oulad_weekly_snapshots.csv`
- Mapping artifact: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/public_benchmark_mapping_summary.md`
- Public-vs-synthetic interpretation: `data/artifacts/experiments/exp_009_oulad_ablation_bbb2013j/public_vs_synthetic_interpretation.md`

## Limitations

- This experiment isolates the marginal contribution of each twin feature block on a public dataset; it is not a comparison of synthetic versus public dataset quality.

- The primary OULAD target is a derived weighted assessment score, not the exact same construct as the synthetic `final_grade`.

- The selected OULAD subset is one module-presentation (`BBB`, `2013J`) to match the current one-course research scope and keep model training reproducible on local hardware.

- OULAD has no clean equivalents for attendance, synthetic weekly topics, hidden trajectory parameters, risk labels, or intervention state.

- This benchmark can support or complicate external transfer plausibility; it cannot establish full institutional external validity.

- The OULAD regression target (final_weighted_score) is partially circular on assessment-score features (cumulative_assessment_weighted_score_to_date feeds the target); interpretation leans on exogenous clickstream features and the classification target.


## Next step

Compare the BBB 2013J fixed-model regression table against the DDD 2013J results from exp_006. If the mixed-to-null replicates, strengthen the dissertation claim that twin blocks do not transfer to OULAD across courses and report a cross-course null. If the result diverges, investigate which structural differences between BBB and DDD (assessment density, VLE activity patterns, cohort size) might explain the divergence.

