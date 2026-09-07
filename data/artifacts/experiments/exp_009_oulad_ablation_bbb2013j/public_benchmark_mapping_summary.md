# Public Benchmark Mapping Summary

This artifact summarizes the OULAD-to-benchmark mapping used by `exp_009_oulad_ablation_bbb2013j`.

## Scope

- Course filter: `{'code_module': 'BBB', 'code_presentation': '2013J'}`
- Snapshot grain: `1 row = 1 student-course presentation x 1 week`
- Weeks used after filtering: `4..39`

## Mapping

- LMS baseline analogue: cumulative assessment scores, submission timing, VLE click intensity, VLE activity categories, course-week progression, and registration status signals.
- Lean mastery analogue: due-to-date mastery proxies that treat scheduled positive-weight assessments as the local content structure available in OULAD.
- No OULAD field is treated as the synthetic `risk_level`; risk is not a supervised target in this benchmark.

## Feature Sets

### `A_simple_oulad`

Minimal OULAD academic baseline: cumulative assessment score signals only.

`week_number`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `has_assessment_score_to_date`

### `B_lms_oulad`

Strong OULAD LMS-style baseline analogue using cumulative assessment performance, submission discipline, VLE activity intensity and category signals, course-week progression, and registration state.


`week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `B_lms_plus_trends_oulad`

OULAD LMS baseline plus short-horizon trend analogues (assessment-score and clicks week-over-week).

`week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `assessment_score_trend_to_date`, `clicks_trend_to_date`, `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `B_lms_plus_mastery_oulad`

OULAD LMS baseline analogue plus a lean mastery-like block derived from dated assessment structure: due-to-date weighted mastery, current assessment-cluster mastery, assessment-type mastery aggregates, and assessment-coverage context.


`week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `overall_mastery_proxy`, `current_assessment_cluster_mastery`, `tma_mastery_to_date`, `cma_mastery_to_date`, `exam_mastery_to_date`, `mastery_assessment_coverage_to_date`, `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`, `has_current_assessment_cluster`, `has_due_assessment_to_date`

### `B_lms_plus_indices_oulad`

OULAD LMS baseline plus composite engagement/performance/discipline indices.

`week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `engagement_index_oulad`, `performance_index_oulad`, `discipline_index_oulad`, `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`

### `C_twin_oulad`

Full OULAD twin analogue: LMS baseline + trends + mastery block + composite indices.

`week_number`, `course_week_progress`, `is_registered_by_week`, `is_unregistered_by_week`, `days_since_registration_start`, `cumulative_assessment_score_mean_to_date`, `cumulative_assessment_score_count_to_date`, `cumulative_assessment_weighted_score_to_date`, `cumulative_submitted_weight_to_date`, `assessment_submission_rate_due_to_date`, `late_submission_rate_to_date`, `banked_assessment_rate_to_date`, `current_week_clicks`, `cumulative_clicks_to_date`, `current_week_activity_types`, `cumulative_assessment_clicks_to_date`, `cumulative_content_clicks_to_date`, `cumulative_social_clicks_to_date`, `cumulative_other_clicks_to_date`, `assessment_score_trend_to_date`, `clicks_trend_to_date`, `overall_mastery_proxy`, `current_assessment_cluster_mastery`, `tma_mastery_to_date`, `cma_mastery_to_date`, `exam_mastery_to_date`, `mastery_assessment_coverage_to_date`, `engagement_index_oulad`, `performance_index_oulad`, `discipline_index_oulad`, `has_assessment_score_to_date`, `has_weighted_score_to_date`, `has_vle_activity_to_date`, `has_current_assessment_cluster`, `has_due_assessment_to_date`
