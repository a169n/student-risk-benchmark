# Public Benchmark vs Synthetic Interpretation

This document compares the direction of the OULAD representation-transfer benchmark with the earlier synthetic experiment line. It does not compare dataset quality.

## Synthetic carry-forward state

- The full Twin representation was not justified by `exp_001`/`exp_002`.
- `B_lms_plus_mastery` was carried forward as the lean Twin candidate.
- `exp_004` found the lean candidate interpretable with an `overall_mastery` redundancy caveat.

## OULAD benchmark result

The OULAD benchmark is directionally consistent with the synthetic carry-forward decision: the lean mastery analogue improves over the OULAD LMS baseline on the primary grouped split. This supports the representation-transfer logic with caveats, not full external validity.

## External validity boundary

The benchmark is a public-dataset stress test of representation logic. It can support, weaken, or complicate the synthetic finding, but it does not establish full external validity or institutional deployment readiness.

Outcome: `supports_with_caveats`.
Short conclusion: `C_twin_oulad` improved over `B_lms_oulad` on OULAD.
