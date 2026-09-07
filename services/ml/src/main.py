from __future__ import annotations

import argparse
from pathlib import Path

from src.generator.config import load_generator_config
from src.generator.pipeline import PipelineSummary
from src.generator.synthetic import SyntheticDatasetGenerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic Student Digital Twin dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/generator_v1.yaml",
        help="Path to the generator YAML config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--num-students",
        type=int,
        default=None,
        help="Optional student count override.",
    )
    parser.add_argument(
        "--num-weeks",
        type=int,
        default=None,
        help="Optional course duration override.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="Optional cohort/group count override.",
    )
    parser.add_argument(
        "--assignments-per-week",
        type=int,
        default=None,
        help="Optional assessed-items-per-week override.",
    )
    parser.add_argument(
        "--sessions-per-week",
        type=int,
        default=None,
        help="Optional class-sessions-per-week override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root; raw/processed/artifacts subdirectories "
            "will be created inside it."
        ),
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip writing the Parquet snapshot output.",
    )
    return parser


def _print_summary(generator: SyntheticDatasetGenerator, summary: PipelineSummary) -> None:
    print(generator.describe())
    print(f"Config: {summary.config_path}")
    print(f"Seed: {summary.seed}")
    print(f"Raw output: {summary.raw_output_dir}")
    print(f"Processed output: {summary.processed_output_dir}")
    print(f"Realism reports: {summary.realism_report_dir}")
    print("Row counts:")
    for table_name, row_count in summary.row_counts.items():
        print(f"  - {table_name}: {row_count}")
    print(f"Risk distribution: {summary.risk_distribution}")
    print(f"Withdrawn students: {summary.withdrawal_count}")


def main() -> None:
    args = _build_parser().parse_args()
    config, resolved_config_path = load_generator_config(
        args.config,
        seed_override=args.seed,
        output_root=args.output_root,
        num_students_override=args.num_students,
        num_weeks_override=args.num_weeks,
        num_groups_override=args.num_groups,
        assignments_per_week_override=args.assignments_per_week,
        sessions_per_week_override=args.sessions_per_week,
    )
    generator = SyntheticDatasetGenerator(config)
    result = generator.run(config_path=resolved_config_path, skip_parquet=args.skip_parquet)
    _print_summary(generator, result.summary)


if __name__ == "__main__":
    main()
