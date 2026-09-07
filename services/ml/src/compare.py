from __future__ import annotations

import argparse
from pathlib import Path

from src.validation.comparison import compare_generated_outputs, write_comparison_reports


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two generated dataset runs or artifact roots.")
    parser.add_argument("--left", required=True, help="Left dataset root, reports directory, or snapshot file.")
    parser.add_argument("--right", required=True, help="Right dataset root, reports directory, or snapshot file.")
    parser.add_argument("--left-label", default=None, help="Optional display label for the left side.")
    parser.add_argument("--right-label", default=None, help="Optional display label for the right side.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for comparison artifacts.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = compare_generated_outputs(
        args.left,
        args.right,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    output_dir = write_comparison_reports(summary, args.output_dir)
    print(f"Comparison written to: {output_dir}")
    print(f"Left: {summary.left_label} ({summary.left_root})")
    print(f"Right: {summary.right_label} ({summary.right_root})")
    print("Key metrics:")
    for row in summary.key_metrics:
        print(f"  - {row['metric']}: left={row['left']} right={row['right']} delta={row['delta']}")


if __name__ == "__main__":
    main()
