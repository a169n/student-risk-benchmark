from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from src.generator.config import GeneratorConfig


@dataclass(frozen=True)
class ValidationFinding:
    table: str
    rule: str
    message: str


class DatasetValidationError(Exception):
    pass


@dataclass
class ValidationReport:
    findings: list[ValidationFinding]

    @property
    def is_valid(self) -> bool:
        return not self.findings

    def format(self) -> str:
        lines = ["Dataset validation failed:"]
        for finding in self.findings:
            lines.append(f"- [{finding.table}] {finding.rule}: {finding.message}")
        return "\n".join(lines)


class ValidationSuite:
    def __init__(self, contract_path: Path, config: GeneratorConfig):
        self.contract_path = contract_path
        self.config = config
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        self.table_specs = {table["name"]: table for table in payload["tables"]}

    def validate(self, datasets: dict[str, pd.DataFrame]) -> ValidationReport:
        findings: list[ValidationFinding] = []
        for table_name, spec in self.table_specs.items():
            if table_name not in datasets:
                findings.append(
                    ValidationFinding(
                        table_name,
                        "dataset_presence",
                        "missing generated dataset",
                    )
                )
                continue
            dataframe = datasets[table_name]
            findings.extend(self._validate_table(table_name, dataframe, spec))

        if not findings:
            findings.extend(self._validate_foreign_keys(datasets))
            findings.extend(self._validate_row_counts(datasets))
            findings.extend(self._validate_submission_score_bounds(datasets))

        return ValidationReport(findings)

    def assert_valid(self, datasets: dict[str, pd.DataFrame]) -> None:
        report = self.validate(datasets)
        if not report.is_valid:
            raise DatasetValidationError(report.format())

    def _validate_table(
        self, table_name: str, dataframe: pd.DataFrame, spec: dict[str, Any]
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        fields = spec["fields"]
        expected_columns = [field["name"] for field in fields]
        missing_columns = [column for column in expected_columns if column not in dataframe.columns]
        if missing_columns:
            findings.append(
                ValidationFinding(
                    table_name,
                    "required_columns",
                    f"missing columns: {missing_columns}",
                )
            )
            return findings

        for field in fields:
            column = field["name"]
            series = dataframe[column]
            if field.get("required") and series.isna().any():
                findings.append(
                    ValidationFinding(
                        table_name,
                        "nullability",
                        f"column '{column}' contains null values",
                    )
                )
            findings.extend(
                self._validate_type(table_name, column, series, field.get("type", "string"))
            )
            if "enum" in field and not series.dropna().isin(field["enum"]).all():
                invalid = sorted(set(series.dropna()) - set(field["enum"]))
                findings.append(
                    ValidationFinding(
                        table_name,
                        "enum_membership",
                        f"column '{column}' has invalid values: {invalid}",
                    )
                )
            if "constraints" in field and is_numeric_dtype(series):
                constraints = field["constraints"]
                non_null = series.dropna()
                if (
                    "min" in constraints
                    and not non_null.empty
                    and (non_null < constraints["min"]).any()
                ):
                    findings.append(
                        ValidationFinding(
                            table_name,
                            "numeric_range",
                            f"column '{column}' violates min={constraints['min']}",
                        )
                    )
                if "max" in constraints:
                    max_value = self._max_constraint(table_name, column, constraints["max"])
                    if not non_null.empty and (non_null > max_value).any():
                        findings.append(
                            ValidationFinding(
                                table_name,
                                "numeric_range",
                                f"column '{column}' violates max={max_value}",
                            )
                        )

        primary_key = spec.get("primary_key")
        if primary_key and dataframe[primary_key].duplicated().any():
            findings.append(
                ValidationFinding(
                    table_name,
                    "primary_key_uniqueness",
                    f"duplicate primary key values in '{primary_key}'",
                )
            )

        for unique_constraint in spec.get("unique_constraints", []):
            if dataframe.duplicated(subset=unique_constraint).any():
                findings.append(
                    ValidationFinding(
                        table_name,
                        "unique_constraint",
                        f"duplicate rows for unique constraint {unique_constraint}",
                    )
                )

        return findings

    def _validate_type(
        self, table_name: str, column: str, series: pd.Series, expected_type: str
    ) -> list[ValidationFinding]:
        if expected_type in {"integer", "decimal"}:
            if is_numeric_dtype(series) or series.dropna().empty:
                return []
            return [ValidationFinding(table_name, "dtype", f"column '{column}' must be numeric")]
        if expected_type == "boolean":
            if is_bool_dtype(series):
                return []
            non_null = series.dropna()
            if not non_null.map(lambda value: isinstance(value, bool)).all():
                return [
                    ValidationFinding(
                        table_name,
                        "dtype",
                        f"column '{column}' must be boolean-compatible",
                    )
                ]
        if expected_type in {"date", "datetime"}:
            if is_datetime64_any_dtype(series):
                return []
            try:
                pd.to_datetime(series.dropna())
            except Exception:
                return [
                    ValidationFinding(
                        table_name,
                        "dtype",
                        f"column '{column}' must be date/datetime-compatible",
                    )
                ]
        return []

    def _max_constraint(
        self, table_name: str, column: str, contract_max: int | float
    ) -> int | float:
        if column == "week_number":
            return max(contract_max, self.config.num_weeks)
        if table_name == "final_results" and column == "completed_weeks":
            return max(contract_max, self.config.num_weeks)
        return contract_max

    def _validate_foreign_keys(self, datasets: dict[str, pd.DataFrame]) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        for table_name, spec in self.table_specs.items():
            dataframe = datasets[table_name]
            for foreign_key in spec.get("foreign_keys", []):
                source_column = foreign_key["field"]
                referenced_table, referenced_column = foreign_key["references"].split(".")
                source_values = set(dataframe[source_column].dropna())
                target_values = set(datasets[referenced_table][referenced_column].dropna())
                missing = sorted(source_values - target_values)
                if missing:
                    findings.append(
                        ValidationFinding(
                            table_name,
                            "foreign_key",
                            (
                                f"column '{source_column}' contains values not present in "
                                f"{referenced_table}.{referenced_column}: {missing[:5]}"
                            ),
                        )
                    )
        return findings

    def _validate_row_counts(self, datasets: dict[str, pd.DataFrame]) -> list[ValidationFinding]:
        expected_counts = {
            "students": self.config.num_students,
            "courses": 1,
            "course_topics": self.config.num_weeks,
            "assignments": self.config.num_weeks * self.config.assignments_per_week,
            "attendance": (
                self.config.num_students * self.config.num_weeks * self.config.sessions_per_week
            ),
            "submissions": (
                self.config.num_students
                * self.config.num_weeks
                * self.config.assignments_per_week
            ),
            "weekly_activity": self.config.num_students * self.config.num_weeks,
            "final_results": self.config.num_students,
            "student_twin_snapshots": self.config.num_students * self.config.num_weeks,
        }

        findings: list[ValidationFinding] = []
        for table_name, expected in expected_counts.items():
            actual = len(datasets[table_name])
            if actual != expected:
                findings.append(
                    ValidationFinding(
                        table_name,
                        "row_count",
                        f"expected {expected} rows, got {actual}",
                    )
                )
        return findings

    def _validate_submission_score_bounds(
        self, datasets: dict[str, pd.DataFrame]
    ) -> list[ValidationFinding]:
        submissions = datasets["submissions"]
        assignments = datasets["assignments"][["assignment_id", "max_score"]]
        merged = submissions.merge(assignments, on="assignment_id", how="left")
        invalid = merged.loc[
            merged["score"].notna()
            & ((merged["score"] < 0) | (merged["score"] > merged["max_score"]))
        ]
        if invalid.empty:
            return []
        return [
            ValidationFinding(
                "submissions",
                "score_bounds",
                "score values must be between 0 and assignment.max_score",
            )
        ]
