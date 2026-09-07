from __future__ import annotations

from pathlib import Path

from src.generator.config import GeneratorConfig
from src.generator.pipeline import PipelineResult, run_generation_pipeline


class SyntheticDatasetGenerator:
    """Wrapper around the config-driven synthetic dataset pipeline."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def describe(self) -> str:
        return (
            "SyntheticDatasetGenerator("
            f"schema_version={self.config.schema_version}, "
            f"trajectory_tuning={self.config.trajectory_tuning}, "
            f"students={self.config.num_students}, weeks={self.config.num_weeks}, seed={self.config.seed})"
        )

    def run(
        self,
        *,
        config_path: Path | None = None,
        skip_parquet: bool = False,
    ) -> PipelineResult:
        return run_generation_pipeline(
            self.config,
            config_path=config_path,
            skip_parquet=skip_parquet,
        )
