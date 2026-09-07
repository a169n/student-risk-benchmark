from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLSettings(BaseSettings):
    ml_env: str = "development"
    data_raw_dir: str = "../../data/raw"
    data_processed_dir: str = "../../data/processed"
    data_artifacts_dir: str = "../../data/artifacts"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> MLSettings:
    return MLSettings()
