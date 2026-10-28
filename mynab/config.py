"""Configuration settings for the YNAB Budget Dashboard using pydantic-settings."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ynab_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("YNAB_API_KEY", "YNAB_API_TOKEN"),
    )
    cache_ttl: int = 3600
    default_months_back: int = 24
    moving_average_window: int = 12
    forecast_periods: int = 3


settings = Settings()

# Backward-compatible module-level constants
YNAB_API_KEY: str | None = settings.ynab_api_key
YNAB_API_TOKEN: str | None = settings.ynab_api_key
CACHE_TTL: int = settings.cache_ttl
DEFAULT_MONTHS_BACK: int = settings.default_months_back
MOVING_AVERAGE_WINDOW: int = settings.moving_average_window
FORECAST_PERIODS: int = settings.forecast_periods
