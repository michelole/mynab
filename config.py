"""
Configuration file for YNAB Budget Dashboard
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# YNAB API Configuration
YNAB_API_KEY: str | None = os.getenv("YNAB_API_KEY") or os.getenv("YNAB_API_TOKEN")
YNAB_API_TOKEN: str | None = YNAB_API_KEY  # Backward-compatible alias

# App Configuration
CACHE_TTL: int = 3600  # 1 hour cache for API calls
DEFAULT_MONTHS_BACK: int = 24  # Default number of months to fetch
MOVING_AVERAGE_WINDOW: int = 12  # Window for moving average calculation
FORECAST_PERIODS: int = 3  # Number of periods to forecast
