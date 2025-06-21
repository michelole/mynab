"""
Configuration file for YNAB Budget Dashboard
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# YNAB API Configuration
YNAB_API_TOKEN = os.getenv('YNAB_API_TOKEN')

# App Configuration
CACHE_TTL = 3600  # 1 hour cache for API calls
DEFAULT_MONTHS_BACK = 24  # Default number of months to fetch
MOVING_AVERAGE_WINDOW = 12  # Window for moving average calculation
FORECAST_PERIODS = 12  # Number of periods to forecast 