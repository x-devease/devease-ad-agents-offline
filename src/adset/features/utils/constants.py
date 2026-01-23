"""
Shared constants for feature extraction.
"""

# Standard numeric columns used across feature extraction
STANDARD_NUMERIC_COLUMNS = (
    "spend",
    "impressions",
    "clicks",
    "purchase_roas",
    "cpc",
    "cpm",
    "ctr",
)

# Time and lifecycle constants
# Day of week >= 5 means weekend (Saturday=5, Sunday=6)
WEEKEND_DAY_THRESHOLD = 5
LIFECYCLE_COLD_START_DAYS = 3  # Days since ad start for cold_start stage
# Days since ad start for early_learning stage
LIFECYCLE_EARLY_LEARNING_DAYS = 7
LIFECYCLE_LEARNING_DAYS = 21  # Days since ad start for learning stage

# Rolling window constants
ROLLING_WINDOW_DAYS = 7  # Number of days for rolling window calculations
ROLLING_MIN_PERIODS = 1  # Minimum periods for rolling calculations

# ROAS thresholds
LOW_ROAS_THRESHOLD = 0.5  # ROAS threshold for low performance
HIGH_ROAS_THRESHOLD = 2.0  # ROAS threshold for high performance

# Percentage and conversion constants
PERCENTAGE_MULTIPLIER = 100  # Multiplier to convert ratios to percentages
CTR_TO_DECIMAL_DIVISOR = 100  # Divisor to convert CTR percentage to decimal

# Default directory
DEFAULT_DATA_DIR = "datasets"  # Default directory for data files

# Columns to sum when aggregating hourly data to daily
BASE_SUM_COLUMNS = [
    "spend",
    "impressions",
    "clicks",
    "reach",
    "unique_clicks",
    "outbound_clicks",
    "video_30_sec_watched_actions",
    "video_avg_time_watched_actions",
    "video_p100_watched_actions",
    "revenue",
]
# Note: 'reach' is included here for hourly-to-daily aggregation,
# but when aggregating ad-to-adset, 'reach' should be averaged
# (handled in extract.py)
