"""Market code constants and enums."""

from enum import Enum


class Market(str, Enum):
    """Standard market codes for creative generation."""

    UNITED_STATES = "US"
    EUROPE = "EU"
    UNITED_KINGDOM = "UK"
    CHINA = "CN"
    JAPAN = "JP"
    KOREA = "KR"


# Common aliases for convenience
US = Market.UNITED_STATES
EU = Market.EUROPE
UK = Market.UNITED_KINGDOM
CN = Market.CHINA
JP = Market.JAPAN
KR = Market.KOREA


def get_default_market() -> str:
    """Get the default market code."""
    return Market.UNITED_STATES.value
