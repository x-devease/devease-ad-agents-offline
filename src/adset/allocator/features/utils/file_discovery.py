"""
File discovery utilities.
Handles automatic discovery of data files in directories.
"""

from pathlib import Path
from typing import Dict, Optional

from .constants import DEFAULT_DATA_DIR


class FileDiscovery:  # pylint: disable=too-few-public-methods
    """Handles automatic discovery of data files."""

    @staticmethod
    def discover_data_files(
        data_dir: str = DEFAULT_DATA_DIR,
        account_file: Optional[str] = None,
        campaign_file: Optional[str] = None,
        adset_file: Optional[str] = None,
        ad_file: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Discover data files in the specified directory.

        Args:
            data_dir: Directory containing data files
            account_file: Optional explicit account file path
            campaign_file: Optional explicit campaign file path
            adset_file: Optional explicit adset file path
            ad_file: Optional explicit ad file path

        Returns:
            Dictionary mapping data types to file paths
        """
        data_dir_path = Path(data_dir)
        files = {}

        # Discover account file
        if account_file is None:
            account_files = list(data_dir_path.glob("*account*.csv"))
            files["account"] = str(account_files[0]) if account_files else None
        else:
            files["account"] = account_file

        # Discover campaign file
        if campaign_file is None:
            campaign_files = list(data_dir_path.glob("*campaign*.csv"))
            files["campaign"] = str(campaign_files[0]) if campaign_files else None
        else:
            files["campaign"] = campaign_file

        # Discover adset file
        if adset_file is None:
            adset_files = list(data_dir_path.glob("*adset*.csv"))
            files["adset"] = str(adset_files[0]) if adset_files else None
        else:
            files["adset"] = adset_file

        # Discover ad file (exclude adset files)
        if ad_file is None:
            all_ad_files = list(data_dir_path.glob("*ad*.csv"))
            # Filter out adset files - they contain "ad" but should not match
            ad_files = [f for f in all_ad_files if "adset" not in f.name.lower()]
            # Prefer daily files over hourly files
            daily_files = [f for f in ad_files if "daily" in f.name.lower()]
            hourly_files = [f for f in ad_files if "hourly" in f.name.lower()]
            if daily_files:
                files["ad"] = str(daily_files[0])
            elif hourly_files:
                files["ad"] = str(hourly_files[0])
            elif ad_files:
                files["ad"] = str(ad_files[0])
            else:
                files["ad"] = None
        else:
            files["ad"] = ad_file

        return files
