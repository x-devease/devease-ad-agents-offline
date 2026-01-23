"""
JSON parsing utilities for nested structures.
Handles parsing of complex JSON columns like adset_targeting.
"""

import json
from typing import Dict

import numpy as np
import pandas as pd


class JSONParser:
    """Utilities for parsing nested JSON structures."""

    @staticmethod
    def parse_json_column(series: pd.Series) -> pd.Series:
        """
        Parse JSON string column into Python objects.

        Args:
            series: Series containing JSON strings

        Returns:
            Series with parsed JSON objects (or NaN for invalid JSON)
        """

        def parse_value(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (dict, list)):
                return val  # Already parsed
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, ValueError, TypeError):
                    return np.nan
            return np.nan

        return series.apply(parse_value)

    @staticmethod
    def _extract_age_features(targeting: dict, feat: dict) -> None:
        """Extract age-related features from targeting dict."""
        if "age_min" in targeting:
            feat["targeting_age_min"] = targeting["age_min"]
        if "age_max" in targeting:
            feat["targeting_age_max"] = targeting["age_max"]
        if "age_range" in targeting and isinstance(targeting["age_range"], list):
            feat["targeting_age_range"] = str(targeting["age_range"])

    @staticmethod
    def _extract_gender_features(targeting: dict, feat: dict) -> None:
        """Extract gender-related features from targeting dict."""
        if "genders" in targeting:
            feat["targeting_genders"] = str(targeting["genders"])

    @staticmethod
    def _extract_geo_features(targeting: dict, feat: dict) -> None:
        """Extract geo location features from targeting dict."""
        if "geo_locations" in targeting and isinstance(
            targeting["geo_locations"], dict
        ):
            geo = targeting["geo_locations"]
            if "countries" in geo:
                feat["targeting_countries"] = str(geo["countries"])
            if "location_types" in geo:
                feat["targeting_location_types"] = str(geo["location_types"])

    @staticmethod
    def _extract_automation_features(targeting: dict, feat: dict) -> None:
        """Extract targeting automation features from targeting dict."""
        if "targeting_automation" in targeting and isinstance(
            targeting["targeting_automation"], dict
        ):
            auto = targeting["targeting_automation"]
            if "advantage_audience" in auto:
                feat["targeting_advantage_audience"] = auto["advantage_audience"]
            if "individual_setting" in auto and isinstance(
                auto["individual_setting"], dict
            ):
                ind = auto["individual_setting"]
                if "age" in ind:
                    feat["targeting_auto_age"] = ind["age"]
                if "gender" in ind:
                    feat["targeting_auto_gender"] = ind["gender"]

    @staticmethod
    def _extract_list_feature(
        targeting: dict, feat: dict, key: str, feat_key: str
    ) -> None:
        """Extract a list feature from targeting dict."""
        if key in targeting and isinstance(targeting[key], list):
            items = [
                i.get("name", i) if isinstance(i, dict) else i for i in targeting[key]
            ]
            feat[feat_key] = str(items)

    @staticmethod
    def _extract_custom_audiences(targeting: dict, feat: dict) -> None:
        """Extract custom audiences features from targeting dict."""
        if "custom_audiences" in targeting and isinstance(
            targeting["custom_audiences"], list
        ):
            feat["targeting_custom_audiences_count"] = len(
                targeting["custom_audiences"]
            )
            items = [
                a.get("name", a) if isinstance(a, dict) else a
                for a in targeting["custom_audiences"]
            ]
            feat["targeting_custom_audiences"] = str(items)

    @staticmethod
    def _extract_excluded_audiences(targeting: dict, feat: dict) -> None:
        """Extract excluded custom audiences features from targeting dict."""
        if "excluded_custom_audiences" in targeting and isinstance(
            targeting["excluded_custom_audiences"], list
        ):
            feat["targeting_excluded_custom_audiences_count"] = len(
                targeting["excluded_custom_audiences"]
            )

    @staticmethod
    def extract_targeting_features(targeting_series: pd.Series) -> pd.DataFrame:
        """
        Extract features from adset_targeting JSON structure.

        Args:
            targeting_series: Series containing adset_targeting JSON
                strings/objects

        Returns:
            DataFrame with extracted targeting features
        """
        parsed = JSONParser.parse_json_column(targeting_series)
        feature_dicts = {}

        for idx, targeting in parsed.items():
            if pd.isna(targeting) or not isinstance(targeting, dict):
                continue

            feat = {}
            JSONParser._extract_age_features(targeting, feat)
            JSONParser._extract_gender_features(targeting, feat)
            JSONParser._extract_geo_features(targeting, feat)
            JSONParser._extract_automation_features(targeting, feat)
            JSONParser._extract_list_feature(
                targeting, feat, "interests", "targeting_interests"
            )
            JSONParser._extract_list_feature(
                targeting, feat, "behaviors", "targeting_behaviors"
            )
            JSONParser._extract_custom_audiences(targeting, feat)
            JSONParser._extract_excluded_audiences(targeting, feat)

            if feat:
                feature_dicts[idx] = feat

        if feature_dicts:
            return pd.DataFrame.from_dict(feature_dicts, orient="index")
        return pd.DataFrame(index=parsed.index)

    @staticmethod
    def flatten_json_column(
        series: pd.Series, column_name: str, max_depth: int = 3
    ) -> pd.DataFrame:
        """
        Flatten a JSON column into multiple columns.

        Args:
            series: Series containing JSON strings/objects
            column_name: Base name for the column
            max_depth: Maximum depth to flatten

        Returns:
            DataFrame with flattened columns
        """
        parsed = JSONParser.parse_json_column(series)
        flattened_rows = []

        def flatten_dict(
            data_dict: Dict,
            prefix: str,
            depth: int = 0,
            result_dict: Dict = None,
        ):
            if result_dict is None:
                result_dict = {}
            if depth > max_depth:
                return result_dict
            for key, value in data_dict.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key, depth + 1, result_dict)
                elif isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], dict):
                        # Flatten first item as example
                        flatten_dict(value[0], f"{new_key}_0", depth + 1, result_dict)
                    else:
                        result_dict[new_key] = str(value)
                else:
                    result_dict[new_key] = value
            return result_dict

        for val in parsed:
            if pd.isna(val):
                flattened_rows.append({})
            elif isinstance(val, dict):
                flattened = flatten_dict(val, column_name)
                flattened_rows.append(flattened)
            else:
                flattened_rows.append({})

        if flattened_rows:
            return pd.DataFrame(flattened_rows, index=parsed.index)
        return pd.DataFrame(index=parsed.index)
