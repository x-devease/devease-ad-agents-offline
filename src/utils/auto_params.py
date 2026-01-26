"""
Auto-calculate rule parameters from customer data.

Analyzes customer's adset and Shopify data to determine optimal thresholds
for detection rules. Uses percentile-based approach with guardrails.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

from src.utils import Config


class AutoParams:
    """
    Auto-calculate rule parameters from customer data.

    Calculates performance thresholds (ROAS, spend, CTR, CPC, etc.) from
    data distribution using percentiles. Fills non-calculable parameters
    with domain knowledge defaults.
    """

    # Domain knowledge defaults (cannot be calculated from data)
    DEFAULT_PARAMS = {
        "age_targeting": {
            "broad_age_range": 30,  # Subjective business rule
            "narrow_age_range": 5,  # Subjective business rule
            "age_min_bound": 18,  # Platform constraint
            "age_max_bound": 65,  # Platform constraint
            "test_segment_size": 20,  # Testing strategy
            "test_segment_mid_offset": 10,
            "narrow_range_expansion": 5,
            "broad_classification_threshold": 45,
        },
        "gender": {
            "total_genders_count": 2,  # Platform fact
            "min_custom_audiences_count": 0,
        },
        "geographic": {
            "max_countries": 5,  # Strategic choice
            "max_countries_display": 10,  # Display preference
            "recommended_countries_to_test": 3,  # Testing strategy
        },
        "engagement": {
            # Statistical minimums
            "min_impressions_for_ctr": 100,
            "min_impressions_for_video": 100,
            "min_clicks_for_conversion": 10,
            "min_clicks_for_format_check": 10,
        },
        "shopify": {
            "top_buyer_states_count": 5,  # Fixed business rule
        },
        "segment_health": {
            "critical_ratio_threshold": 0.3,  # Business decision
            "high_ratio_threshold": 0.3,
            "medium_ratio_threshold": 0.5,
        },
    }

    # Guardrails to prevent extreme values
    GUARDRAILS = {
        "roas": {
            "high_min": 0.5,  # Minimum 0.5x for "high"
            "high_max": 10.0,  # Maximum 10x for "high"
            "low_min": 0.5,  # Minimum 0.5x floor
            "low_max": 2.0,  # Maximum 2.0x for "low"
        },
        "spend": {
            "min_min": 10.0,  # Minimum $10 for scaling
            "high_max": 1000.0,  # Maximum $1000 for "high"
        },
        "engagement": {
            "ctr_min": 0.001,  # Minimum 0.1% CTR
            "ctr_max": 0.05,  # Maximum 5% CTR
            "cpc_min": 0.5,  # Minimum $0.50 CPC
            "cpc_max": 20.0,  # Maximum $20 CPC
            "conversion_rate_min": 0.001,  # Minimum 0.1%
        },
        "cpm": {
            "min": 5.0,  # Minimum $5 CPM
            "max": 100.0,  # Maximum $100 CPM
        },
    }

    @staticmethod
    def calculate_from_data(customer: str, platform: str = "meta") -> Dict:
        """
        Calculate optimal parameters from customer data.

        Args:
            customer: Customer name
            platform: Platform name (default: "meta")

        Returns:
            Dictionary of calculated parameters
        """
        print(f"Auto-calculating parameters for {customer}/{platform}...")

        # Load data
        from src.meta.adset.allocator.features.feature_store import FeatureStore

        fs = FeatureStore(customer=customer, platform=platform)

        adset_data = fs.load_adset_data()
        shopify_data = fs.load_shopify_data()

        # Validate data quality
        if len(adset_data) < 100:
            print(f"  WARNING: Only {len(adset_data)} adsets, using defaults")
            return AutoParams._get_defaults_only()

        # Check data quality and warn
        AutoParams._validate_data_quality(adset_data)

        # Calculate parameters
        params = AutoParams._calculate_performance_params(adset_data)
        shopify_params = AutoParams._calculate_shopify_params(shopify_data)
        params.update(shopify_params)

        # Merge with defaults (fill in non-calculable params)
        params = AutoParams._merge_with_defaults(params)

        # Save to params.yaml (pass adset count for metadata)
        AutoParams._save_params(customer, platform, params, len(adset_data))

        print(f"  Updated params.yaml for {customer}/{platform}")
        return params

    @staticmethod
    def _calculate_performance_params(adset_data: pd.DataFrame) -> Dict:
        """
        Calculate performance threshold parameters from adset data.

        Uses percentile-based approach with guardrails.
        """
        params = {}

        # ROAS thresholds - use positive ROAS only for meaningful percentiles
        roas_col = "purchase_roas"
        if roas_col in adset_data.columns:
            roas_all = adset_data[roas_col].dropna()
            roas_positive = roas_all[roas_all > 0]

            # Use positive ROAS if we have enough samples, otherwise use all
            roas = roas_positive if len(roas_positive) >= 50 else roas_all

            if len(roas) > 0:
                params["roas"] = {
                    "high_roas_threshold": AutoParams._percentile_with_guardrails(
                        roas, 75, "roas", "high"
                    ),
                    "medium_roas_threshold": float(np.percentile(roas, 50)),
                    "low_roas_threshold": AutoParams._percentile_with_guardrails(
                        roas, 25, "roas", "low"
                    ),
                    "high_roas_low_spend_threshold": float(np.percentile(roas, 50)),
                    "video_preferred_roas_threshold": float(np.percentile(roas, 50)),
                    "warm_audience_roas_threshold": float(np.percentile(roas, 75)),
                }

        # Spend thresholds
        spend_col = "spend"
        if spend_col in adset_data.columns:
            spend = adset_data[spend_col][adset_data[spend_col] > 0].dropna()
            if len(spend) > 0:
                params["spend"] = {
                    "high_spend_threshold": AutoParams._percentile_with_guardrails(
                        spend, 75, "spend", "high"
                    ),
                    "medium_spend_threshold": float(np.percentile(spend, 50)),
                    "low_spend_threshold": float(np.percentile(spend, 25)),
                    "min_spend_for_scale": max(float(np.percentile(spend, 10)), 10.0),
                    "min_spend_for_video_format_check": float(np.percentile(spend, 10)),
                }

        # Engagement thresholds
        if "ctr" in adset_data.columns:
            # CTR is stored as percentage (2.69 = 2.69%), convert to decimal
            ctr = adset_data["ctr"].dropna() / 100.0
            if len(ctr) > 0:
                params["engagement"] = params.get("engagement", {})
                params["engagement"]["low_ctr_threshold"] = (
                    AutoParams._percentile_with_guardrails(ctr, 25, "engagement", "ctr")
                )
                params["engagement"]["image_preferred_ctr_threshold"] = float(
                    np.percentile(ctr, 75)
                )

        if "cpc" in adset_data.columns:
            cpc = adset_data["cpc"].dropna()
            if len(cpc) > 0:
                params["engagement"] = params.get("engagement", {})
                params["engagement"]["high_cpc_threshold"] = (
                    AutoParams._percentile_with_guardrails(cpc, 75, "engagement", "cpc")
                )

        if "conversion_rate" in adset_data.columns:
            conv_rate = adset_data["conversion_rate"].dropna()
            if len(conv_rate) > 0:
                params["engagement"] = params.get("engagement", {})
                params["engagement"]["low_conversion_rate_threshold"] = float(
                    np.percentile(conv_rate, 25)
                )

        # CPM thresholds
        if "cpm" in adset_data.columns:
            cpm = adset_data["cpm"].dropna()
            if len(cpm) > 0:
                p75 = float(np.percentile(cpm, 75))
                # Apply guardrails
                p75 = max(AutoParams.GUARDRAILS["cpm"]["min"], p75)
                p75 = min(AutoParams.GUARDRAILS["cpm"]["max"], p75)
                params["cpm"] = {"high_cpm_threshold": p75}

        # Reach thresholds - use positive reach only
        if "reach" in adset_data.columns:
            reach_all = adset_data["reach"].dropna()
            reach_positive = reach_all[reach_all > 0]

            # Use positive reach if we have enough samples, otherwise use all
            reach = reach_positive if len(reach_positive) >= 50 else reach_all

            if len(reach) > 0:
                params["reach"] = {
                    "low_reach_threshold": float(np.percentile(reach, 25))
                }

        # Frequency thresholds - use meaningful frequency with business logic guardrail
        if "frequency" in adset_data.columns:
            freq_all = adset_data["frequency"].dropna()
            # Filter to frequency > 1.0 (meaningful engagement: saw ad at least once)
            freq_positive = freq_all[freq_all > 1.0]

            # Use positive frequency if we have enough samples, otherwise use all
            freq = freq_positive if len(freq_positive) >= 50 else freq_all[freq_all > 0]

            if len(freq) > 0:
                # Business logic: Frequency < 3.0 is NOT saturation
                # Min of 3.0 and 90th percentile prevents flagging under-exposed ads
                high_freq_calc = float(np.percentile(freq, 90))
                high_freq_threshold = max(3.0, high_freq_calc)

                params["frequency"] = {
                    "high_frequency_threshold": high_freq_threshold,
                    "optimal_frequency": float(np.percentile(freq, 50)),
                    "medium_frequency": float(np.percentile(freq, 75)),
                    "low_frequency": float(np.percentile(freq, 10)),
                }

        # Opportunity size thresholds (duplicate of above for convenience)
        if "frequency" in adset_data.columns:
            freq_all = adset_data["frequency"].dropna()
            freq_positive = freq_all[freq_all > 1.0]
            freq = freq_positive if len(freq_positive) >= 50 else freq_all[freq_all > 0]

            if len(freq) > 0:
                if "opportunity_size" not in params:
                    params["opportunity_size"] = {}
                params["opportunity_size"]["high_frequency_threshold"] = float(
                    np.percentile(freq, 25)
                )
                params["opportunity_size"]["medium_frequency_threshold"] = float(
                    np.percentile(freq, 50)
                )
                params["opportunity_size"]["low_frequency_threshold"] = float(
                    np.percentile(freq, 75)
                )

        if "purchase_roas" in adset_data.columns:
            roas_all = adset_data["purchase_roas"].dropna()
            roas_positive = roas_all[roas_all > 0]

            # Use positive ROAS if we have enough samples, otherwise use all
            roas = roas_positive if len(roas_positive) >= 50 else roas_all

            if len(roas) > 0:
                if "opportunity_size" not in params:
                    params["opportunity_size"] = {}
                params["opportunity_size"]["high_roas_threshold"] = float(
                    np.percentile(roas, 75)
                )
                params["opportunity_size"]["medium_roas_threshold"] = float(
                    np.percentile(roas, 50)
                )

        if "spend" in adset_data.columns:
            spend = adset_data[spend_col][adset_data[spend_col] > 0].dropna()
            if len(spend) > 0:
                if "opportunity_size" not in params:
                    params["opportunity_size"] = {}
                params["opportunity_size"]["high_spend_threshold"] = float(
                    np.percentile(spend, 75)
                )
                params["opportunity_size"]["medium_spend_threshold"] = float(
                    np.percentile(spend, 50)
                )

        return params

    @staticmethod
    def _validate_data_quality(adset_data: pd.DataFrame) -> None:
        """
        Validate data quality and print warnings.

        Args:
            adset_data: Adset DataFrame to validate
        """
        print(f"\n  Data Quality Report:")
        print(f"  Total adsets: {len(adset_data)}")

        # Check ROAS quality
        if "purchase_roas" in adset_data.columns:
            roas = adset_data["purchase_roas"].dropna()
            roas_positive = roas[roas > 0]
            roas_zero_pct = (roas == 0).sum() / len(roas) * 100 if len(roas) > 0 else 0

            print(
                f"  ROAS: {len(roas)} non-null, {len(roas_positive)} positive ({len(roas_positive)/len(roas)*100:.1f}%)"
            )

            if roas_zero_pct > 90:
                print(f"    ⚠️  WARNING: {roas_zero_pct:.1f}% of adsets have zero ROAS")
                print(f"    ⚠️  Using positive ROAS only for threshold calculation")

        # Check spend quality
        if "spend" in adset_data.columns:
            spend = adset_data["spend"]
            spend_positive = spend[spend > 0]
            spend_zero_pct = (
                (spend == 0).sum() / len(spend) * 100 if len(spend) > 0 else 0
            )

            print(
                f"  Spend: {len(spend_positive)} positive ({len(spend_positive)/len(spend)*100:.1f}%)"
            )

            if spend_zero_pct > 80:
                print(
                    f"    ⚠️  WARNING: {spend_zero_pct:.1f}% of adsets have zero spend"
                )

        # Check CTR quality
        if "ctr" in adset_data.columns:
            ctr = adset_data["ctr"].dropna()
            print(
                f"  CTR: {len(ctr)} non-null (range: {ctr.min():.2f}% - {ctr.max():.2f}%)"
            )

        # Check frequency quality
        if "frequency" in adset_data.columns:
            freq = adset_data["frequency"].dropna()
            freq_positive = freq[freq > 1.0]
            freq_zero_pct = (freq == 0).sum() / len(freq) * 100 if len(freq) > 0 else 0

            print(
                f"  Frequency: {len(freq_positive)} > 1.0 ({len(freq_positive)/len(freq)*100:.1f}%)"
            )

            if freq_zero_pct > 80:
                print(
                    f"    ⚠️  WARNING: {freq_zero_pct:.1f}% of adsets have zero frequency"
                )
                print(f"    ⚠️  Using frequency > 1.0 only for threshold calculation")
                print(f"    ⚠️  High frequency threshold = max(3.0, 90th percentile)")

        print()  # Empty line for readability

    @staticmethod
    def _calculate_shopify_params(shopify_data: pd.DataFrame) -> Dict:
        """
        Calculate Shopify-based parameters from order data.

        Args:
            shopify_data: Shopify order DataFrame

        Returns:
            Dictionary of Shopify parameters
        """
        params = {}

        if shopify_data is None or len(shopify_data) == 0:
            return params

        # Calculate buyer age threshold
        # Note: Shopify data doesn't directly have ages, so we use a heuristic
        # In production, you'd need actual customer birthdays
        params["shopify"] = {
            "min_buyer_age_threshold": 35,  # Default if can't calculate
            "buyer_state_concentration_threshold": 0.4,  # Default
        }

        # Try to extract age from customer data if available
        if "customer_age" in shopify_data.columns:
            ages = shopify_data["customer_age"].dropna()
            if len(ages) > 0:
                # Use 5th percentile as floor (conservative)
                params["shopify"]["min_buyer_age_threshold"] = int(
                    np.percentile(ages, 5)
                )

        # Calculate geographic concentration
        if "Shipping Province" in shopify_data.columns:
            state_counts = shopify_data["Shipping Province"].value_counts().head(5)
            total_orders = len(shopify_data)
            if total_orders > 0:
                concentration = state_counts.sum() / total_orders
                params["shopify"]["buyer_state_concentration_threshold"] = float(
                    round(concentration * 0.9, 2)  # 90% of observed concentration
                )

        return params

    @staticmethod
    def _merge_with_defaults(calculated: Dict) -> Dict:
        """
        Merge calculated parameters with domain knowledge defaults.

        Args:
            calculated: Calculated parameters

        Returns:
            Complete parameter dictionary
        """
        # Start with defaults
        params = AutoParams._deep_copy_dict(AutoParams.DEFAULT_PARAMS)

        # Update with calculated values (preserve what we calculated)
        for section, values in calculated.items():
            if section not in params:
                params[section] = {}
            params[section].update(values)

        return params

    @staticmethod
    def _percentile_with_guardrails(
        data: pd.Series, percentile: float, param_type: str, threshold_type: str
    ) -> float:
        """
        Calculate percentile with guardrails.

        Args:
            data: Data series
            percentile: Percentile to calculate (0-100)
            param_type: Parameter type (roas, spend, engagement)
            threshold_type: Threshold type (high, low)

        Returns:
            Guardrailed percentile value
        """
        value = float(np.percentile(data, percentile))

        # Apply guardrails if defined
        if param_type in AutoParams.GUARDRAILS:
            guardrails = AutoParams.GUARDRAILS[param_type]

            if threshold_type == "high":
                key = f"{threshold_type}_min"
                if key in guardrails:
                    value = max(value, guardrails[key])
                key = f"{threshold_type}_max"
                if key in guardrails:
                    value = min(value, guardrails[key])
            elif threshold_type == "low":
                key = f"{threshold_type}_min"
                if key in guardrails:
                    value = max(value, guardrails[key])
                key = f"{threshold_type}_max"
                if key in guardrails:
                    value = min(value, guardrails[key])

        return value

    @staticmethod
    def _save_params(
        customer: str, platform: str, params: Dict, adset_count: int
    ) -> None:
        """
        Save calculated parameters to params.yaml.

        Preserves file structure and comments where possible.

        Args:
            customer: Customer name
            platform: Platform name
            params: Parameter dictionary
            adset_count: Number of adsets analyzed
        """
        config_path = Config.CONFIG_DIR / "adset" / "generator" / customer / platform / "params.yaml"

        # Load existing params to preserve structure
        existing_params = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                existing_params = yaml.safe_load(f) or {}

        # Update existing params with calculated values
        # This preserves comments and structure
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(existing_params, params)

        # Add metadata
        existing_params["_metadata"] = {
            "auto_calculated": True,
            "calculated_at": datetime.now().isoformat(),
            "adsets_analyzed": adset_count,
        }

        # Save
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(existing_params, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _get_defaults_only() -> Dict:
        """Get default parameters without calculation."""
        return AutoParams._deep_copy_dict(AutoParams.DEFAULT_PARAMS)

    @staticmethod
    def _deep_copy_dict(d: Dict) -> Dict:
        """Deep copy a dictionary."""
        import copy

        return copy.deepcopy(d)
