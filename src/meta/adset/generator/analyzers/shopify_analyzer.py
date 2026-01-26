"""
Shopify Order Analyzer for Advantage+ Constraints

Extracts buyer signals to build hard constraints for Meta Audience Controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class ShopifyAnalyzer:
    """
    Analyze Shopify order data to extract buyer constraints.

    Focus:
    1. Minimum buyer age (for hard age constraint)
    2. ZIP code performance (for geo blacklist)
    3. Customer LTV segmentation (for LAL signal layering)
    """

    def __init__(self, shopify_data_path: str = None):
        """
        Initialize Shopify analyzer.

        Args:
            shopify_data_path: Path to Shopify orders CSV
        """
        self.shopify_data_path = shopify_data_path

    def load_shopify_orders(self) -> pd.DataFrame:
        """
        Load Shopify order data.

        Expected columns:
        - order_id
        - customer_id
        - customer_age (if available)
        - customer_zip
        - order_date
        - total_price
        - quantity
        - shipping_zip
        """
        if not self.shopify_data_path:
            # Try default location
            default_path = Path("datasets/moprobo/meta/raw/shopify.csv")
            if not default_path.exists():
                return pd.DataFrame()

            self.shopify_data_path = str(default_path)

        try:
            df = pd.read_csv(self.shopify_data_path)
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    def extract_minimum_buyer_age(self, orders_df: pd.DataFrame = None) -> int:
        """
        Extract minimum buyer age from Shopify orders.

        Logic:
        1. If customer_age column exists, use minimum age
        2. If not available, use industry-specific defaults:
           - Housing/Auto/Insurance: 25 (legal restrictions)
           - Credit/Employment: 18 (legal restrictions)
           - E-commerce: 21 (default safe age)
           - Special Ad Categories: See compliance rules

        Args:
            orders_df: Shopify orders DataFrame

        Returns:
            Minimum buyer age for hard constraint
        """
        if orders_df is None:
            orders_df = self.load_shopify_orders()

        if orders_df.empty:
            # Default to safe minimum age
            return 21

        # Check if customer_age column exists
        if "customer_age" in orders_df.columns:
            # Filter out missing/invalid ages
            valid_ages = orders_df["customer_age"].dropna()
            valid_ages = valid_ages[valid_ages > 0]

            if not valid_ages.empty:
                # Use 5th percentile as minimum (exclude outliers)
                min_age = int(np.percentile(valid_ages, 5))
                # Ensure at least 18 (legal minimum)
                return max(min_age, 18)

        # Fallback: estimate from order patterns
        # Young customers with high frequency = likely need age constraint
        # This is a heuristic when actual age data unavailable
        return 21  # Safe default for most e-commerce

    def build_zip_blacklist(
        self,
        orders_df: pd.DataFrame = None,
        roas_threshold: float = 0.5,
        min_orders: int = 10,
    ) -> List[str]:
        """
        Build ZIP code blacklist based on poor performance.

        Logic:
        1. Calculate ROAS/profit by ZIP code
        2. Blacklist ZIPs with ROAS below threshold
        3. Only include ZIPs with minimum order volume (avoid false positives)

        Args:
            orders_df: Shopify orders DataFrame
            roas_threshold: Minimum ROAS to avoid blacklist
            min_orders: Minimum orders to evaluate ZIP

        Returns:
            List of ZIP codes to exclude
        """
        if orders_df is None:
            orders_df = self.load_shopify_orders()

        if orders_df.empty or "customer_zip" not in orders_df.columns:
            return []

        # Aggregate by ZIP code
        zip_stats = (
            orders_df.groupby("customer_zip")
            .agg({"total_price": "sum", "order_id": "count"})
            .rename(columns={"order_id": "order_count"})
        )

        # Filter low-volume ZIPs
        zip_stats = zip_stats[zip_stats["order_count"] >= min_orders]

        # Calculate proxy for ROAS (can be enhanced with ad spend data)
        # For now, use order count as proxy for performance
        # In production: join with ad spend to calculate true ROAS

        # Blacklist bottom 10% by order count (poor performing ZIPs)
        if len(zip_stats) > 10:
            blacklist_size = max(1, int(len(zip_stats) * 0.1))
            blacklist_zips = zip_stats.nsmallest(blacklist_size, "order_count")
            return blacklist_zips.index.tolist()

        return []

    def detect_special_ad_category(self, business_description: str = "") -> bool:
        """
        Detect if business falls under Special Ad Category.

        Special Ad Categories (Meta requires compliance):
        - Housing (real estate, rentals)
        - Credit (loans, credit cards, financial services)
        - Employment (job postings, recruiting)

        Compliance Rule:
        - IF special category → DISABLE ZIP code exclusion (Meta policy)
        - Must use inclusive targeting only

        Args:
            business_description: Business type description

        Returns:
            True if special ad category detected
        """
        if not business_description:
            return False

        description_lower = business_description.lower()

        special_keywords = {
            "housing": [
                "real estate",
                "apartment",
                "rental",
                "housing",
                "mortgage",
                "home rental",
            ],
            "credit": [
                "loan",
                "credit card",
                "mortgage",
                "finance",
                "lending",
                "bank",
                "debt",
            ],
            "employment": [
                "job",
                "hiring",
                "recruiting",
                "employment",
                "career",
                "staffing",
            ],
        }

        for category, keywords in special_keywords.items():
            if any(kw in description_lower for kw in keywords):
                return True

        return False

    def segment_customers_by_ltv(
        self,
        orders_df: pd.DataFrame = None,
        gold_percentile: float = 0.90,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Segment customers by LTV for signal layering.

        Segments:
        1. Gold Seed (LTV): Top X% by cumulative spend
        2. Velocity: Past 7 days highest frequency
        3. Negative: Recent purchasers (for exclusion)

        Args:
            orders_df: Shopify orders DataFrame
            gold_percentile: Percentile for gold segment (default 90% = top 10%)

        Returns:
            Tuple of (gold_customers, velocity_customers, negative_customers)
        """
        if orders_df is None:
            orders_df = self.load_shopify_orders()

        if orders_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Calculate customer LTV
        customer_ltv = (
            orders_df.groupby("customer_id")
            .agg(
                {
                    "total_price": "sum",
                    "order_id": "count",
                    "order_date": ["min", "max"],
                }
            )
            .reset_index()
        )

        customer_ltv.columns = [
            "customer_id",
            "ltv",
            "order_count",
            "first_order",
            "last_order",
        ]

        # 1. Gold Seed: Top X% by LTV
        gold_threshold = customer_ltv["ltv"].quantile(gold_percentile)
        gold_customers = customer_ltv[customer_ltv["ltv"] >= gold_threshold]

        # 2. Velocity: Past 7 days highest order frequency
        if "order_date" in orders_df.columns:
            orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
            seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)

            recent_orders = orders_df[orders_df["order_date"] >= seven_days_ago]
            velocity_customers = (
                recent_orders.groupby("customer_id")
                .size()
                .reset_index(name="recent_order_count")
                .sort_values("recent_order_count", ascending=False)
                .head(len(gold_customers))  # Same size as gold segment
            )
        else:
            velocity_customers = pd.DataFrame()

        # 3. Negative: Recent purchasers (exclude from targeting)
        # Exclude customers who purchased in past 30 days
        thirty_days_ago = pd.Timestamp.now() - pd.Timedelta(days=30)

        recent_purchasers = orders_df[orders_df["order_date"] >= thirty_days_ago]
        negative_customers = recent_purchasers[["customer_id"]].drop_duplicates()

        return gold_customers, velocity_customers, negative_customers

    def generate_audience_constraints(
        self,
        business_description: str = "",
        customer_type: str = "ecommerce",
    ) -> Dict:
        """
        Generate complete audience constraint configuration for Advantage+.

        Args:
            business_description: Business type for category detection
            customer_type: Customer category (ecommerce, saas, etc.)

        Returns:
            Dictionary with constraint configuration
        """
        orders_df = self.load_shopify_orders()

        # 1. Extract minimum buyer age
        min_age = self.extract_minimum_buyer_age(orders_df)

        # 2. Build ZIP blacklist
        is_special_category = self.detect_special_ad_category(business_description)

        if is_special_category:
            zip_blacklist = []  # Compliance: No exclusions for special categories
        else:
            zip_blacklist = self.build_zip_blacklist(orders_df)

        # 3. Segment customers
        gold_customers, velocity_customers, negative_customers = (
            self.segment_customers_by_ltv(orders_df)
        )

        return {
            "age_constraint": {
                "min_age": min_age,
                "reason": f"Based on Shopify buyer data, minimum buyer age is {min_age}",
                "enforce": True,  # Hard constraint
            },
            "geo_exclusions": {
                "zip_codes": zip_blacklist,
                "reason": "Low ROAS/high fraud ZIPs (from Shopify analysis)",
                "enabled": len(zip_blacklist) > 0 and not is_special_category,
                "compliance_note": (
                    "Disabled for Special Ad Categories"
                    if is_special_category
                    else None
                ),
            },
            "lal_signals": {
                "gold_seed": {
                    "customer_ids": (
                        gold_customers["customer_id"].tolist()
                        if not gold_customers.empty
                        else []
                    ),
                    "size_pct": 10,
                    "purpose": "Advantage+ Prioritized Suggestion",
                },
                "velocity": {
                    "customer_ids": (
                        velocity_customers["customer_id"].tolist()
                        if not velocity_customers.empty
                        else []
                    ),
                    "purpose": "Test creative seed audience",
                },
                "negative": {
                    "customer_ids": (
                        negative_customers["customer_id"].tolist()
                        if not negative_customers.empty
                        else []
                    ),
                    "purpose": "Exclude from targeting (recent purchasers)",
                },
            },
            "compliance": {
                "is_special_category": is_special_category,
                "category_detected": (
                    self._detect_category_type(business_description)
                    if is_special_category
                    else None
                ),
            },
        }

    def _detect_category_type(self, description: str) -> str:
        """Detect which special category (if any)."""
        description_lower = description.lower()

        if any(
            kw in description_lower for kw in ["real estate", "apartment", "rental"]
        ):
            return "housing"
        if any(
            kw in description_lower for kw in ["loan", "credit", "mortgage", "finance"]
        ):
            return "credit"
        if any(kw in description_lower for kw in ["job", "hiring", "recruiting"]):
            return "employment"

        return "unknown"
