"""
Stage 1: The Miner - V1.8 Configurable Context Miner

Applies winner quantile to extract winners and losers, then extracts raw tags.
Uses MiningStrategySelector for intelligent quantile determination.
"""
import logging
import pandas as pd
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from ..config import MiningStrategySelector

logger = logging.getLogger(__name__)


# Existing 29 visual features
VISUAL_FEATURES = [
    "direction", "lighting_style", "primary_colors", "product_position",
    "lighting_type", "human_elements", "product_visibility", "visual_prominence",
    "color_balance", "temperature", "context_richness", "product_context",
    "relationship_depiction", "visual_flow", "composition_style", "depth_layers",
    "contrast_level", "color_saturation", "color_vibrancy", "background_content_type",
    "mood_lighting", "emotional_tone", "activity_level", "primary_focal_point",
    "framing", "architectural_elements_presence", "person_count",
    "person_relationship_type", "person_gender", "person_age_group",
    "person_activity", "text_elements", "cta_visuals", "problem_solution_narrative"
]


class AdMiner:
    """V1.8 Configurable Context Miner."""

    def __init__(
        self,
        input_config: dict,
        strategy_selector: MiningStrategySelector,
        customer_config: Optional[dict] = None
    ):
        """
        Initialize miner with input config and strategy selector.

        Args:
            input_config: Parsed input_config.yaml
            strategy_selector: MiningStrategySelector instance
            customer_config: Pre-loaded customer config (optional)
        """
        self.customer_id = input_config.get("customer_context", {}).get("customer_id", "")
        self.platform = input_config.get("customer_context", {}).get("platform", "")
        self.product = input_config.get("customer_context", {}).get("product")
        self.daily_budget_cents = input_config.get("campaign_context", {}).get("daily_budget_cents", 0)
        self.lookback_days = input_config.get("customer_context", {}).get("lookback_days", 90)

        self.strategy_selector = strategy_selector
        self.customer_config = customer_config

        self.selected_quantile: Optional[float] = None
        self.selection_reason: Optional[str] = None

    def determine_winner_quantile(self) -> float:
        """
        Determine winner quantile using strategy selector.

        Returns:
            Winner quantile threshold
        """
        mining_strategy = self._get_mining_strategy()

        manual_quantile = mining_strategy.get("winner_quantile")
        manual_profile = mining_strategy.get("mining_profile")

        self.selected_quantile = self.strategy_selector.determine_winner_quantile(
            customer_id=self.customer_id,
            platform=self.platform,
            product=self.product,
            daily_budget_cents=self.daily_budget_cents,
            manual_quantile=manual_quantile,
            manual_profile=manual_profile
        )

        logger.info(
            f"Selected winner quantile: {self.selected_quantile} "
            f"(Top {(1-self.selected_quantile)*100:.0f}%)"
        )

        return self.selected_quantile

    def _get_mining_strategy(self) -> dict:
        """Extract mining strategy from input config."""
        # Input config might have it at root or under mining_strategy key
        if "mining_strategy" in self._input_config:
            return self._input_config["mining_strategy"]
        return {}

    @property
    def _input_config(self) -> dict:
        """Get input config dict (for internal use)."""
        return {
            "customer_context": {
                "customer_id": self.customer_id,
                "platform": self.platform,
                "product": self.product
            },
            "campaign_context": {
                "daily_budget_cents": self.daily_budget_cents,
                "lookback_days": self.lookback_days
            }
        }

    def extract_winners_and_losers(
        self,
        df: pd.DataFrame,
        winner_quantile: Optional[float] = None,
        loser_quantile: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract winner and loser creatives based on quantile.

        Args:
            df: Creative features DataFrame
            winner_quantile: Winner quantile threshold (uses determined if None)
            loser_quantile: Loser quantile threshold (auto-calculated as 1-winner_quantile if None)

        Returns:
            (winners_df, losers_df)
        """
        if winner_quantile is None:
            winner_quantile = self.determine_winner_quantile()

        # Calculate loser quantile if not provided
        if loser_quantile is None:
            loser_quantile = 1 - winner_quantile
            logger.info(f"Auto-calculated loser quantile: {loser_quantile:.2f} (from winner_quantile {winner_quantile:.2f})")
        else:
            logger.info(f"Using configured loser quantile: {loser_quantile:.2f}")

        # Calculate quantile threshold
        roas_threshold = df["roas"].quantile(winner_quantile)

        # Split into winners and losers
        winners = df[df["roas"] >= roas_threshold].copy()
        losers = df[df["roas"] < df["roas"].quantile(loser_quantile)].copy()

        # Validate sample size
        min_sample_size = self.strategy_selector.get_min_sample_size(
            winner_quantile, self.customer_config
        )

        if len(winners) < min_sample_size:
            logger.warning(
                f"Insufficient winners: {len(winners)} < {min_sample_size}. "
                f"Consider lowering quantile or expanding lookback window."
            )

        logger.info(
            f"Extracted {len(winners)} winners (ROAS ≥ {roas_threshold:.2f}) "
            f"and {len(losers)} losers"
        )

        return winners, losers

    def extract_raw_tags(
        self,
        winners_df: pd.DataFrame
    ) -> Dict[str, List]:
        """
        Extract raw visual tags from winner images.

        Args:
            winners_df: Winner creatives DataFrame

        Returns:
            Dict mapping feature_name → list of unique values
        """
        raw_tags = {}

        for feature in VISUAL_FEATURES:
            if feature in winners_df.columns:
                # Get unique values, excluding nulls
                unique_values = winners_df[feature].dropna().unique().tolist()
                raw_tags[feature] = unique_values

        logger.info(f"Extracted raw tags from {len(raw_tags)} features")
        return raw_tags

    def run(
        self,
        df: pd.DataFrame,
        winner_quantile: Optional[float] = None,
        loser_quantile: Optional[float] = None
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, Dict[str, List]]:
        """
        Run complete mining pipeline.

        Args:
            df: Creative features DataFrame
            winner_quantile: Optional manual winner quantile override
            loser_quantile: Optional manual loser quantile override (auto-calculated if None)

        Returns:
            (winner_quantile, winners_df, losers_df, raw_tags)
        """
        logger.info("Starting Stage 1: The Miner")

        # Determine winner quantile
        if winner_quantile is None:
            winner_quantile = self.determine_winner_quantile()

        # Extract winners and losers
        winners, losers = self.extract_winners_and_losers(df, winner_quantile, loser_quantile)

        # Extract raw tags from winners
        raw_tags = self.extract_raw_tags(winners)

        logger.info("Stage 1: The Miner completed")

        return winner_quantile, winners, losers, raw_tags
