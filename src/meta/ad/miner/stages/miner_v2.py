"""
Stage 1: The Miner V2.0 - Enhanced with Psychology Tagging

Applies winner quantile to extract winners and losers, extracts raw tags,
AND extracts psychological triggers from winner creatives.

Enhancements over V1.8:
- Psychology tagging using VLM or rule-based analysis
- Automatic target psychology detection from winners
- Psychology-aware tag extraction
"""
import logging
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from collections import Counter
from pathlib import Path

from .miner import AdMiner, VISUAL_FEATURES
from ..config import MiningStrategySelector

logger = logging.getLogger(__name__)


# Psychology types
PSYCHOLOGY_TYPES = [
    "Trust_Authority",
    "Luxury_Aspiration",
    "FOMO",
    "Social_Proof"
]


class AdMinerV2(AdMiner):
    """
    V2.0 Enhanced Miner with Psychology Tagging.

    Extends V1.8 miner with psychological analysis capabilities:
    - Extracts psychology tags from winner creatives
    - Detects dominant psychology from winner pool
    - Enables psychology-driven filtering in downstream stages
    """

    # Psychology keyword mappings (rule-based fallback)
    PSYCHOLOGY_KEYWORDS = {
        "Trust_Authority": {
            "visual": ["clean", "minimalist", "white", "clinical", "professional", "sterile",
                      "studio", "cool", "blue", "laboratory", "expert", "medical"],
            "emotional": ["trust", "safe", "professional", "credible", "expert", "reliable"]
        },
        "Luxury_Aspiration": {
            "visual": ["marble", "gold", "premium", "elegant", "dramatic", "sophisticated",
                      "expensive", "luxury", "rich", "velvet", "granite"],
            "emotional": ["exclusive", "premium", "status", "achievement", "sophisticated",
                         "aspirational", "luxury"]
        },
        "FOMO": {
            "visual": ["urgent", "red", "contrast", "countdown", "limited", "timer",
                      "scarce", "deadline", "alert", "bright"],
            "emotional": ["urgent", "scarce", "limited", "now", "miss", "opportunity"]
        },
        "Social_Proof": {
            "visual": ["people", "lifestyle", "natural", "authentic", "candid", "crowd",
                      "together", "social", "community", "group"],
            "emotional": ["popular", "belonging", "validated", "community", "social",
                         "together", "everyone"]
        }
    }

    def __init__(
        self,
        input_config: dict,
        strategy_selector: MiningStrategySelector,
        customer_config: Optional[dict] = None,
        vlm_client: Optional[Any] = None
    ):
        """
        Initialize V2 miner with psychology capabilities.

        Args:
            input_config: Parsed input_config.yaml
            strategy_selector: MiningStrategySelector instance
            customer_config: Pre-loaded customer config (optional)
            vlm_client: Vision Language Model client for psychology extraction (optional)
        """
        super().__init__(input_config, strategy_selector, customer_config)

        self.vlm_client = vlm_client
        self.enable_psych_tagging = input_config.get("psychology_config", {}).get(
            "enable_psych_tagging", True
        )

    def extract_psychology_tags_rule_based(
        self,
        row: pd.Series
    ) -> Dict[str, Any]:
        """
        Extract psychology tags using rule-based analysis (fallback).

        Analyzes visual features to infer psychological triggers.

        Args:
            row: DataFrame row with creative features

        Returns:
            Psychology tags dict
        """
        psychology_scores = {}

        # Build text representation from visual features
        visual_text = ""
        for feature in VISUAL_FEATURES:
            if feature in row and pd.notna(row[feature]):
                visual_text += f" {row[feature]}"

        visual_text = visual_text.lower()

        # Score each psychology
        for psych_type in PSYCHOLOGY_TYPES:
            psych_key = psych_type.replace("-", "_").replace(" ", "_")

            keywords = self.PSYCHOLOGY_KEYWORDS[psych_key]

            # Count keyword matches
            visual_matches = sum(1 for kw in keywords["visual"] if kw in visual_text)

            # Calculate score
            psychology_scores[psych_type] = visual_matches

        # Find primary and secondary psychology
        sorted_psych = sorted(psychology_scores.items(), key=lambda x: x[1], reverse=True)

        primary_psych = sorted_psych[0][0] if sorted_psych[0][1] > 0 else "Trust_Authority"
        primary_score = sorted_psych[0][1]

        secondary_psych = sorted_psych[1][0] if len(sorted_psych) > 1 and sorted_psych[1][1] > 0 else None
        secondary_score = sorted_psych[1][1] if len(sorted_psych) > 1 else 0

        # Infer emotional tone
        emotional_tone = self._infer_emotional_tone(visual_text, primary_psych)

        # Calculate confidence (normalized by max possible score)
        confidence = min(1.0, primary_score / 5.0)  # Assume 5+ keywords = high confidence

        return {
            "primary": primary_psych,
            "secondary": secondary_psych,
            "primary_score": primary_score,
            "secondary_score": secondary_score,
            "emotional_tone": emotional_tone,
            "confidence": confidence,
            "method": "rule_based"
        }

    def extract_psychology_tags_vlm(
        self,
        image_url: str,
        visual_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract psychology tags using VLM (if available).

        Args:
            image_url: URL to creative image
            visual_features: Extracted visual features

        Returns:
            Psychology tags dict
        """
        if not self.vlm_client:
            logger.warning("VLM client not available, falling back to rule-based analysis")
            return self.extract_psychology_tags_rule_based(
                pd.Series(visual_features)
            )

        # Build psychology-focused prompt
        prompt = self._build_psychology_prompt(visual_features)

        try:
            # Call VLM
            response = self.vlm_client.analyze(image_url, prompt)

            # Parse response (implementation depends on VLM client)
            psychology_tags = self._parse_psychology_response(response)

            psychology_tags["method"] = "vlm"

            return psychology_tags

        except Exception as e:
            logger.error(f"VLM psychology extraction failed: {e}, falling back to rule-based")
            return self.extract_psychology_tags_rule_based(
                pd.Series(visual_features)
            )

    def _build_psychology_prompt(self, visual_features: Dict[str, Any]) -> str:
        """Build psychology-focused VLM prompt."""
        features_str = "\n".join([
            f"  - {k}: {v}" for k, v in visual_features.items()
            if k in VISUAL_FEATURES and pd.notna(v)
        ])

        return f"""
Analyze this advertisement image and extract:

1. CONFIRMED VISUAL FEATURES:
{features_str}

2. PSYCHOLOGICAL TRIGGERS:
   What primary emotion does this ad trigger?
   Options: Trust_Authority, Luxury_Aspiration, FOMO, Social_Proof

3. EMOTIONAL TONE:
   Describe the emotional tone in 2-3 words (underscore separated).
   Examples: "professional_calm", "premium_achievable", "urgent_excited",
             "authentic_warm", "sterile_cold", "dramatic_powerful"

4. BRAND PERCEPTION:
   How does this image make you perceive the brand?
   - Credible/Expert? (Trust_Authority)
   - Premium/Luxurious? (Luxury_Aspiration)
   - Urgent/Time-sensitive? (FOMO)
   - Popular/Validated? (Social_Proof)

5. TARGET AUDIENCE'S ASPIRATIONAL SELF:
   What version of themselves does the viewer aspire to be when seeing this ad?

Output JSON format:
{{
  "psychology_primary": "Trust_Authority|Luxury_Aspiration|FOMO|Social_Proof",
  "psychology_secondary": "Optional secondary psychology",
  "emotional_tone": "word1_word2",
  "psychology_confidence": 0.0-1.0,
  "reasoning": "Brief explanation..."
}}
"""

    def _parse_psychology_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into psychology tags."""
        # Implementation depends on VLM client format
        # This is a placeholder - actual implementation will vary
        try:
            import json
            parsed = json.loads(response)

            return {
                "primary": parsed.get("psychology_primary", "Trust_Authority"),
                "secondary": parsed.get("psychology_secondary"),
                "emotional_tone": parsed.get("emotional_tone", "professional_calm"),
                "confidence": parsed.get("psychology_confidence", 0.7),
                "reasoning": parsed.get("reasoning", "")
            }
        except:
            # Fallback if parsing fails
            logger.warning("Failed to parse VLM response, using defaults")
            return {
                "primary": "Trust_Authority",
                "secondary": None,
                "emotional_tone": "professional_calm",
                "confidence": 0.5,
                "reasoning": "VLM parsing failed"
            }

    def _infer_emotional_tone(self, visual_text: str, psychology: str) -> str:
        """Infer emotional tone from visual text and psychology."""
        tone_keywords = {
            "professional": ["clean", "minimalist", "studio", "professional"],
            "calm": ["cool", "soft", "gentle", "natural"],
            "premium": ["luxury", "premium", "elegant", "sophisticated"],
            "dramatic": ["dramatic", "bold", "contrast", "moody"],
            "urgent": ["urgent", "bright", "alert", "red"],
            "warm": ["warm", "golden", "soft", "natural"],
            "cold": ["cool", "sterile", "clinical", "white"],
            "authentic": ["natural", "authentic", "candid", "real"],
        }

        detected_tones = []
        for tone, keywords in tone_keywords.items():
            if any(kw in visual_text for kw in keywords):
                detected_tones.append(tone)

        if detected_tones:
            return f"{detected_tones[0]}_{detected_tones[1] if len(detected_tones) > 1 else 'balanced'}"

        # Psychology-based defaults
        default_tones = {
            "Trust_Authority": "professional_calm",
            "Luxury_Aspiration": "premium_dramatic",
            "FOMO": "urgent_excited",
            "Social_Proof": "authentic_warm"
        }

        return default_tones.get(psychology, "professional_balanced")

    def detect_target_psychology(
        self,
        winners_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Auto-detect dominant psychology from winner pool.

        Args:
            winners_df: Winner creatives with psychology tags

        Returns:
            Target psychology detection result
        """
        if "psychology_tags" not in winners_df.columns:
            logger.warning("No psychology tags found, cannot detect target psychology")
            return {
                "target_psychology": "Trust_Authority",
                "confidence": 0.0,
                "distribution": {}
            }

        # Extract primary psychology tags
        psychology_tags = winners_df["psychology_tags"].apply(
            lambda x: x.get("primary", "Unknown") if isinstance(x, dict) else "Unknown"
        ).tolist()

        # Count occurrences
        counts = Counter(psychology_tags)

        # Calculate distribution
        total = len(psychology_tags)
        distribution = {psych: {"count": count, "percentage": count/total*100}
                       for psych, count in counts.items()}

        # Find dominant psychology
        dominant, count = counts.most_common(1)[0]
        confidence = count / total

        result = {
            "target_psychology": dominant,
            "confidence": confidence,
            "distribution": distribution
        }

        logger.info(f"Auto-detected target psychology: {dominant} ({confidence:.1%} confidence)")
        logger.info(f"Psychology distribution: {distribution}")

        return result

    def run(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, Dict[str, List]]:
        """
        Run complete V2 mining pipeline with psychology tagging.

        Args:
            df: Creative features DataFrame

        Returns:
            (winner_quantile, winners_df, losers_df, raw_tags)
        """
        logger.info("Starting Stage 1: The Miner V2.0 (with Psychology)")

        # Run V1.8 visual extraction
        winner_quantile, winners, losers, raw_tags = super().run(df)

        # V2.0: Extract psychology tags from winners
        if self.enable_psych_tagging:
            logger.info("Extracting psychology tags from winners...")

            psychology_tags_list = []

            for idx, row in winners.iterrows():
                # Extract visual features
                visual_features = {}
                for feature in VISUAL_FEATURES:
                    if feature in row and pd.notna(row[feature]):
                        visual_features[feature] = row[feature]

                # Extract psychology tags
                if "image_url" in row and pd.notna(row["image_url"]):
                    # Use VLM if image URL available
                    psych_tags = self.extract_psychology_tags_vlm(
                        row["image_url"],
                        visual_features
                    )
                else:
                    # Fallback to rule-based
                    psych_tags = self.extract_psychology_tags_rule_based(row)

                psychology_tags_list.append(psych_tags)

            # Add psychology tags to winners dataframe
            winners["psychology_tags"] = psychology_tags_list

            # Aggregate psychology tags into raw_tags
            raw_tags["psychology_primary"] = [
                tags["primary"] for tags in psychology_tags_list
            ]

            raw_tags["psychology_secondary"] = [
                tags.get("secondary") for tags in psychology_tags_list
            ]

            raw_tags["emotional_tone"] = [
                tags["emotional_tone"] for tags in psychology_tags_list
            ]

            # Auto-detect target psychology
            psych_detection = self.detect_target_psychology(winners)
            raw_tags["_target_psychology_detection"] = psych_detection

            logger.info(
                f"Psychology tagging complete: "
                f"{len(set(raw_tags['psychology_primary']))} unique psychology types detected"
            )
        else:
            logger.info("Psychology tagging disabled")

        logger.info("Stage 1: The Miner V2.0 completed")

        return winner_quantile, winners, losers, raw_tags
